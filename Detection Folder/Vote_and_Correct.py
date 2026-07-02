import pandas as pd
import json
import re
import torch

from spellchecker import SpellChecker

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM
)

# =====================================================
# DETECT-THEN-CORRECT  PIPELINE
# =====================================================
#   STEP 1  4 BERT models each independently DETECT wrong words
#           (masked-language-model, multi-subword aware).
#   STEP 2  Their opinions are MERGED by voting + averaged
#           confidence (like Combined_result.py).
#   STEP 3  FLAN-T5 does ONE correction pass, focused on the words
#           the BERT models agreed were wrong.
#
#   The BERT encoders only DETECT (they cannot rewrite text).
#   FLAN-T5 is the only model that actually CORRECTS.
# =====================================================

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"medical_checker.xlsx"

OUTPUT_FILE = r"output\Vote_and_Correct_New.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# CONFIG
# =====================================================

# A word is judged wrong by a single BERT model if its in-context
# probability is below this.
SUSPICION_THRESHOLD = 0.001

# A word is passed to FLAN-T5 for correction only if at least this
# many BERT models flagged it (the "vote").
MIN_VOTES = 2

# Ignore very short tokens.
MIN_WORD_LENGTH = 0

HL_OPEN = "<<"
HL_CLOSE = ">>"

# =====================================================
# SPELL-CHECKER GATE
# =====================================================
# A masked LM measures how PREDICTABLE a word is, not whether it is
# a real word. In fragmented text it finds almost every word
# improbable, so it flags correct words too. To stop that, a word
# only becomes a CANDIDATE if the spell-checker does not recognise
# it as a real word. The BERT models then only score confidence
# among genuine non-words. This is not a hand-written list - it is a
# full English vocabulary of hundreds of thousands of words.

spell = SpellChecker()


def is_real_word(clean_word):
    """True if the spell-checker recognises this as a real word."""
    return len(spell.unknown([clean_word.lower()])) == 0

# =====================================================
# DETECTION MODELS  (the 4 voters)
# =====================================================

BERT_MODELS = {
    "BioClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "PubMedBERT":      "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "ClinicalBERT":    "medicalai/ClinicalBERT",
    "SapBERT":         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}

# =====================================================
# CORRECTION MODEL
# =====================================================

CORRECTION_MODEL_NAME = "google/flan-t5-base"

# =====================================================
# LOAD EVERYTHING
# =====================================================

bert = {}

for label, name in BERT_MODELS.items():
    print(f"Loading detector {label} ({name})...")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForMaskedLM.from_pretrained(name)
    mdl.eval()
    bert[label] = (tok, mdl)

print(f"Loading corrector FLAN-T5 ({CORRECTION_MODEL_NAME})...")
t5_tokenizer = AutoTokenizer.from_pretrained(CORRECTION_MODEL_NAME)
t5_model = AutoModelForSeq2SeqLM.from_pretrained(CORRECTION_MODEL_NAME)
t5_model.eval()

print("All models loaded.\n")

# =====================================================
# STEP 1  -  ONE BERT MODEL DETECTS  (multi-subword aware)
# =====================================================

def word_probability(tokenizer, model, plain_words, index, clean_word):
    """Mask the target word (with the correct number of [MASK]
    tokens) inside the full paragraph and return the model's average
    probability for the original word. Lower -> less likely. 
    The model shuld skip the words from which full stop in connected,
    or commas are connected, it should only focus words and do not include delimmeters
    and still they make sense"""

    word_token_ids = tokenizer(
        clean_word,
        add_special_tokens=False
    )["input_ids"]

    n_tokens = len(word_token_ids)

    if n_tokens == 0:
        return None

    masked = plain_words.copy()
    masked[index] = " ".join([tokenizer.mask_token] * n_tokens)
    masked_sentence = " ".join(masked)

    inputs = tokenizer(
        masked_sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    mask_positions = (
        inputs["input_ids"][0] == tokenizer.mask_token_id
    ).nonzero(as_tuple=True)[0]

    if len(mask_positions) == 0:
        return None

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits[0], dim=-1)

    usable = min(n_tokens, len(mask_positions))

    token_probs = []
    for k in range(usable):
        pos = mask_positions[k].item()
        tid = word_token_ids[k]
        token_probs.append(probs[pos, tid].item())

    if not token_probs:
        return None

    return sum(token_probs) / len(token_probs)


def detect_with_model(tokenizer, model, word_spans):
    """Return {word_index: confidence} for words this model flags."""

    plain_words = [w for (w, _s, _e) in word_spans]

    flagged = {}

    for i, (raw_word, _start, _end) in enumerate(word_spans):

        clean_word = re.sub(r"[^A-Za-z]", "", raw_word)

        if len(clean_word) < MIN_WORD_LENGTH:
            continue

        # GATE: only non-words (real misspellings) are candidates.
        # Real words like "antibody", "dengue", "saline" are skipped.
        if is_real_word(clean_word):
            continue

        prob = word_probability(
            tokenizer, model, plain_words, i, clean_word
        )

        if prob is None:
            continue

        if prob < SUSPICION_THRESHOLD:
            flagged[i] = 1.0 - prob

    return flagged

# =====================================================
# STEP 2  -  MERGE THE 4 OPINIONS  (vote + average)
# =====================================================

def vote(word_spans):
    """Run all BERT models, then merge. Returns a list of agreed
    detections: word, start, end, votes, model confidences,
    and merged combined_confidence.
    The model shuld skip the words from which full stop in connected,
    or commas are connected, it should only focus words and do not include delimmeters
    and still they make sense"""

    # word_index -> {model_label: confidence}
    tally = {}

    for label, (tok, mdl) in bert.items():
        flagged = detect_with_model(tok, mdl, word_spans)
        for idx, conf in flagged.items():
            tally.setdefault(idx, {})[label] = conf

    detections = []

    for idx, model_confs in tally.items():

        votes = len(model_confs)

        if votes < MIN_VOTES:
            continue

        raw_word, start, end = word_spans[idx]

        # Combined confidence = average over the models that flagged it.
        combined = sum(model_confs.values()) / votes

        detections.append({
            "word": raw_word,
            "start": start,
            "end": end,
            "votes": votes,
            "model_confidences": {
                k: round(v, 6) for k, v in model_confs.items()
            },
            "combined_confidence": round(combined, 6)
        })

    # Sort by position for readable highlighting.
    detections.sort(key=lambda d: d["start"])

    return detections


def highlight_paragraph(text, detections):
    """Wrap each agreed-wrong word in highlight markers."""

    highlighted = text

    for d in sorted(detections, key=lambda x: x["start"], reverse=True):
        s, e = d["start"], d["end"]
        highlighted = (
            highlighted[:s]
            + HL_OPEN + highlighted[s:e] + HL_CLOSE
            + highlighted[e:]
        )

    return highlighted

# =====================================================
# STEP 3  -  FLAN-T5 VERIFIES  (does NOT rewrite the paragraph)
# =====================================================

def verify_word_with_flan_t5(text, word):
    """Ask FLAN-T5 whether ONE flagged word is really medically or
    grammatically wrong, given the full paragraph as context.
    Returns ("yes"/"no"/"unsure", raw_answer). FLAN-T5 does NOT
    generate a corrected paragraph - it only judges the word."""

    prompt = f"""You are a medical language reviewer.
Read the paragraph below. A word from it has been flagged as
possibly wrong.

Decide whether the word "{word}" is actually wrong in this
paragraph - that is, misspelled, medically incorrect, or
grammatically incorrect.

Answer with only one word: "yes" if it is wrong, "no" if it is
correct.

Paragraph:
{text}

Is "{word}" wrong? Answer yes or no.
"""

    inputs = t5_tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = t5_model.generate(
            **inputs,
            max_new_tokens=5,
            num_beams=4,
            early_stopping=True
        )

    answer = t5_tokenizer.decode(
        outputs[0], skip_special_tokens=True
    ).strip().lower()

    if "yes" in answer:
        verdict = "yes"
    elif "no" in answer:
        verdict = "no"
    else:
        verdict = "unsure"

    return verdict, answer


def verify_flagged_words(text, detections):
    """Run FLAN-T5 over every BERT-flagged word and attach its
    verdict. Returns a new list with 'flan_verdict' added to each."""

    verified = []

    for d in detections:
        verdict, raw = verify_word_with_flan_t5(text, d["word"])

        item = dict(d)
        item["flan_verdict"] = verdict          # yes / no / unsure
        item["flan_answer"] = raw               # raw model output
        item["really_wrong"] = (verdict == "yes")

        verified.append(item)

    return verified

# =====================================================
# FULL PER-ROW PIPELINE
# =====================================================

def process_text(text):

    if pd.isna(text):
        return {
            "highlighted_text": "",
            "flagged_words": [],
            "confirmed_wrong_words": [],
            "combined_confidence": 0.0
        }

    text = str(text)

    word_spans = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]

    # STEP 1 + 2: BERT models detect and merge votes
    detections = vote(word_spans)

    # STEP 3: FLAN-T5 verifies each flagged word (no rewriting)
    verified = verify_flagged_words(text, detections)

    # The words FLAN-T5 confirmed are really wrong.
    confirmed = [v["word"] for v in verified if v["really_wrong"]]

    # Overall combined confidence across agreed words.
    if verified:
        overall = sum(
            v["combined_confidence"] for v in verified
        ) / len(verified)
    else:
        overall = 0.0

    return {
        "highlighted_text": highlight_paragraph(text, detections),
        # "flagged_words": verified,
        # "confirmed_wrong_words": confirmed,
        "combined_confidence": round(overall, 6)
    }

# =====================================================
# READ EXCEL FILE
# =====================================================

print(f"Reading Excel file: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

if INPUT_COLUMN not in df.columns:
    raise Exception(
        f"Excel file must contain '{INPUT_COLUMN}' column"
    )

# =====================================================
# PROCESS ROWS
# =====================================================

outputs = []

for index, row in df.iterrows():

    result = process_text(row[INPUT_COLUMN])

    outputs.append(json.dumps(result, ensure_ascii=False))

    # print(
    #     f"Row {index + 1}: "
    #     f"{len(result['flagged_words'])} flagged by BERT "
    #     f"(>= {MIN_VOTES} votes), "
    #     f"{len(result['confirmed_wrong_words'])} confirmed wrong by FLAN-T5"
    # )

# =====================================================
# SAVE OUTPUT
# =====================================================

df["vote_and_verify"] = outputs

df.to_excel(OUTPUT_FILE, index=False)

print("\nDetect (4 BERT vote) -> Verify (FLAN-T5, no rewriting) completed!")
print(f"Saved output file: {OUTPUT_FILE}")
