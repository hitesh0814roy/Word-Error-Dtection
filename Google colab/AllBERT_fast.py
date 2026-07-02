import pandas as pd
import json
import re
import torch

from spellchecker import SpellChecker

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)

# =====================================================
# ALL 4 BERT MODELS  --  FAST VOTING DETECTOR
# =====================================================
#   Detection only (no rewriting). Two speedups combined:
#
#   FIX 1  Spell-check GATE first  -> the expensive models run only
#          on words that are NOT real words. Real words such as
#          "antibody" are skipped before any model is called.
#
#   FIX 2  BATCHING  -> for each model, ALL candidate words are
#          scored in ONE forward pass, instead of one pass per word.
#
#   Together: ~50 words x 4 models = 200 passes  ->  4 passes total
#   (one batched pass per model). Same output as the gated voting
#   detector, far faster.
# =====================================================

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output\Detection_AllBERT_Fast.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# CONFIG
# =====================================================

SUSPICION_THRESHOLD = 0.0001

# A word is reported only if at least this many models flag it.
MIN_VOTES = 2

MIN_WORD_LENGTH = 1

HL_OPEN = "<<"
HL_CLOSE = ">>"

BERT_MODELS = {
    "BioClinicalBERT": "emilyalsentzer/Bio_ClinicalBERT",
    "PubMedBERT":      "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
    "ClinicalBERT":    "medicalai/ClinicalBERT",
    "SapBERT":         "cambridgeltl/SapBERT-from-PubMedBERT-fulltext",
}

# =====================================================
# SPELL-CHECKER GATE
# =====================================================

spell = SpellChecker()


def is_real_word(clean_word):
    """True if the spell-checker recognises this as a real word."""
    return len(spell.unknown([clean_word.lower()])) == 0

# =====================================================
# LOAD MODELS
# =====================================================

bert = {}

for label, name in BERT_MODELS.items():
    print(f"Loading {label} ({name})...")
    tok = AutoTokenizer.from_pretrained(name)
    mdl = AutoModelForMaskedLM.from_pretrained(name)
    mdl.eval()
    bert[label] = (tok, mdl)

print("All models loaded.\n")

# =====================================================
# BATCHED SCORING FOR ONE MODEL
# =====================================================

def score_candidates_batched(tokenizer, model, plain_words, candidates):
    """Score every candidate word for ONE model in a single batched
    forward pass.

    candidates: list of (word_index, clean_word)
    Returns: {word_index: probability}
    """

    if not candidates:
        return {}

    masked_sentences = []
    per_candidate_tokens = []   # original subword ids for each candidate

    for idx, clean_word in candidates:

        word_token_ids = tokenizer(
            clean_word,
            add_special_tokens=False
        )["input_ids"]

        n_tokens = max(1, len(word_token_ids))

        masked = plain_words.copy()
        masked[idx] = " ".join([tokenizer.mask_token] * n_tokens)

        masked_sentences.append(" ".join(masked))
        per_candidate_tokens.append(word_token_ids)

    # ONE batched tokenization + ONE forward pass for ALL candidates.
    inputs = tokenizer(
        masked_sentences,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True
    )

    with torch.no_grad():
        logits = model(**inputs).logits          # (batch, seq, vocab)

    probs = torch.softmax(logits, dim=-1)

    results = {}

    for row, (idx, _clean_word) in enumerate(candidates):

        word_token_ids = per_candidate_tokens[row]

        if len(word_token_ids) == 0:
            continue

        # Mask positions in THIS row.
        mask_positions = (
            inputs["input_ids"][row] == tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            continue

        usable = min(len(word_token_ids), len(mask_positions))

        token_probs = []
        for k in range(usable):
            pos = mask_positions[k].item()
            tid = word_token_ids[k]
            token_probs.append(probs[row, pos, tid].item())

        if token_probs:
            results[idx] = sum(token_probs) / len(token_probs)

    return results

# =====================================================
# DETECTION  (gate -> batched scoring -> vote)
# =====================================================

def detect(text):

    word_spans = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]

    plain_words = [w for (w, _s, _e) in word_spans]

    # STEP 1 - GATE: keep only non-words as candidates.
    candidates = []
    for i, (raw_word, _s, _e) in enumerate(word_spans):
        clean_word = re.sub(r"[^A-Za-z]", "", raw_word)
        if len(clean_word) < MIN_WORD_LENGTH:
            continue
        if is_real_word(clean_word):
            continue
        candidates.append((i, clean_word))

    # STEP 2 - each model scores all candidates in ONE batched pass.
    # tally: word_index -> {model_label: confidence}
    tally = {}

    for label, (tok, mdl) in bert.items():
        scores = score_candidates_batched(
            tok, mdl, plain_words, candidates
        )
        for idx, prob in scores.items():
            if prob < SUSPICION_THRESHOLD:
                tally.setdefault(idx, {})[label] = 1.0 - prob

    # STEP 3 - vote + merge.
    detections = []

    for idx, model_confs in tally.items():

        votes = len(model_confs)

        if votes < MIN_VOTES:
            continue

        raw_word, start, end = word_spans[idx]
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

    detections.sort(key=lambda d: d["start"])

    return detections


def highlight_paragraph(text, detections):
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
# PER-ROW PROCESSING
# =====================================================

def process_text(text):

    if pd.isna(text):
        return {
            "highlighted_text": "",
            "flagged_words": [],
            "combined_confidence": 0.0
        }

    text = str(text)

    detections = detect(text)

    if detections:
        overall = sum(
            d["combined_confidence"] for d in detections
        ) / len(detections)
    else:
        overall = 0.0

    return {
        "highlighted_text": highlight_paragraph(text, detections),
        "flagged_words": detections,
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

    print(
        f"Row {index + 1}: "
        f"{len(result['flagged_words'])} word(s) flagged "
        f"(>= {MIN_VOTES} votes), "
        f"combined confidence = {result['combined_confidence']}"
    )

# =====================================================
# SAVE OUTPUT
# =====================================================

df["all_bert_detection"] = outputs

df.to_excel(OUTPUT_FILE, index=False)

print("\nAll-BERT FAST voting detection completed!")
print(f"Saved output file: {OUTPUT_FILE}")
