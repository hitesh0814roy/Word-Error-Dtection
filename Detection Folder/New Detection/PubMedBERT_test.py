import pandas as pd
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)

# =====================================================
# MODEL  :  BioClinicalBERT  --  TEST VERSION
# TASK   :  Detect (NOT correct) incorrect medical words
# CHANGE :  Fixes the multi-subword masking bug.
#
#   The original masked ONE [MASK] for a whole word, then read the
#   probability of ALL its subword pieces at that single position.
#   Multi-subword words (e.g. "antibody" -> "anti" + "##body")
#   therefore scored artificially low and were wrongly flagged.
#
#   This version masks the CORRECT number of [MASK] tokens for the
#   word and scores the model's actual prediction at each position,
#   then averages. Correct multi-token words are no longer falsely
#   flagged. Still fully model-driven (no predefined word lists).
# =====================================================

MODEL_LABEL = "PubMedBERT"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output\Detection_PubMedBERT_Test.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# DETECTION CONFIG
# =====================================================

# If the model's average probability for the original word (across
# all its subword pieces) is below this, the model considers the
# word unlikely / wrong in context. Raise -> flag more, lower -> flag fewer.
SUSPICION_THRESHOLD = 0.0001

# Ignore very short tokens (the, of, mg, etc.)
MIN_WORD_LENGTH = 1

HL_OPEN = "<<"
HL_CLOSE = ">>"

# =====================================================
# LOAD DETECTION MODEL
# =====================================================

print(f"Loading {MODEL_LABEL} ({MODEL_NAME})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

print("Model loaded.")

# =====================================================
# DETECTION (masked-language-model, multi-subword aware)
# =====================================================

def word_probability(plain_words, index, clean_word):
    """Mask the target word inside the FULL paragraph using the
    CORRECT number of [MASK] tokens (one per subword piece), then
    average the model's predicted probability of each original
    subword at its own masked position.

    Returns the averaged probability (lower -> less likely word),
    or None if it cannot be scored."""

    # How many subword tokens does this word actually use?
    word_token_ids = tokenizer(
        clean_word,
        add_special_tokens=False
    )["input_ids"]

    n_tokens = len(word_token_ids)

    if n_tokens == 0:
        return None

    # Replace the word with n_tokens mask tokens.
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

    # Score each subword at ITS OWN masked position.
    usable = min(n_tokens, len(mask_positions))

    token_probs = []
    for k in range(usable):
        pos = mask_positions[k].item()
        tid = word_token_ids[k]
        token_probs.append(probs[pos, tid].item())

    if not token_probs:
        return None

    return sum(token_probs) / len(token_probs)


def detect_incorrect_words(text):
    """Let the model decide which words are wrong, using the whole
    paragraph as context."""

    detections = []

    word_spans = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]

    plain_words = [w for (w, _s, _e) in word_spans]

    for i, (raw_word, start, end) in enumerate(word_spans):

        clean_word = re.sub(r"[^A-Za-z]", "", raw_word)

        if len(clean_word) < MIN_WORD_LENGTH:
            continue

        prob = word_probability(plain_words, i, clean_word)

        if prob is None:
            continue

        if prob < SUSPICION_THRESHOLD:
            detections.append({
                "word": raw_word,
                "start": start,
                "end": end,
                "detection_confidence": round(1.0 - prob, 6)
            })

    return detections


def highlight_paragraph(text, detections):
    """Return the full paragraph with every wrong word wrapped in
    the highlight markers."""

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
            "detected_words": [],
            "combined_confidence": 0.0
        }

    text = str(text)

    detections = detect_incorrect_words(text)

    if detections:
        combined = sum(
            d["detection_confidence"] for d in detections
        ) / len(detections)
    else:
        combined = 0.0

    return {
        "highlighted_text": highlight_paragraph(text, detections),
        "detected_words": detections,
        "combined_confidence": round(combined, 6)
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
        f"{len(result['detected_words'])} word(s) flagged, "
        f"combined confidence = {result['combined_confidence']}"
    )

# =====================================================
# SAVE OUTPUT
# =====================================================

df[f"{MODEL_LABEL}_detection"] = outputs

df.to_excel(OUTPUT_FILE, index=False)

print(f"\n{MODEL_LABEL} TEST detection completed!")
print(f"Saved output file: {OUTPUT_FILE}")
