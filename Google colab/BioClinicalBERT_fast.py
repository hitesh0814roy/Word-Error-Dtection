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
# MODEL  :  BioClinicalBERT  --  FAST VERSION (Fix 1)
# TASK   :  Detect (NOT correct) incorrect medical words
# SPEEDUP:  Gate FIRST, then run the model only on candidates.
#
#   The slow part was running one model forward-pass for EVERY
#   word. But only NON-words can ever be flagged, so we first do a
#   cheap spell-check and run the (expensive) model ONLY on the few
#   words that are not real words. Real words like "antibody" are
#   skipped before the model is ever called.
#
#   ~50 words -> maybe 3 model calls instead of 50. Same output as
#   the gated detector, just much faster.
# =====================================================

MODEL_LABEL = "BioClinicalBERT"
MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output\Detection_BioClinicalBERT_Fast.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# DETECTION CONFIG
# =====================================================

# Among NON-words, the model still scores how unlikely each is.
SUSPICION_THRESHOLD = 0.0001

# Ignore very short tokens.
MIN_WORD_LENGTH = 1

HL_OPEN = "<<"
HL_CLOSE = ">>"

# =====================================================
# SPELL-CHECKER GATE  (the cheap pre-filter)
# =====================================================

spell = SpellChecker()


def is_real_word(clean_word):
    """True if the spell-checker recognises this as a real word.
    Real words are skipped before the model ever runs."""
    return len(spell.unknown([clean_word.lower()])) == 0

# =====================================================
# LOAD DETECTION MODEL
# =====================================================

print(f"Loading {MODEL_LABEL} ({MODEL_NAME})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForMaskedLM.from_pretrained(MODEL_NAME)
model.eval()

print("Model loaded.")

# =====================================================
# MODEL SCORING (only called for candidate non-words)
# =====================================================

def word_probability(plain_words, index, clean_word):
    """Mask the target word inside the FULL paragraph using the
    correct number of [MASK] tokens, then average the model's
    predicted probability of each original subword piece.
    Lower probability -> the model finds the word less likely."""

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

# =====================================================
# DETECTION  (gate first, then model)
# =====================================================

def detect_incorrect_words(text):
    """Gate every word with the spell-checker FIRST. Only words that
    are not real words are scored by the model."""

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

        # CHEAP GATE: skip real words before touching the model.
        if is_real_word(clean_word):
            continue

        # Only non-words reach the (expensive) model.
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

print(f"\n{MODEL_LABEL} FAST detection completed!")
print(f"Saved output file: {OUTPUT_FILE}")
