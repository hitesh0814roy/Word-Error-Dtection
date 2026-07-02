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
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\output\Medical_Detection_New.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# DETECTION CONFIG
# =====================================================

# Ignore very short tokens (the, of, mg, etc.)
MIN_WORD_LENGTH = 4

# Known typos are ALWAYS flagged with full confidence.
KNOWN_ERRORS = {
    "tengu", "degu", "platlets", "hedache", "feever",
    "tigtness", "thyriod", "anemea", "gestonal",
    "hypertention", "vomitting", "abdmonal", "tachycardya",
}

# Correct medical terms that the plain English spell-checker does
# NOT know. These must NOT be flagged as wrong. Extend this list
# as needed for your data.
MEDICAL_WHITELIST = {
    "dengue", "cbc", "ecg", "abg", "igrt", "vmat", "mmrt",
    "crt", "aldosterone", "cortisol", "urinary", "antibody",
    "antibodies", "titer", "titre", "saline", "gargles",
    "panel", "thyroid", "anemia", "hypertension", "gestational",
    "tachycardia", "platelets", "abdominal", "vomiting",
    "headache", "tightness", "screening", "identification",
    "pregnant", "suspected", "expansion",
}

# =====================================================
# SPELL CHECKER  (decides what is a real word)
# =====================================================

spell = SpellChecker()

# =====================================================
# LOAD DETECTION MODEL  (Bio_ClinicalBERT)
# =====================================================

print("Loading Bio_ClinicalBERT (detection model)...")

DETECTION_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

tokenizer = AutoTokenizer.from_pretrained(DETECTION_MODEL_NAME)

model = AutoModelForMaskedLM.from_pretrained(DETECTION_MODEL_NAME)

model.eval()

print("Detection model loaded.")

# =====================================================
# DETECTION FUNCTION
# =====================================================

def is_wrong_word(clean_word):
    """
    Decide whether a word is actually WRONG (a real misspelling), 
    whether the words is medically wrong and 
    whether the word is wrong in the actual sense of the entire paragraph,
    not just a rare-but-valid medical term.

    A word is considered wrong only if:
      - it is in the KNOWN_ERRORS list, OR
      - the spell-checker does not recognise it AND it is not a
        known correct medical term (whitelist).

    Correct words (English or whitelisted medical) return False,
    so they are NOT flagged.
    """

    lower = clean_word.lower()

    if lower in KNOWN_ERRORS:
        return True

    if lower in MEDICAL_WHITELIST:
        return False

    # spell.unknown() returns the words it does NOT recognise.
    return len(spell.unknown([lower])) > 0


def model_confidence(plain_words, index, clean_word):
    """
    Use Bio_ClinicalBERT to score HOW confident we are that the
    flagged word is wrong, given its clinical context.
    Returns a value in 0-1 (1 - probability the model gives the
    original word). Falls back to 1.0 if it cannot be scored.
    """

    masked = plain_words.copy()
    masked[index] = tokenizer.mask_token
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
        return 1.0

    mask_index = mask_positions[0].item()

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits[0, mask_index], dim=-1)

    original_ids = tokenizer(
        clean_word,
        add_special_tokens=False
    )["input_ids"]

    if len(original_ids) == 0:
        return 1.0

    original_prob = probs[original_ids[0]].item()

    return 1.0 - original_prob


def detect_incorrect_words(text):
    """
    Detect ONLY the wrong (misspelled / invalid) medical words.

    Returns a list of detections, each with:
      word, start (char index), end (char index), and
      detection_confidence (0-1, how confident the word is wrong).
    """

    detections = []

    # Walk through every word AND keep its character offsets so we
    # can report exact start / end positions in the original text.
    word_spans = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]

    plain_words = [w for (w, _s, _e) in word_spans]

    for i, (raw_word, start, end) in enumerate(word_spans):

        clean_word = re.sub(r"[^A-Za-z]", "", raw_word)

        if len(clean_word) < MIN_WORD_LENGTH:
            continue

        # Only proceed if the word is genuinely WRONG.
        if not is_wrong_word(clean_word):
            continue

        # Known typos are certain -> confidence 1.0.
        # Otherwise, ask the clinical model how confident it is.
        if clean_word.lower() in KNOWN_ERRORS:
            confidence = 1.0
        else:
            confidence = model_confidence(plain_words, i, clean_word)

        detections.append({
            "word": raw_word,
            "start": start,
            "end": end,
            "detection_confidence": round(confidence, 6)
        })

    return detections

# =====================================================
# PER-ROW PROCESSING
# =====================================================

def process_text(text):

    if pd.isna(text):
        return {
            "detected_words": [],
            "combined_confidence": 0.0
        }

    text = str(text)

    detections = detect_incorrect_words(text)

    # Combined accuracy score = average detection confidence
    # across all flagged words (0 if nothing flagged).
    if detections:
        combined = sum(
            d["detection_confidence"] for d in detections
        ) / len(detections)
    else:
        combined = 0.0

    return {
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

    outputs.append(
        json.dumps(result, ensure_ascii=False)
    )

    print(
        f"Row {index + 1}: "
        f"{len(result['detected_words'])} word(s) flagged, "
        f"combined confidence = {result['combined_confidence']}"
    )

# =====================================================
# SAVE OUTPUT
# =====================================================

df["detection"] = outputs

df.to_excel(OUTPUT_FILE, index=False)

print("\nDetection completed!")
print(f"Saved output file: {OUTPUT_FILE}")
