import pandas as pd
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)

# =====================================================
# MODEL  :  BioClinicalBERT
# TASK   :  Detect (NOT correct) incorrect medical words
# METHOD :  Pure model-driven. No predefined word lists.
#           The model reads the whole paragraph, and for each
#           word decides on its own whether that word is likely
#           wrong in context.
# =====================================================

MODEL_LABEL = "PubMedBERT"
MODEL_NAME = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Detection_PubMedBERT.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# DETECTION CONFIG
# =====================================================

# The model masks each word inside the full paragraph and looks at
# the probability it assigns to the ORIGINAL word. If that
# probability is below this threshold, the model considers the word
# unlikely / wrong in context. Raise it to flag MORE words, lower it
# to flag FEWER.
SUSPICION_THRESHOLD = 0.001

# Ignore very short tokens (the, of, mg, etc.)
MIN_WORD_LENGTH = 4

# Markers used to highlight wrong words inside the paragraph.
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
# DETECTION (pure masked-language-model)
# =====================================================

def word_probability(plain_words, index, clean_word):
    """Mask the target word inside the FULL paragraph and return the
    probability the model assigns to the original word in context.
    Lower probability  ->  the model finds the word less likely."""

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
        return None

    mask_index = mask_positions[0].item()

    with torch.no_grad():
        logits = model(**inputs).logits

    probs = torch.softmax(logits[0, mask_index], dim=-1)

    original_ids = tokenizer(
        clean_word,
        add_special_tokens=False
    )["input_ids"]

    if len(original_ids) == 0:
        return None

    # Average the probability across the word's subword pieces.
    token_probs = [probs[tid].item() for tid in original_ids]
    return sum(token_probs) / len(token_probs)


def detect_incorrect_words(text):
    """Let the model decide which words are wrong, using the whole
    paragraph as context. Returns detections with word, start, end,
    detection_confidence."""

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

        # The model itself decides: low probability -> wrong word.
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
    the highlight markers, using the detected start/end positions."""

    highlighted = text

    # Insert from the end so earlier indices stay valid.
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

print(f"\n{MODEL_LABEL} detection completed!")
print(f"Saved output file: {OUTPUT_FILE}")
