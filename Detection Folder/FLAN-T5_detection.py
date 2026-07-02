import pandas as pd
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =====================================================
# MODEL  :  FLAN-T5  (generative)
# TASK   :  Detect (NOT correct) incorrect medical words
# METHOD :  Pure model-driven. No predefined word lists.
#           FLAN-T5 READS the entire paragraph and decides on its
#           own which words are misspelled / medically incorrect.
#           We then locate those words in the text.
# =====================================================

MODEL_LABEL = "FLAN-T5"
MODEL_NAME = "google/flan-t5-base"

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output\Detection_FLAN-T5_Test.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# DETECTION CONFIG
# =====================================================

MIN_WORD_LENGTH = 1

HL_OPEN = "<<"
HL_CLOSE = ">>"

# =====================================================
# LOAD MODEL
# =====================================================

print(f"Loading {MODEL_LABEL} ({MODEL_NAME})...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

print("Model loaded.")

# =====================================================
# STEP 1 - FLAN-T5 READS THE WHOLE PARAGRAPH
# =====================================================

def flan_listed_wrong_words(text):
    """Ask FLAN-T5 to read the entire paragraph and decide on its
    own which words are misspelled or medically incorrect.
    Returns a lowercase set of those words."""

    prompt = f"""Read the following medical paragraph.
List ONLY the words that are misspelled , medically incorrect or does not make any sense.
Skip the words that are misspelled but correct in gramatical sense to the model.
Return them as a comma-separated list. Do not correct them.

Paragraph:
{text}
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            num_beams=4,
            early_stopping=True
        )

    listed = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Split on commas / whitespace into clean lowercase tokens.
    words = re.split(r"[,\s]+", listed.lower())

    return {re.sub(r"[^a-z]", "", w) for w in words if w}

# =====================================================
# STEP 2 - LOCATE THE WRONG WORDS IN THE TEXT
# =====================================================

def detect_incorrect_words(text, flan_words):
    """Locate, in the original paragraph, the words FLAN-T5 flagged.
    Remove the flag words that are not misspelled
    Returns detections with word, start, end, detection_confidence."""

    detections = []

    word_spans = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]

    for raw_word, start, end in word_spans:

        clean_word = re.sub(r"[^A-Za-z]", "", raw_word)

        if len(clean_word) < MIN_WORD_LENGTH:
            continue

        if clean_word.lower() not in flan_words:
            continue

        detections.append({
            "word": raw_word,
            "start": start,
            "end": end,
            # FLAN-T5 does not expose token probabilities, so the
            # detection is binary: the model flagged it -> 1.0.
            "detection_confidence": 1.0
        })

    return detections


def highlight_paragraph(text, detections):
    """Wrap each flagged word in highlight markers."""

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

    flan_words = flan_listed_wrong_words(text)
    detections = detect_incorrect_words(text, flan_words)

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
