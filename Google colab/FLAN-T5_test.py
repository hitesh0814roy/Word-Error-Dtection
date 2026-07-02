import pandas as pd
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =====================================================
# MODEL  :  FLAN-T5  (generative)  --  TEST VERSION
# TASK   :  Detect (NOT correct) incorrect medical words
# CHANGE :  Adds a SECOND verification pass. After FLAN-T5 lists
#           the words it thinks are wrong, every candidate is sent
#           back to the model with a yes/no question:
#               "Is <word> a correctly spelled, valid word?"
#           Words the model confirms are valid (e.g. "antibody")
#           are dropped. This removes false positives, while
#           staying fully model-driven (no predefined word lists).
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
# GENERIC FLAN-T5 CALL
# =====================================================

def ask_flan(prompt, max_new_tokens=128):
    """Run a prompt through FLAN-T5 and return the decoded text."""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=4,
            early_stopping=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =====================================================
# STEP 1 - FLAN-T5 READS THE WHOLE PARAGRAPH
# =====================================================

def flan_listed_wrong_words(text):
    """Ask FLAN-T5 to read the entire paragraph and decide on its
    own which words are misspelled or medically incorrect.
    Returns a lowercase set of candidate words."""

    prompt = f"""Read the following medical paragraph.
List ONLY the words that are misspelled or medically incorrect.
Return them as a comma-separated list. Do not correct them.

Paragraph:
{text}
"""

    listed = ask_flan(prompt)

    words = re.split(r"[,\s]+", listed.lower())

    return {re.sub(r"[^a-z]", "", w) for w in words if w}

# =====================================================
# STEP 2 - VERIFY EACH CANDIDATE  (removes false positives)
# =====================================================

def flan_says_word_is_wrong(word):
    """Ask FLAN-T5 to confirm whether a single word is actually
    wrong. Returns True only if the model says it is NOT valid.
    This drops false positives such as 'antibody'."""

    prompt = f"""Is "{word}" a correctly spelled, valid English or medical word?
Answer only "yes" or "no"."""

    answer = ask_flan(prompt, max_new_tokens=5).strip().lower()

    # If the model says it is NOT a valid word -> it is wrong.
    if "no" in answer:
        return True

    if "yes" in answer:
        return False

    # Ambiguous answer -> be conservative, treat as NOT wrong.
    return False

# =====================================================
# STEP 3 - LOCATE THE CONFIRMED WRONG WORDS
# =====================================================

def detect_incorrect_words(text, confirmed_wrong):
    """Locate the confirmed-wrong words in the original paragraph."""

    detections = []

    word_spans = [
        (m.group(), m.start(), m.end())
        for m in re.finditer(r"\S+", text)
    ]

    for raw_word, start, end in word_spans:

        clean_word = re.sub(r"[^A-Za-z]", "", raw_word)

        if len(clean_word) < MIN_WORD_LENGTH:
            continue

        if clean_word.lower() not in confirmed_wrong:
            continue

        detections.append({
            "word": raw_word,
            "start": start,
            "end": end,
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

    # STEP 1: candidates from reading the whole paragraph
    candidates = flan_listed_wrong_words(text)

    # STEP 2: verify each candidate, keep only confirmed-wrong ones
    confirmed_wrong = {
        w for w in candidates if flan_says_word_is_wrong(w)
    }

    # STEP 3: locate + highlight
    detections = detect_incorrect_words(text, confirmed_wrong)

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
