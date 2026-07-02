import pandas as pd
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    AutoModelForSeq2SeqLM
)

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\output\Medical_BioClinic.xlsx"

INPUT_COLUMN = "error"

# =====================================================
# DETECTION CONFIG
# =====================================================

# A token whose model probability falls below this is treated
# as a suspicious / likely-incorrect medical word.
SUSPICION_THRESHOLD = 0.01

# Ignore very short tokens (the, of, mg, etc.)
MIN_WORD_LENGTH = 4

# =====================================================
# KNOWN MEDICAL TYPO DICTIONARY
# =====================================================
# These are GUARANTEED, deterministic corrections. They are
# applied in code (not left to the model), so they always work.

MEDICAL_CORRECTIONS = {
    "tengu": "dengue",
    "degu": "dengue",
    "platlets": "platelets",
    "hedache": "headache",
    "feever": "fever",
    "tigtness": "tightness",
    "ecg": "ECG",
    "cbc": "CBC",
    "thyriod": "thyroid",
    "anemea": "anemia",
    "gestonal": "gestational",
    "hypertention": "hypertension",
    "vomitting": "vomiting",
    "abdmonal": "abdominal",
    "tachycardya": "tachycardia",
}


def apply_known_corrections(text):
    """
    Replace every known typo with its correct term using
    whole-word, case-insensitive matching. This runs as real
    code, so these fixes are always enforced.
    """

    def _replace(match):
        return MEDICAL_CORRECTIONS[match.group(0).lower()]

    pattern = r"\b(" + "|".join(
        re.escape(k) for k in MEDICAL_CORRECTIONS
    ) + r")\b"

    return re.sub(pattern, _replace, text, flags=re.IGNORECASE)

# =====================================================
# LOAD DETECTION MODEL  (Bio_ClinicalBERT)
# =====================================================

print("Loading Bio_ClinicalBERT (detection model)...")

DETECTION_MODEL_NAME = "emilyalsentzer/Bio_ClinicalBERT"

detect_tokenizer = AutoTokenizer.from_pretrained(
    DETECTION_MODEL_NAME
)

# MaskedLM head is required so we can score how likely each
# word is in its clinical context.
detect_model = AutoModelForMaskedLM.from_pretrained(
    DETECTION_MODEL_NAME
)

detect_model.eval()

print("Detection model loaded.")

# =====================================================
# LOAD CORRECTION MODEL  (FLAN-T5)
# =====================================================

print("Loading FLAN-T5 (correction model)...")

CORRECTION_MODEL_NAME = "google/flan-t5-base"

t5_tokenizer = AutoTokenizer.from_pretrained(
    CORRECTION_MODEL_NAME
)

t5_model = AutoModelForSeq2SeqLM.from_pretrained(
    CORRECTION_MODEL_NAME
)

t5_model.eval()

print("Correction model loaded.")

# =====================================================
# STEP 1 - DETECTION  (Bio_ClinicalBERT)
# =====================================================

def detect_suspicious_words(text):
    """
    Use Bio_ClinicalBERT as a masked language model.

    For each word, mask it and ask the clinical model how
    likely the original word is in that context. Words the
    clinical model finds very improbable are flagged as
    suspicious (likely misspelled / wrong medical term).
    Change the words:
    """

    words = text.split()

    suspicious = []

    for i, word in enumerate(words):

        clean_word = re.sub(r"[^A-Za-z]", "", word)

        if len(clean_word) < MIN_WORD_LENGTH:
            continue

        # Build a version of the sentence with this word masked.
        masked_words = words.copy()
        masked_words[i] = detect_tokenizer.mask_token
        masked_sentence = " ".join(masked_words)

        inputs = detect_tokenizer(
            masked_sentence,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # Locate the [MASK] position.
        mask_positions = (
            inputs["input_ids"][0]
            == detect_tokenizer.mask_token_id
        ).nonzero(as_tuple=True)[0]

        if len(mask_positions) == 0:
            # Word got truncated away; skip it.
            continue

        mask_index = mask_positions[0].item()

        with torch.no_grad():
            logits = detect_model(**inputs).logits

        probs = torch.softmax(logits[0, mask_index], dim=-1)

        # How likely did the clinical model think the ORIGINAL word was?
        original_ids = detect_tokenizer(
            clean_word,
            add_special_tokens=False
        )["input_ids"]

        if len(original_ids) == 0:
            continue

        # Use the first subword token to score the word.
        original_prob = probs[original_ids[0]].item()

        if original_prob < SUSPICION_THRESHOLD:
            suspicious.append({
                "word": clean_word,
                "position": i,
                "confidence": round(original_prob, 8)
            })

    return suspicious

# =====================================================
# STEP 2 - CORRECTION  (FLAN-T5)
# =====================================================

def correct_with_flan_t5(text, suspicious_words):
    """
    Feed the original text AND the suspicious words detected by
    Bio_ClinicalBERT into FLAN-T5, so correction is focused on
    the words the clinical model flagged.
    Change the words:
    "tengu": "dengue",
    "platlets": "platelets",
    "hedache": "headache",
    "feever": "fever",
    "tigtness": "tightness",
    "ecg": "ECG",
    "cbc": "CBC",
    "thyriod": "thyroid",
    "anemea": "anemia",
    "gestonal": "gestational",
    "hypertention": "hypertension",
    "vomitting": "vomiting",
    "abdmonal": "abdominal",
    "tachycardya": "tachycardia"
    if comes in the text for correction.
    """

    if suspicious_words:
        flagged = ", ".join(s["word"] for s in suspicious_words)
        focus_line = (
            f"Pay special attention to these likely-incorrect "
            f"medical words: {flagged}."
        )
    else:
        focus_line = "No specific words were flagged."

    prompt = f"""You are a medical documentation assistant.

Correct the following clinical text.
Fix spelling mistakes, incorrect medical terminology and grammar.
Preserve the medical meaning. Return ONLY the corrected paragraph.

{focus_line}

Text:
{text}
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
            max_new_tokens=256,
            num_beams=5,
            early_stopping=True
        )

    return t5_tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

# =====================================================
# FULL PIPELINE  (detection -> correction)
# =====================================================

def process_text(text):

    if pd.isna(text):
        return {}

    text = str(text)

    # STEP 0: enforce known typo fixes BEFORE the model sees the text
    text = apply_known_corrections(text)

    # STEP 1: clinical model detects suspicious words
    suspicious_words = detect_suspicious_words(text)

    # STEP 2: FLAN-T5 corrects, guided by the detection output
    corrected_text = correct_with_flan_t5(
        text,
        suspicious_words
    )

    # STEP 3: enforce known typo fixes AGAIN, in case FLAN-T5
    # reintroduced or missed any of them
    corrected_text = apply_known_corrections(corrected_text)

    return {
        " " : corrected_text
        # "detected_words": [s["word"] for s in suspicious_words]
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

    text = row[INPUT_COLUMN]

    result = process_text(text)

    outputs.append(
        json.dumps(result, ensure_ascii=False)
    )

    print(f"Processed row {index + 1}")

# =====================================================
# SAVE OUTPUT
# =====================================================

df["output"] = outputs

df.to_excel(OUTPUT_FILE, index=False)

print("\nDetection (Bio_ClinicalBERT) + Correction (FLAN-T5) completed!")
print(f"Saved output file: {OUTPUT_FILE}")
