import pandas as pd
import json
import re
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM
)

from spellchecker import SpellChecker

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\output\medical_checker_SapBERT.xlsx"

# =====================================================
# LOAD SPELL CHECKER
# =====================================================

spell = SpellChecker()

# =====================================================
# LOAD SAPBERT
# =====================================================

print("Loading SapBERT model...")

MODEL_NAME = (
    "cambridgeltl/"
    "SapBERT-from-PubMedBERT-fulltext"
)

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForMaskedLM.from_pretrained(
    MODEL_NAME
)

model.eval()

print("SapBERT loaded successfully!")

# =====================================================
# MEDICAL TERM DICTIONARY
# =====================================================

medical_terms = {
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
}

# =====================================================
# SPELL + MEDICAL CORRECTION
# =====================================================

def correct_medical_word(word):

    # ------------------------------------------
    # CUSTOM MEDICAL DICTIONARY
    # ------------------------------------------

    if word.lower() in medical_terms:

        return medical_terms[word.lower()]

    # ------------------------------------------
    # SPELL CHECKER
    # ------------------------------------------

    corrected = spell.correction(word)

    if corrected:

        return corrected

    return word

# =====================================================
# SAPBERT CONTEXT ANALYSIS
# =====================================================

def analyze_context(sentence):

    inputs = tokenizer(
        sentence,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    with torch.no_grad():

        outputs = model(**inputs)

    logits = outputs.logits

    probabilities = torch.softmax(
        logits,
        dim=-1
    )

    token_ids = inputs["input_ids"][0]

    suspicious_tokens = []

    for i, token_id in enumerate(token_ids):

        token = tokenizer.convert_ids_to_tokens(
            int(token_id)
        )

        probability = probabilities[
            0,
            i,
            token_id
        ].item()

        clean_token = token.replace(
            "##",
            ""
        )

        if (
            probability < 0.0001
            and len(clean_token) > 3
            and clean_token.isalpha()
        ):

            suspicious_tokens.append({
                "token": clean_token,
                "confidence": round(
                    probability,
                    8
                )
            })

    return suspicious_tokens

# =====================================================
# MAIN PROCESS FUNCTION
# =====================================================

def process_medical_paragraph(text):

    if pd.isna(text):

        return {}

    text = str(text)

    detected_errors = []

    corrected_words = []

    # ------------------------------------------
    # TOKENIZE WORDS
    # ------------------------------------------

    matches = list(
        re.finditer(r"\b\w+\b", text)
    )

    for match in matches:

        word = match.group()

        start = match.start()

        end = match.end()

        corrected = correct_medical_word(
            word
        )

        corrected_words.append(corrected)

        if corrected.lower() != word.lower():

            detected_errors.append({
                "word": word,
                "start": start,
                "end": end,
                "suggestion": corrected,
                "type": "medical_spelling_error"
            })

    # ------------------------------------------
    # REBUILD SENTENCE
    # ------------------------------------------

    corrected_text = " ".join(
        corrected_words
    )

    # ------------------------------------------
    # CONTEXT ANALYSIS
    # ------------------------------------------

    try:

        suspicious = analyze_context(
            corrected_text
        )

    except Exception as e:

        suspicious = []

        print(
            "Context analysis error:",
            e
        )

    return {
        # "original_text": text,
        "corrected_text": corrected_text,
        # "detected_errors": detected_errors,
        # "suspicious_medical_terms": suspicious
    }

# =====================================================
# READ EXCEL FILE
# =====================================================

print(f"Reading Excel file: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

# =====================================================
# CHECK COLUMN
# =====================================================

if "error" not in df.columns:

    raise Exception(
        "Excel file must contain "
        "'error' column"
    )

# =====================================================
# PROCESS ROWS
# =====================================================

outputs = []

for index, row in df.iterrows():

    text = row["error"]

    result = process_medical_paragraph(
        text
    )

    outputs.append(
        json.dumps(
            result,
            ensure_ascii=False
        )
    )

    print(f"Processed row {index + 1}")

# =====================================================
# SAVE OUTPUT
# =====================================================

df["output"] = outputs

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nSapBERT correction completed!")

print(f"Saved output file: {OUTPUT_FILE}")

