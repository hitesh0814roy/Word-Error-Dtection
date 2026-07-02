import pandas as pd
import json
import torch

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =====================================================
# EXCEL FILE PATHS
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\output\Medical_PubMedBERT1.xlsx"

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading medical correction model...")

# Using FLAN-T5 with PubMed-style prompting
MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully!")

# =====================================================
# MEDICAL CORRECTION FUNCTION
# =====================================================

def correct_medical_conversation(text):

    prompt = f"""
    You are a biomedical and clinical text correction assistant.

    Correct the following patient conversation.

    Fix:
    - spelling mistakes
    - incorrect medical terminology
    - grammar
    - contextual medical mistakes
    - malformed clinical phrases

    Return ONLY corrected medical conversation.
    change the words
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

    Conversation:
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
            max_new_tokens=256,
            temperature=0.2,
            do_sample=False
        )

    corrected_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return corrected_text

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

    text = str(row["error"])

    corrected = correct_medical_conversation(
        text
    )

    result = {
        # "original_text": text,
        "corrected_text": corrected
    }

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

print("\nProcessing completed successfully!")

print(f"Saved output file: {OUTPUT_FILE}")