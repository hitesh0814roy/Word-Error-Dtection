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

OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\output\medical_checker_clinicalBERT.xlsx"

# =====================================================
# LOAD CLINICAL CORRECTION MODEL
# =====================================================

print("Loading Clinical Medical Correction Model...")

# ClinicalBERT-inspired correction pipeline
# Using FLAN-T5 for generation + clinical prompting

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully!")

# =====================================================
# CLINICAL MEDICAL CORRECTION FUNCTION
# =====================================================

def correct_clinical_paragraph(text):

    prompt = f"""
    You are a medical data correction agent.

    Correct the following clinical paragraph.

    Fix:
    - spelling mistakes
    - clinical terminology
    - biomedical terminology
    - grammar mistakes
    - malformed patient conversations
    - incorrect abbreviations
    - contextual medical errors

    Preserve clinical paragraph without changing many details.

    Return ONLY corrected paragraph.

    Clinical Paragraph:
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
# CHECK REQUIRED COLUMN
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

    corrected = correct_clinical_paragraph(
        text
    )

    result = {
        # "original_text": text,
        "": corrected
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

print("\nClinical correction completed!")

print(f"Saved output file: {OUTPUT_FILE}")