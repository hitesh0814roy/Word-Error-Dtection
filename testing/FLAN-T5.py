import pandas as pd
import json
import os

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading FLAN-T5 model...")

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

def correct_medical_text(text):

    prompt = f"""
    You are a medical text correction assistant.

    Correct:
    - spelling mistakes
    - medical terminology
    - grammar
    - contextual medical errors

    Correct the medically error terms.

    Return only corrected medical words and sentences.

    Text:
    {text}
    """

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    outputs = model.generate(
        **inputs,
        max_new_tokens=128,
        temperature=0.2,
        do_sample=False
    )

    corrected_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return corrected_text

# =====================================================
# READ EXCEL
# =====================================================

INPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\medical_checker.xlsx"

print(f"Reading file: {INPUT_FILE}")

df = pd.read_excel(INPUT_FILE)

# =====================================================
# CHECK COLUMN
# =====================================================

if "error" not in df.columns:
    raise Exception(
        "Excel must contain 'error' column"
    )

# =====================================================
# PROCESS ROWS
# =====================================================

outputs = []

for index, row in df.iterrows():

    text = str(row["error"])

    corrected = correct_medical_text(text)

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

# Create output folder
os.makedirs("output", exist_ok=True)

OUTPUT_FILE = "output/medical_output.xlsx"

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nProcessing completed!")
print(f"Saved file: {OUTPUT_FILE}")