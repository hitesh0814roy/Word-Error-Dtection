import pandas as pd
import json
import re
import torch

from spellchecker import SpellChecker

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

# =====================================================
# SPELL CHECKER
# =====================================================

spell = SpellChecker()

# =====================================================
# LOAD FLAN-T5 MODEL
# =====================================================

print("Loading Medical Correction Model...")

MODEL_NAME = "google/flan-t5-base"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME
)

print("Model loaded successfully!")

# =====================================================
# BASIC SPELL CORRECTION
# =====================================================

def basic_spell_fix(text):

    words = text.split()

    corrected_words = []

    for word in words:

        clean_word = re.sub(r'[^\w]', '', word)

        corrected = spell.correction(clean_word)

        if corrected:
            corrected_words.append(corrected)
        else:
            corrected_words.append(word)

    return " ".join(corrected_words)

# =====================================================
# MEDICAL CONTEXT CORRECTION
# =====================================================

def medical_correct(text):

    prompt = f"""
    Correct the following medical conversation.
    
    Fix:
    - spelling mistakes
    - medically incorrect words
    - invalid medical terminology
    - sentence grammar
    
    Return only corrected medical text.

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
        temperature=0.2
    )

    corrected_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True
    )

    return corrected_text

# =====================================================
# FULL PIPELINE
# =====================================================

def process_text(text):

    if pd.isna(text):
        return {}

    text = str(text)

    # -----------------------------------------
    # STEP 1 - SPELL FIX
    # -----------------------------------------

    spell_fixed = basic_spell_fix(text)

    # -----------------------------------------
    # STEP 2 - MEDICAL CORRECTION
    # -----------------------------------------

    medically_corrected = medical_correct(
        spell_fixed
    )

    return {
        # "original_text": text,
        # "spell_corrected": spell_fixed,
        "": medically_corrected
    }

# =====================================================
# READ EXCEL FILE
# =====================================================

INPUT_FILE = "medical_checker.xlsx"

df = pd.read_excel(INPUT_FILE)

if "error" not in df.columns:
    raise Exception(
        "Excel file must contain 'error' column"
    )

outputs = []

# =====================================================
# PROCESS ROWS
# =====================================================

for index, row in df.iterrows():

    text = row["error"]

    result = process_text(text)

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

OUTPUT_FILE = "medical_output.xlsx"

df.to_excel(
    OUTPUT_FILE,
    index=False
)

print("\nProcessing complete!")
print(f"Saved: {OUTPUT_FILE}")