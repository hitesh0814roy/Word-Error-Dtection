import pandas as pd
import json

# =====================================================
# COMBINE  combined_confidence  ACROSS MODELS
# =====================================================
# Reads each model's output Excel file, pulls the
# "combined_confidence" value from its detection column, and writes
# everything into ONE Excel workbook:
#   - one sheet per model
#   - one final "Summary" sheet comparing all models side by side
# =====================================================

# -----------------------------------------------------
# CONFIG  -  specify each model's output Excel path here
# -----------------------------------------------------
# Format:  "Sheet / model name" : "path to that model's Excel file"

MODEL_FILES = {
    "BioClinicalBERT": r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Detection_BioClinicalBERT.xlsx",
    "PubMedBERT":      r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Detection_PubMedBERT.xlsx",
    "ClinicalBERT":    r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Detection_ClinicalBERT.xlsx",
    "SapBERT":         r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Detection_SapBERT.xlsx",
    "FLAN-T5":         r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Detection_FLAN-T5.xlsx",
}

# Where to write the combined workbook.
OUTPUT_FILE = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\Output_Detection\Combined_Result.xlsx"

# =====================================================
# HELPERS
# =====================================================

def extract_combined_confidence(cell):
    """Given one cell that holds a detection JSON string, return its
    'combined_confidence' value. Returns None if it cannot be read."""

    if pd.isna(cell):
        return None

    try:
        data = json.loads(str(cell))
    except (ValueError, TypeError):
        return None

    if isinstance(data, dict):
        return data.get("combined_confidence")

    return None


def find_confidence_series(df):
    """Find the column in a model's sheet that contains the detection
    JSON, and return a Series of its combined_confidence values."""

    for col in df.columns:
        extracted = df[col].apply(extract_combined_confidence)

        # If at least one value parsed successfully, this is the
        # detection column.
        if extracted.notna().any():
            return extracted

    # Nothing found -> all None.
    return pd.Series([None] * len(df))

# =====================================================
# READ EACH MODEL FILE
# =====================================================

per_model_confidence = {}

for model_name, path in MODEL_FILES.items():

    print(f"Reading {model_name}: {path}")

    try:
        df = pd.read_excel(path)
    except FileNotFoundError:
        print(f"  !! File not found, skipping {model_name}")
        continue

    confidence = find_confidence_series(df)

    per_model_confidence[model_name] = confidence.reset_index(drop=True)

    print(
        f"  -> {confidence.notna().sum()} row(s) with "
        f"combined_confidence"
    )

if not per_model_confidence:
    raise Exception(
        "No model files could be read. Check the paths in MODEL_FILES."
    )

# =====================================================
# BUILD THE SUMMARY TABLE
# =====================================================

# Each model becomes a column; rows line up by row number.
summary = pd.DataFrame(per_model_confidence)

summary.insert(0, "Row", range(1, len(summary) + 1))

# Average confidence across all models for each row.
model_cols = list(per_model_confidence.keys())
summary["Average_Confidence"] = summary[model_cols].mean(
    axis=1,
    skipna=True
).round(6)

# =====================================================
# WRITE COMBINED WORKBOOK  (one sheet per model + Summary)
# =====================================================

print(f"\nWriting combined workbook: {OUTPUT_FILE}")

with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:

    # One sheet per model.
    for model_name, confidence in per_model_confidence.items():

        sheet_df = pd.DataFrame({
            "Row": range(1, len(confidence) + 1),
            "combined_confidence": confidence
        })

        # Excel sheet names max 31 chars and cannot contain : \ / ? * [ ]
        safe_name = model_name[:31].replace("/", "-")

        sheet_df.to_excel(writer, sheet_name=safe_name, index=False)

    # Final comparison sheet.
    summary.to_excel(writer, sheet_name="Summary", index=False)

print("Done! Combined confidence written to:")
print(OUTPUT_FILE)
