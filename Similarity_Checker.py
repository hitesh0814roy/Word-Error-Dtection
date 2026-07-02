import pandas as pd
from rapidfuzz import distance

excel_path = r"C:\Users\HITESH S ROY\OneDrive\Documents\Medical_Checker\output\Medical_BioClinic.xlsx"

sheet_names = pd.ExcelFile(excel_path).sheet_names

column1 = "Real_Data"
column2 = "output"

output_column = "Similarity_Percentage"

def calculate_similarity(text1, text2):
    if pd.isna(text1) or pd.isna(text2):
        return None

    return round(
        distance.JaroWinkler.normalized_similarity(
            str(text1),
            str(text2)
        ) * 100,
        2
    )

all_sheets = {}

for sheet_name in sheet_names:

    print(f"Processing {sheet_name}")

    df = pd.read_excel(
        excel_path,
        sheet_name=sheet_name
    )

    if column1 not in df.columns:
        print(f"Column '{column1}' not found in {sheet_name}")
        continue

    if column2 not in df.columns:
        print(f"Column '{column2}' not found in {sheet_name}")
        continue

    df[output_column] = df.apply(
        lambda row: calculate_similarity(
            row[column1],
            row[column2]
        ),
        axis=1
    )

    print(
        f"{sheet_name} Average Similarity = "
        f"{df[output_column].mean():.2f}%"
    )

    all_sheets[sheet_name] = df

with pd.ExcelWriter(
    excel_path,
    engine="openpyxl"
) as writer:

    for sheet_name, df in all_sheets.items():
        df.to_excel(
            writer,
            sheet_name=sheet_name,
            index=False
        )

print("Done!")