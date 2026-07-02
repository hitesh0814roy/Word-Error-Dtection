import pandas as pd

from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer

# ==========================================
# READ EXCEL
# ==========================================

FILE = "medical_checker.xlsx"

df = pd.read_excel(FILE)

# ==========================================
# CHECK REQUIRED COLUMNS
# ==========================================

required_columns = [
    "error",
    "output"
]

for col in required_columns:

    if col not in df.columns:
        raise Exception(
            f"Missing column: {col}"
        )

# ==========================================
# INITIALIZE METRICS
# ==========================================

total = 0

exact_matches = 0

bleu_scores = []

wer_scores = []

rouge1_scores = []

rougeL_scores = []

scorer = rouge_scorer.RougeScorer(
    ['rouge1', 'rougeL'],
    use_stemmer=True
)

# ==========================================
# EVALUATE
# ==========================================

for _, row in df.iterrows():

    expected = str(row["error"]).strip().lower()

    predicted = str(row["output"]).strip().lower()

    total += 1

    # --------------------------------------
    # EXACT MATCH
    # --------------------------------------

    if expected == predicted:
        exact_matches += 1

    # --------------------------------------
    # BLEU
    # --------------------------------------

    bleu = sentence_bleu(
        [expected.split()],
        predicted.split()
    )

    bleu_scores.append(bleu)

    # --------------------------------------
    # WER
    # --------------------------------------

    error_rate = wer(
        expected,
        predicted
    )

    wer_scores.append(error_rate)

    # --------------------------------------
    # ROUGE
    # --------------------------------------

    rouge_scores = scorer.score(
        expected,
        predicted
    )

    rouge1_scores.append(
        rouge_scores["rouge1"].fmeasure
    )

    rougeL_scores.append(
        rouge_scores["rougeL"].fmeasure
    )

# ==========================================
# FINAL RESULTS
# ==========================================

exact_match_accuracy = (
    exact_matches / total
) * 100

average_bleu = (
    sum(bleu_scores) / total
)

average_wer = (
    sum(wer_scores) / total
)

average_rouge1 = (
    sum(rouge1_scores) / total
)

average_rougeL = (
    sum(rougeL_scores) / total
)

# ==========================================
# PRINT RESULTS
# ==========================================

print("\n========== MODEL EVALUATION ==========\n")

print(
    f"Exact Match Accuracy: "
    f"{exact_match_accuracy:.2f}%"
)

print(
    f"Average BLEU Score: "
    f"{average_bleu:.4f}"
)

print(
    f"Average WER: "
    f"{average_wer:.4f}"
)

print(
    f"Average ROUGE-1: "
    f"{average_rouge1:.4f}"
)

print(
    f"Average ROUGE-L: "
    f"{average_rougeL:.4f}"
)

print("\n======================================")