# 🏥 Medical Checker — Clinical Text Error Detection & Correction

A research pipeline for **detecting and correcting errors in clinical/medical text** using an ensemble of domain-specific BERT models and FLAN-T5. The system identifies misspelled, medically incorrect, or grammatically wrong words in clinical paragraphs and can optionally produce corrected output.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Models Used](#models-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Input / Output Format](#input--output-format)
- [Evaluation Metrics](#evaluation-metrics)
- [Configuration](#configuration)

---

## Overview

Clinical notes often contain typos, abbreviation errors, and medical terminology mistakes. This project builds a two-phase pipeline:

1. **Detection** — multiple domain-specific BERT masked language models score each word in context. A cheap spell-check gate ensures only genuine non-words are passed to the (expensive) neural models.
2. **Correction / Verification** — FLAN-T5 either verifies a flagged word is truly wrong or rewrites the clinical paragraph with errors fixed.

A voting mechanism merges the opinions of all four BERT detectors; a word must be flagged by at least **2 models** (configurable) to be considered incorrect.

---

## Pipeline Architecture

```
Input Excel (medical_checker.xlsx)
         │
         ▼
  ┌──────────────────────────────────────────────────┐
  │  STEP 1 — Spell-Check Gate (pyspellchecker)     │
  │  Real words (e.g., "antibody", "saline") → SKIP │
  │  Unknown words → pass to BERT models             │
  └──────────────────────┬───────────────────────────┘
                         │
         ┌───────────────┼───────────────┐
         ▼               ▼               ▼               ▼
  BioClinicalBERT   PubMedBERT    ClinicalBERT      SapBERT
  (masked LM —      (masked LM)   (masked LM)       (masked LM)
   scores P(word|context))
         │               │               │               │
         └───────────────┴───────────────┴───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  STEP 2 — Vote + Merge        │
         │  Word flagged by ≥ MIN_VOTES  │
         │  → confirmed detection        │
         │  Combined confidence averaged │
         └───────────────┬───────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │  STEP 3 — FLAN-T5             │
         │  Verify: "is this word wrong?"│
         │  OR Correct: rewrite paragraph│
         └───────────────┬───────────────┘
                         │
                         ▼
         Output Excel with detections,
         highlighted text, and confidence
```

---

## Models Used

| Role | Model | HuggingFace ID |
|------|-------|----------------|
| Detector | **BioClinicalBERT** | `emilyalsentzer/Bio_ClinicalBERT` |
| Detector | **PubMedBERT** | `microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext` |
| Detector | **ClinicalBERT** | `medicalai/ClinicalBERT` |
| Detector | **SapBERT** | `cambridgeltl/SapBERT-from-PubMedBERT-fulltext` |
| Corrector / Verifier | **FLAN-T5** | `google/flan-t5-base` |

All BERT models are used as **masked language models** — they predict the probability of each word given its surrounding context. A low probability signals the word may be wrong.

---

## Project Structure

```
Medical_Checker/
├── medical_checker.xlsx          # Main input dataset (columns: error, output)
├── Models_Name.xlsx              # Model comparison tracker
├── Models_Sim.xlsx               # Similarity results across models
├── frequency_dictionary_en_82_765.txt  # English frequency dictionary (spell-check)
├── requirement.txt               # Python dependencies
│
├── Detection Folder/             # Detection-only scripts (no text rewriting)
│   ├── BioClinicalBERT_fast.py   # BioClinicalBERT with spell-check gate (fast)
│   ├── BioClinicalBERT_test.py   # BioClinicalBERT baseline
│   ├── PubMedBERT_detection.py   # PubMedBERT detector
│   ├── ClinicalBERT_detection.py # ClinicalBERT detector
│   ├── SapBERT_detection.py      # SapBERT detector
│   ├── FLAN-T5_detection.py      # FLAN-T5 as verifier (yes/no per word)
│   ├── FLAN-T5_test.py           # FLAN-T5 baseline test
│   ├── Vote_and_Correct.py       # ★ Full pipeline: 4-BERT vote → FLAN-T5 verify
│   ├── Vote_and_Correct.ipynb    # Notebook version of the above
│   ├── Combined_result.py        # Aggregates all model outputs into one Excel
│   └── AllBERT_fast.py           # All 4 BERTs in one fast detection run
│
├── Files/                        # Correction scripts (rewrite the full paragraph)
│   ├── BioClinicalBERT.py        # Deprecated: BioClinicalBERT via Seq2Seq
│   ├── ClinicalBERT.py           # FLAN-T5 prompted as ClinicalBERT corrector
│   ├── PubMedBERT.py             # PubMedBERT correction
│   ├── SapBERT.py                # SapBERT correction
│   ├── FLAN-T51.py               # FLAN-T5 paragraph correction
│   └── detection_text.py         # Utility for detection text processing
│
├── output/                       # Correction model outputs (Excel)
├── Output_Detection/             # Detection model outputs (Excel)
├── testing/                      # Test scripts and notebooks
├── Google colab/                 # Google Colab versions of all scripts
│
├── Similarity_Checker.py         # Jaro-Winkler similarity scoring (Real vs output)
└── model_testing.py              # Evaluation: BLEU, WER, ROUGE, Exact Match
```

---

## Installation

### Prerequisites

- Python 3.9+
- CUDA-capable GPU recommended (models run on CPU but are slow)

### Install Dependencies

```bash
pip install -r requirement.txt
```

Additional packages used in evaluation scripts:

```bash
pip install jiwer nltk rouge-score rapidfuzz
```

---

## Usage

### 1. Run the Full Vote-and-Correct Pipeline (Recommended)

Detects errors using 4 BERT models + verifies with FLAN-T5:

```bash
python "Detection Folder/Vote_and_Correct.py"
```

Output: `output/Vote_and_Correct_New.xlsx`

---

### 2. Run a Single Fast Detector

Uses BioClinicalBERT with spell-check gating for speed:

```bash
python "Detection Folder/BioClinicalBERT_fast.py"
```

Output: `Output/Detection_BioClinicalBERT_Fast.xlsx`

---

### 3. Run All BERT Detectors and Combine Results

Run each detection script individually, then:

```bash
python "Detection Folder/Combined_result.py"
```

Output: `Output_Detection/Combined_Result.xlsx` — one sheet per model + a Summary sheet.

---

### 4. Run a Correction Script (FLAN-T5 rewrites paragraph)

```bash
python Files/ClinicalBERT.py
```

Output: `output/medical_checker_clinicalBERT.xlsx`

---

### 5. Evaluate Model Output

Compute BLEU, WER, ROUGE, and exact match accuracy:

```bash
python model_testing.py
```

Compute Jaro-Winkler text similarity:

```bash
python Similarity_Checker.py
```

---

## Input / Output Format

### Input Excel (`medical_checker.xlsx`)

| Column | Description |
|--------|-------------|
| `error` | Clinical text with intentional or real errors |
| `output` | (Optional) Ground-truth corrected text for evaluation |

### Detection Output

Each detection script appends a column (e.g., `BioClinicalBERT_detection`) containing a JSON string:

```json
{
  "highlighted_text": "Patient has <<haemorrphage>> in the left ventricle.",
  "detected_words": [
    {
      "word": "haemorrphage",
      "start": 12,
      "end": 24,
      "detection_confidence": 0.999812
    }
  ],
  "combined_confidence": 0.999812
}
```

Wrong words are wrapped in `<<` `>>` markers in `highlighted_text`.

### Vote-and-Correct Output

Adds a `vote_and_verify` column with JSON including:

- `highlighted_text` — text with `<<flagged>>` words
- `flagged_words` — list with per-model votes and FLAN-T5 verdict (`yes`/`no`/`unsure`)
- `confirmed_wrong_words` — only words FLAN-T5 confirmed as wrong
- `combined_confidence` — average detection confidence

---

## Evaluation Metrics

`model_testing.py` computes the following against ground-truth corrections:

| Metric | Description |
|--------|-------------|
| **Exact Match Accuracy** | % of outputs identical to expected |
| **BLEU** | N-gram overlap (0–1, higher is better) |
| **WER** | Word Error Rate (lower is better) |
| **ROUGE-1** | Unigram F1 overlap |
| **ROUGE-L** | Longest common subsequence F1 |

`Similarity_Checker.py` computes **Jaro-Winkler similarity** (%) between `Real_Data` and `output` columns per Excel sheet.

---

## Configuration

Key constants at the top of each detection script:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `SUSPICION_THRESHOLD` | `0.001` | MLM probability below which a word is flagged |
| `MIN_VOTES` | `2` | Minimum BERT models that must agree to flag a word |
| `MIN_WORD_LENGTH` | `1` | Ignore tokens shorter than this many characters |
| `HL_OPEN` / `HL_CLOSE` | `<<` / `>>` | Highlight markers for flagged words |

---

## How Detection Works

1. **Spell-check gate** — `pyspellchecker` quickly labels each word as a real English/medical word or not. Real words like *antibody*, *dengue*, *saline* are **skipped entirely** before any neural model runs. This reduces model calls from ~50 per sentence to ~3, giving a large speed-up with identical output.

2. **Masked Language Model scoring** — for each candidate non-word, the word is replaced with `[MASK]` tokens (one per subword piece) and the model predicts the probability of the original token. A very low probability means the model finds the word unlikely in context → flagged as incorrect.

3. **Multi-model voting** — four independent BERT models each produce a confidence score. Only words receiving votes from at least `MIN_VOTES` models are surfaced.

4. **FLAN-T5 verification / correction** — either a yes/no judgment per word ("is this word wrong?") or a full paragraph rewrite with errors fixed.

---

## License

This project is for research and educational purposes.
