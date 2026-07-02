from transformers import AutoTokenizer, AutoModel

model_name = "emilyalsentzer/Bio_ClinicalBERT"

print("Downloading tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)

print("Downloading model...")

model = AutoModel.from_pretrained(model_name)

print("BioClinicalBERT loaded successfully!")