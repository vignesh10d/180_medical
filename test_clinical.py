from transformers import pipeline, AutoTokenizer, AutoModelForMaskedLM

# Load the BiomedBERT model for masked language modeling
model_name = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Create the fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Your example from the URL
text = "The patient was diagnosed with [MASK] cancer."

# Get predictions
results = fill_mask(text)

print("Top predictions for cancer type:")
print("-" * 40)
for i, result in enumerate(results[:5], 1):
    token = result['token_str']
    score = result['score']
    sentence = result['sequence']
    print(f"{i}. {token} (confidence: {score:.4f})")
    print(f"   Full sentence: {sentence}")
    print()

# Try other medical examples
medical_examples = [
    "The patient presented with [MASK] chest pain.",
    "Blood pressure medication [MASK] was prescribed.",
    "The [MASK] showed abnormal results.",
    "Patient has a history of [MASK] diabetes.",
    "The surgery was [MASK] successful."
]

print("\n" + "="*50)
print("OTHER MEDICAL EXAMPLES:")
print("="*50)

for example in medical_examples:
    print(f"\nExample: {example}")
    print("-" * 30)
    results = fill_mask(example)
    for i, result in enumerate(results[:3], 1):
        print(f"{i}. {result['token_str']} ({result['score']:.4f})")