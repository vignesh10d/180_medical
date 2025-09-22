from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    pipeline
)
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict
import warnings
warnings.filterwarnings('ignore')

class UrologyBioGPTEvaluator:
    def __init__(self, model_name: str = "microsoft/biogpt"):
        """
        Initialize evaluator with BioGPT model
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.urology_test_cases = [
            "Patient presents with acute kidney injury and elevated creatinine levels.",
            "Chronic kidney disease stage 4 with proteinuria and hypertension.",
            "Bladder cancer with muscle invasion requiring radical cystectomy.",
            "Benign prostatic hyperplasia causing urinary retention and frequency.",
            "Nephrolithiasis with calcium oxalate stones in the right ureter.",
            "Urinary tract infection with E. coli bacteria and dysuria symptoms.",
            "Prostate-specific antigen elevated at 8.5 ng/mL suggesting malignancy.",
            "Glomerulonephritis with hematuria and decreased glomerular filtration rate.",
            "Urethral stricture causing voiding dysfunction and weak stream.",
            "Renal cell carcinoma in the left kidney with partial nephrectomy planned."
        ]
        self.urology_ontology_terms = {
            'anatomical_structures': [
                'kidney', 'bladder', 'prostate', 'urethra', 'ureter', 
                'glomerulus', 'nephron', 'renal cortex', 'renal medulla'
            ],
            'conditions': [
                'kidney disease', 'bladder cancer', 'prostate cancer', 
                'urinary tract infection', 'kidney stones', 'proteinuria',
                'hematuria', 'urinary retention', 'nephritis'
            ],
            'procedures': [
                'nephrectomy', 'cystectomy', 'prostatectomy', 'lithotripsy',
                'ureteroscopy', 'cystoscopy', 'dialysis', 'kidney transplant'
            ],
            'biomarkers': [
                'creatinine', 'urea', 'PSA', 'protein', 'albumin',
                'glomerular filtration rate', 'blood urea nitrogen'
            ]
        }

    def load_model(self):
        print(f"Loading BioGPT model: {self.model_name}")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            print("✓ BioGPT loaded successfully")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {e}")
            return False

    def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings by averaging last hidden states
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)
                # average last hidden state across tokens
                last_hidden = outputs.hidden_states[-1].squeeze(0).numpy()
                embedding = np.mean(last_hidden, axis=0)
                embeddings.append(embedding)
        return np.array(embeddings)

    def evaluate_semantic_similarity(self) -> Dict:
        print("\n🔬 Evaluating semantic similarity for urology concepts...")
        results = {}
        test_embeddings = self.get_embeddings(self.urology_test_cases)
        for category, terms in self.urology_ontology_terms.items():
            print(f"\nAnalyzing {category}...")
            term_embeddings = self.get_embeddings(terms)
            similarities = cosine_similarity(test_embeddings, term_embeddings)
            best_matches = []
            for i, test_case in enumerate(self.urology_test_cases):
                best_term_idx = np.argmax(similarities[i])
                best_score = similarities[i][best_term_idx]
                best_term = terms[best_term_idx]
                best_matches.append({
                    'text': test_case[:60] + "...",
                    'best_term': best_term,
                    'similarity': best_score
                })
            results[category] = {
                'avg_similarity': np.mean(similarities),
                'max_similarity': np.max(similarities),
                'best_matches': best_matches
            }
        return results

    def generate_evaluation_report(self) -> str:
        if not self.load_model():
            return "Failed to load model."
        similarity_results = self.evaluate_semantic_similarity()
        report = f"\n=== BioGPT Urology Ontology Evaluation Report ===\nModel: {self.model_name}\n"
        for category, results in similarity_results.items():
            report += f"\n### {category.upper()}\n"
            report += f"Average Similarity: {results['avg_similarity']:.3f}\n"
            report += f"Max Similarity: {results['max_similarity']:.3f}\n"
            for match in results['best_matches'][:3]:
                report += f"  - {match['text']} → {match['best_term']} ({match['similarity']:.3f})\n"
        return report


def main():
    evaluator = UrologyBioGPTEvaluator()
    report = evaluator.generate_evaluation_report()
    print(report)
    with open("biogpt_urology_evaluation.txt", "w") as f:
        f.write(report)


if __name__ == "__main__":
    main()
