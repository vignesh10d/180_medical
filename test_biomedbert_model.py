#!/usr/bin/env python3
"""
BiomedBERT Model Evaluation for Urology Ontology Tasks (FIXED VERSION)

This script evaluates the Microsoft BiomedNLP-BiomedBERT model
for urology-specific ontology and NER tasks.

Fixed model identifiers and added error handling.
"""

import torch
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoModelForTokenClassification,
    pipeline
)
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
from typing import List, Dict, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class UrologyBiomedBERTEvaluator:
    def __init__(self, model_name: str = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract"):
        """
        Initialize the evaluator with the BiomedBERT model
        
        Args:
            model_name: HuggingFace model identifier
                      Options:
                      - "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract" (default)
                      - "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
                      - "microsoft/BiomedNLP-BiomedBERT-large-uncased-abstract"
                      - "emilyalsentzer/Bio_ClinicalBERT" (alternative clinical BERT)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.ner_pipeline = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        print(f"Device: {self.device}")
        
        # Urology-specific test cases
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
        
        # Urology ontology terms for similarity testing
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
        
        # Alternative model options for fallback
        self.alternative_models = [
            "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
            "emilyalsentzer/Bio_ClinicalBERT",
            "NeuML/pubmedbert-base-embeddings"
        ]
    
    def load_model(self, retry_alternatives: bool = True) -> bool:
        """
        Load the BiomedBERT model and tokenizer with fallback options
        
        Args:
            retry_alternatives: If True, try alternative models if the primary fails
        """
        models_to_try = [self.model_name] + (self.alternative_models if retry_alternatives else [])
        
        for model_name in models_to_try:
            print(f"Loading model: {model_name}")
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModel.from_pretrained(model_name)
                self.model.to(self.device)
                self.model_name = model_name  # Update to the successfully loaded model
                print(f"✓ Model loaded successfully: {model_name}")
                return True
            except Exception as e:
                print(f"✗ Error loading {model_name}: {e}")
                continue
        
        print("❌ Failed to load any model. Please check your internet connection and model availability.")
        return False
    
    def get_embeddings(self, texts: List[str], batch_size: int = 8) -> np.ndarray:
        """
        Get BERT embeddings for a list of texts with batching for efficiency
        
        Args:
            texts: List of text strings
            batch_size: Number of texts to process at once
            
        Returns:
            numpy array of embeddings
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        embeddings = []
        
        # Process in batches
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and encode batch
            inputs = self.tokenizer(batch_texts, return_tensors='pt', 
                                  truncation=True, padding=True, 
                                  max_length=512)
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                # Use [CLS] token embedding (first token)
                batch_embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
                embeddings.extend(batch_embeddings)
        
        return np.array(embeddings)
    
    def evaluate_semantic_similarity(self) -> Dict:
        """
        Evaluate semantic similarity between urology terms and clinical text
        
        Returns:
            Dictionary with similarity scores and analysis
        """
        print("\n🔬 Evaluating semantic similarity for urology concepts...")
        
        results = {}
        
        try:
            # Get embeddings for test cases
            print("  Computing embeddings for test cases...")
            test_embeddings = self.get_embeddings(self.urology_test_cases)
            
            for category, terms in self.urology_ontology_terms.items():
                print(f"  Analyzing {category}...")
                
                # Get embeddings for ontology terms
                term_embeddings = self.get_embeddings(terms)
                
                # Calculate similarities
                similarities = cosine_similarity(test_embeddings, term_embeddings)
                
                # Find best matches
                best_matches = []
                for i, test_case in enumerate(self.urology_test_cases):
                    best_term_idx = np.argmax(similarities[i])
                    best_score = similarities[i][best_term_idx]
                    best_term = terms[best_term_idx]
                    
                    best_matches.append({
                        'text': test_case[:60] + "..." if len(test_case) > 60 else test_case,
                        'best_term': best_term,
                        'similarity': best_score
                    })
                
                results[category] = {
                    'avg_similarity': float(np.mean(similarities)),
                    'max_similarity': float(np.max(similarities)),
                    'min_similarity': float(np.min(similarities)),
                    'std_similarity': float(np.std(similarities)),
                    'best_matches': best_matches
                }
                
                print(f"    ✓ {category}: avg={results[category]['avg_similarity']:.3f}")
        
        except Exception as e:
            print(f"  ✗ Error in semantic similarity evaluation: {e}")
            return {}
        
        return results
    
    def test_named_entity_recognition(self) -> Optional[List]:
        """
        Test the model's ability to identify medical entities
        """
        print("\n🏥 Testing Named Entity Recognition capabilities...")
        
        try:
            # Create NER pipeline (note: base model may not be fine-tuned for NER)
            print("  Creating NER pipeline...")
            
            # Try different approaches for NER
            try:
                # Method 1: Try direct NER pipeline
                self.ner_pipeline = pipeline("ner", 
                                            model=self.model_name, 
                                            tokenizer=self.model_name,
                                            aggregation_strategy="simple",
                                            device=0 if torch.cuda.is_available() else -1)
                
                print("  ✓ NER pipeline created successfully")
                
                # Test on sample urology text
                sample_texts = [
                    "Patient with chronic kidney disease and elevated creatinine of 2.5 mg/dL underwent nephrectomy.",
                    "Prostate-specific antigen levels are elevated at 12.3 ng/mL indicating possible malignancy.",
                    "Urinary tract infection with E. coli requires antibiotic treatment."
                ]
                
                all_entities = []
                for i, text in enumerate(sample_texts):
                    print(f"\n  Sample text {i+1}: {text}")
                    entities = self.ner_pipeline(text)
                    
                    if entities:
                        print("  Identified entities:")
                        for entity in entities:
                            print(f"    - {entity['word']}: {entity['entity_group']} (confidence: {entity['score']:.3f})")
                        all_entities.extend(entities)
                    else:
                        print("    No entities identified")
                
                return all_entities
                
            except Exception as e1:
                print(f"  ✗ Direct NER pipeline failed: {e1}")
                print("  ℹ Note: The base BiomedBERT model is not fine-tuned for NER")
                print("    For optimal NER performance, use a model specifically fine-tuned for medical NER")
                return None
                
        except Exception as e:
            print(f"  ✗ NER testing failed: {e}")
            return None
    
    def evaluate_domain_adaptation(self) -> Dict:
        """
        Evaluate how well the model understands urology-specific terminology
        """
        print("\n📊 Evaluating domain adaptation for urology...")
        
        try:
            # Medical vs general terminology comparison
            medical_terms = [
                "nephrology", "urology", "creatinine", "glomerular filtration",
                "proteinuria", "hematuria", "nephrectomy", "cystoscopy"
            ]
            
            general_terms = [
                "doctor", "hospital", "patient", "treatment",
                "medicine", "health", "disease", "surgery"
            ]
            
            # Clinical context
            clinical_context = "Patient evaluation in nephrology clinic for kidney function assessment"
            
            print("  Computing embeddings for domain analysis...")
            
            # Get embeddings
            clinical_emb = self.get_embeddings([clinical_context])
            medical_embs = self.get_embeddings(medical_terms)
            general_embs = self.get_embeddings(general_terms)
            
            # Calculate similarities
            medical_sims = cosine_similarity(clinical_emb, medical_embs).flatten()
            general_sims = cosine_similarity(clinical_emb, general_embs).flatten()
            
            results = {
                'medical_term_avg_similarity': float(np.mean(medical_sims)),
                'general_term_avg_similarity': float(np.mean(general_sims)),
                'domain_specificity_ratio': float(np.mean(medical_sims) / np.mean(general_sims)) if np.mean(general_sims) > 0 else 0,
                'top_medical_matches': [(medical_terms[i], float(medical_sims[i])) 
                                      for i in np.argsort(medical_sims)[-3:][::-1]],
                'top_general_matches': [(general_terms[i], float(general_sims[i])) 
                                      for i in np.argsort(general_sims)[-3:][::-1]]
            }
            
            print(f"  ✓ Domain specificity ratio: {results['domain_specificity_ratio']:.3f}")
            
            return results
            
        except Exception as e:
            print(f"  ✗ Error in domain adaptation evaluation: {e}")
            return {}
    
    def generate_evaluation_report(self) -> str:
        """
        Generate comprehensive evaluation report
        """
        print("\n📋 Generating comprehensive evaluation report...")
        
        # Load model
        if not self.load_model():
            return "❌ Failed to load model for evaluation."
        
        print(f"Successfully loaded model: {self.model_name}")
        
        # Run evaluations
        similarity_results = self.evaluate_semantic_similarity()
        ner_results = self.test_named_entity_recognition()
        domain_results = self.evaluate_domain_adaptation()
        
        # Create report
        report = f"""
=== BiomedBERT Urology Ontology Evaluation Report ===

Model: {self.model_name}
Device: {self.device}
Evaluation Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## 1. SEMANTIC SIMILARITY ANALYSIS
"""
        
        if similarity_results:
            for category, results in similarity_results.items():
                report += f"\n### {category.upper().replace('_', ' ')}\n"
                report += f"Average Similarity: {results['avg_similarity']:.3f}\n"
                report += f"Max Similarity: {results['max_similarity']:.3f}\n"
                report += f"Min Similarity: {results['min_similarity']:.3f}\n"
                report += f"Std Deviation: {results['std_similarity']:.3f}\n"
                report += "Top Matches:\n"
                for match in results['best_matches'][:3]:
                    report += f"  - {match['text'][:50]}... → {match['best_term']} ({match['similarity']:.3f})\n"
        else:
            report += "❌ Semantic similarity analysis failed\n"
        
        report += f"\n## 2. DOMAIN ADAPTATION ANALYSIS\n"
        if domain_results:
            report += f"Medical Term Similarity: {domain_results['medical_term_avg_similarity']:.3f}\n"
            report += f"General Term Similarity: {domain_results['general_term_avg_similarity']:.3f}\n"
            report += f"Domain Specificity Ratio: {domain_results['domain_specificity_ratio']:.3f}\n"
            report += "\nTop Medical Term Matches:\n"
            for term, score in domain_results['top_medical_matches']:
                report += f"  - {term}: {score:.3f}\n"
        else:
            report += "❌ Domain adaptation analysis failed\n"
        
        report += f"\n## 3. NAMED ENTITY RECOGNITION\n"
        if ner_results:
            report += f"✓ NER pipeline functional\n"
            report += f"Entities detected: {len(ner_results)}\n"
            if ner_results:
                unique_types = set(e['entity_group'] for e in ner_results)
                report += f"Entity types found: {', '.join(unique_types)}\n"
        else:
            report += "⚠ NER requires fine-tuning for optimal performance\n"
            report += "  Recommendation: Use a medical NER-specific model\n"
        
        report += f"\n## 4. RECOMMENDATIONS\n"
        
        if domain_results:
            domain_ratio = domain_results['domain_specificity_ratio']
            if domain_ratio > 1.2:
                report += "✓ Excellent domain adaptation for urology terminology\n"
            elif domain_ratio > 1.1:
                report += "✓ Good domain adaptation for urology terminology\n"
            elif domain_ratio > 1.0:
                report += "⚠ Moderate domain adaptation - consider fine-tuning\n"
            else:
                report += "✗ Poor domain adaptation - fine-tuning recommended\n"
        
        if similarity_results:
            avg_similarity = np.mean([r['avg_similarity'] for r in similarity_results.values()])
            if avg_similarity > 0.7:
                report += "✓ Strong semantic understanding of urology concepts\n"
            elif avg_similarity > 0.5:
                report += "⚠ Moderate semantic understanding\n"
            else:
                report += "✗ Weak semantic understanding - consider alternative models\n"
            
            report += f"\nOverall Model Suitability: "
            domain_ratio = domain_results.get('domain_specificity_ratio', 0) if domain_results else 0
            if domain_ratio > 1.1 and avg_similarity > 0.6:
                report += "EXCELLENT - Highly suitable for urology ontology tasks\n"
            elif domain_ratio > 1.0 and avg_similarity > 0.5:
                report += "GOOD - Suitable with potential for fine-tuning\n"
            elif avg_similarity > 0.4:
                report += "MODERATE - May benefit from fine-tuning\n"
            else:
                report += "POOR - Consider alternative models or extensive fine-tuning\n"
        
        report += f"\n## 5. TECHNICAL DETAILS\n"
        report += f"Model Parameters: {sum(p.numel() for p in self.model.parameters()) if self.model else 'N/A'}\n"
        report += f"Vocabulary Size: {len(self.tokenizer) if self.tokenizer else 'N/A'}\n"
        report += f"Max Sequence Length: {self.tokenizer.model_max_length if self.tokenizer else 'N/A'}\n"
        
        return report


def main():
    """
    Main function to run the evaluation
    """
    print("🔬 BiomedBERT Urology Ontology Evaluator (Fixed Version)")
    print("=" * 60)
    
    # You can specify different models here:
    model_options = {
        1: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract",
        2: "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", 
        3: "emilyalsentzer/Bio_ClinicalBERT"
    }
    
    print("Available models:")
    for key, model in model_options.items():
        print(f"  {key}. {model}")
    
    try:
        choice = input("\nSelect model (1-3, or press Enter for default): ").strip()
        if choice and choice.isdigit() and int(choice) in model_options:
            selected_model = model_options[int(choice)]
        else:
            selected_model = model_options[1]  # Default
            
        print(f"Selected: {selected_model}")
        
    except (KeyboardInterrupt, EOFError):
        selected_model = model_options[1]  # Default if interrupted
        print(f"\nUsing default: {selected_model}")
    
    # Initialize evaluator with selected model
    evaluator = UrologyBiomedBERTEvaluator(selected_model)
    
    # Generate and display report
    try:
        report = evaluator.generate_evaluation_report()
        print(report)
        
        # Save report to file
        filename = 'biomed_bert_urology_evaluation_fixed.txt'
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"\n💾 Report saved to '{filename}'")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install transformers torch scikit-learn pandas numpy")
        print("\nAlso ensure you have a stable internet connection to download the model.")


if __name__ == "__main__":
    main()