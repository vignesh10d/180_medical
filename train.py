import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModelForTokenClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup
)
import numpy as np
from torch.utils.data import DataLoader, Dataset
import json
import os
from typing import List, Dict, Tuple
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import warnings

class KnowledgeDistillationLoss(nn.Module):
    """Custom loss function for knowledge distillation in NER"""

    def __init__(self, temperature=4.0, alpha=0.7, beta=0.3):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.beta = beta
        self.ce_loss = nn.CrossEntropyLoss(ignore_index=-100)
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, labels):
        batch_size, seq_len, num_classes = student_logits.shape
        
        student_logits_flat = student_logits.view(-1, num_classes)
        teacher_logits_flat = teacher_logits.view(-1, num_classes)
        labels_flat = labels.view(-1)
        
        mask = (labels_flat != -100)
        
        # Hard target loss
        hard_loss = self.ce_loss(student_logits_flat, labels_flat)
        
        # Soft target loss
        student_soft = F.log_softmax(student_logits_flat / self.temperature, dim=-1)
        teacher_soft = F.softmax(teacher_logits_flat / self.temperature, dim=-1)
        
        if mask.sum() > 0:
            student_soft_masked = student_soft[mask]
            teacher_soft_masked = teacher_soft[mask]
            soft_loss = self.kl_loss(student_soft_masked, teacher_soft_masked)
            soft_loss *= (self.temperature ** 2)
        else:
            soft_loss = torch.tensor(0.0, device=student_logits.device)
        
        total_loss = self.alpha * soft_loss + self.beta * hard_loss
        
        return {
            'total_loss': total_loss,
            'hard_loss': hard_loss,
            'soft_loss': soft_loss
        }

class TeacherModel:
    """Teacher model wrapper"""

    def __init__(self, model_name_or_path, num_labels):
        self.model = AutoModelForTokenClassification.from_pretrained(
            model_name_or_path, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def get_predictions(self, input_ids, attention_mask):
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.logits

class StudentModel(nn.Module):
    """Student model for distillation"""

    def __init__(self, model_name, num_labels):
        super().__init__()
        self.base_model = AutoModelForTokenClassification.from_pretrained(
            model_name, num_labels=num_labels
        )

    def forward(self, input_ids, attention_mask, labels=None):
        return self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )

class MedicalNERDataset(Dataset):
    """Dataset class for medical NER data"""

    def __init__(self, texts, labels, tokenizer, max_length=128, label2id=None):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

        if label2id is None:
            unique_labels = set()
            for label_seq in labels:
                unique_labels.update(label_seq)
            self.label2id = {label: idx for idx, label in enumerate(sorted(unique_labels))}
        else:
            self.label2id = label2id

        self.id2label = {v: k for k, v in self.label2id.items()}
        self.num_labels = len(self.label2id)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        labels = self.labels[idx]

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt',
            is_split_into_words=True
        )

        aligned_labels = self.align_labels_with_tokens(text, labels, encoding)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(aligned_labels, dtype=torch.long)
        }

    def align_labels_with_tokens(self, words, labels, encoding):
        aligned_labels = []
        word_ids = encoding.word_ids()
        previous_word_idx = None
        
        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != previous_word_idx:
                if word_idx < len(labels):
                    label = labels[word_idx]
                    if isinstance(label, str):
                        label = self.label2id.get(label, 0)
                    aligned_labels.append(label)
                else:
                    aligned_labels.append(0)
            else:
                aligned_labels.append(-100)
            previous_word_idx = word_idx

        return aligned_labels

class DistillationTrainer:
    """Main trainer for knowledge distillation"""

    def __init__(self, teacher_model, student_model, tokenizer, distill_config=None):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.tokenizer = tokenizer
        
        self.config = distill_config or {
            'temperature': 4.0,
            'alpha': 0.7,
            'beta': 0.3,
            'learning_rate': 5e-5,
            'num_epochs': 5,
            'batch_size': 16,
        }
        
        self.distill_loss = KnowledgeDistillationLoss(
            temperature=self.config['temperature'],
            alpha=self.config['alpha'],
            beta=self.config['beta']
        )
        
        self.training_history = {
            'epoch': [], 'train_loss': [], 'val_f1': [], 'learning_rate': []
        }
        self.best_val_f1 = 0.0
        self.patience = 3
        self.patience_counter = 0

    def train_step(self, batch):
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']

        teacher_logits = self.teacher_model.get_predictions(input_ids, attention_mask)
        student_outputs = self.student_model(
            input_ids=input_ids, attention_mask=attention_mask, labels=labels
        )
        
        return self.distill_loss(student_outputs.logits, teacher_logits, labels)

    def train(self, train_dataloader, val_dataloader=None, save_path="./distilled_model"):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {device}")
        
        # Move models to device and setup
        self.teacher_model.model.to(device)
        self.teacher_model.model.eval()
        self.student_model.to(device)
        
        # Freeze teacher model
        for param in self.teacher_model.model.parameters():
            param.requires_grad = False

        optimizer = torch.optim.AdamW(
            self.student_model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=0.01
        )
        
        total_steps = len(train_dataloader) * self.config['num_epochs']
        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps
        )

        print(f"Training on {len(train_dataloader)} batches for {self.config['num_epochs']} epochs")
        if val_dataloader:
            print(f"Validation on {len(val_dataloader)} batches")

        for epoch in range(self.config['num_epochs']):
            print(f"\nEpoch {epoch + 1}/{self.config['num_epochs']}")
            print("-" * 50)
            
            self.student_model.train()
            epoch_losses = []

            progress_bar = tqdm(train_dataloader, desc=f"Training")
            for batch in progress_bar:
                batch = {k: v.to(device) for k, v in batch.items()}
                
                loss_dict = self.train_step(batch)
                
                optimizer.zero_grad()
                loss_dict['total_loss'].backward()
                torch.nn.utils.clip_grad_norm_(self.student_model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                
                epoch_losses.append(loss_dict['total_loss'].item())
                progress_bar.set_postfix({
                    'Loss': f"{loss_dict['total_loss'].item():.4f}",
                    'Hard': f"{loss_dict['hard_loss'].item():.4f}",
                    'Soft': f"{loss_dict['soft_loss'].item():.4f}"
                })

            avg_train_loss = np.mean(epoch_losses)
            print(f"Average Training Loss: {avg_train_loss:.4f}")

            # Validation
            if val_dataloader:
                val_metrics = self.evaluate(val_dataloader, device)
                val_f1 = val_metrics['f1']
                
                print(f"Validation F1: {val_f1:.4f}")
                print(f"Validation Precision: {val_metrics['precision']:.4f}")
                print(f"Validation Recall: {val_metrics['recall']:.4f}")
                
                # Update history
                self.training_history['epoch'].append(epoch + 1)
                self.training_history['train_loss'].append(avg_train_loss)
                self.training_history['val_f1'].append(val_f1)
                self.training_history['learning_rate'].append(scheduler.get_last_lr()[0])
                
                # Early stopping
                if val_f1 > self.best_val_f1:
                    self.best_val_f1 = val_f1
                    self.patience_counter = 0
                    self.save_student_model(f"{save_path}/best_model")
                    print(f"New best model! F1: {val_f1:.4f}")
                else:
                    self.patience_counter += 1
                    
                if self.patience_counter >= self.patience:
                    print(f"Early stopping triggered after {epoch + 1} epochs")
                    break

            self.save_student_model(f"{save_path}/epoch_{epoch + 1}")

        print("Training completed!")
        return self.training_history

    def evaluate(self, dataloader, device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
        self.student_model.eval()
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.student_model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                predictions = torch.argmax(outputs.logits, dim=-1)
                
                for pred, label, mask in zip(predictions, batch['labels'], batch['attention_mask']):
                    valid_length = mask.sum().item()
                    pred_seq = pred[:valid_length].cpu().numpy()
                    label_seq = label[:valid_length].cpu().numpy()
                    
                    valid_indices = label_seq != -100
                    if valid_indices.sum() > 0:
                        all_predictions.extend(pred_seq[valid_indices])
                        all_labels.extend(label_seq[valid_indices])

        # Handle edge case with no valid predictions
        if len(all_predictions) == 0 or len(all_labels) == 0:
            return {'f1': 0.0, 'precision': 0.0, 'recall': 0.0}

        # Suppress sklearn warnings for small datasets
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
            precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
            recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return {'f1': f1, 'precision': precision, 'recall': recall}

    def save_student_model(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.student_model.state_dict(), f"{path}/pytorch_model.bin")
        self.tokenizer.save_pretrained(path)
        self.student_model.base_model.config.save_pretrained(path)
        
        # Save label mappings if available
        if hasattr(self, 'label2id'):
            with open(f"{path}/label_mappings.json", 'w') as f:
                json.dump(self.label2id, f)

    def plot_training_history(self, save_path="training_plots.png"):
        if not self.training_history['epoch']:
            print("No training history to plot")
            return
            
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle('Training History', fontsize=16)

        # Training loss
        ax1.plot(self.training_history['epoch'], self.training_history['train_loss'], 'b-o')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True)

        # Validation F1
        ax2.plot(self.training_history['epoch'], self.training_history['val_f1'], 'g-o')
        ax2.set_title('Validation F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.grid(True)

        # Learning rate
        ax3.plot(self.training_history['epoch'], self.training_history['learning_rate'], 'r-o')
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.set_yscale('log')
        ax3.grid(True)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
        print(f"Training plots saved to {save_path}")

class DataProcessor:
    """Utility for processing medical NER data"""

    @staticmethod
    def load_conll_format(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        sentences = []
        labels = []
        current_sentence = []
        current_labels = []

        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        labels.append(current_labels)
                        current_sentence = []
                        current_labels = []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        current_sentence.append(parts[0])
                        current_labels.append(parts[-1])

            if current_sentence:
                sentences.append(current_sentence)
                labels.append(current_labels)

        return sentences, labels

    @staticmethod
    def load_json_format(file_path: str) -> Tuple[List[List[str]], List[List[str]]]:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        sentences = []
        labels = []

        for item in data:
            if 'tokens' in item and 'labels' in item:
                sentences.append(item['tokens'])
                labels.append(item['labels'])
            elif 'text' in item and 'entities' in item:
                tokens, iob_labels = DataProcessor.entities_to_iob(item['text'], item['entities'])
                sentences.append(tokens)
                labels.append(iob_labels)

        return sentences, labels

    @staticmethod
    def entities_to_iob(text: str, entities: List[Dict]) -> Tuple[List[str], List[str]]:
        tokens = text.split()
        labels = ['O'] * len(tokens)
        entities = sorted(entities, key=lambda x: x['start'])

        char_to_token = {}
        current_pos = 0
        for i, token in enumerate(tokens):
            start_pos = text.find(token, current_pos)
            end_pos = start_pos + len(token)
            for j in range(start_pos, end_pos):
                char_to_token[j] = i
            current_pos = end_pos

        for entity in entities:
            start_char = entity['start']
            end_char = entity['end']
            entity_type = entity['label']

            start_token = char_to_token.get(start_char)
            end_token = char_to_token.get(end_char - 1)

            if start_token is not None and end_token is not None:
                labels[start_token] = f'B-{entity_type}'
                for i in range(start_token + 1, min(end_token + 1, len(labels))):
                    labels[i] = f'I-{entity_type}'

        return tokens, labels

    @staticmethod
    def create_label_mappings(all_labels: List[List[str]]) -> Dict[str, int]:
        unique_labels = set()
        for label_seq in all_labels:
            unique_labels.update(label_seq)
        return {label: idx for idx, label in enumerate(sorted(unique_labels))}

class ProductionInference:
    """Optimized inference for production"""

    def __init__(self, model_path, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        
        config = AutoModelForTokenClassification.from_pretrained(model_path).config
        self.model = StudentModel("distilbert-base-uncased", config.num_labels)
        
        state_dict = torch.load(f"{model_path}/pytorch_model.bin", map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
        
        # Load label mappings
        with open(f"{model_path}/label_mappings.json", 'r') as f:
            self.label2id = json.load(f)
            self.id2label = {v: k for k, v in self.label2id.items()}

    def predict(self, text, max_length=128):
        if isinstance(text, str):
            tokens = text.split()
        else:
            tokens = text

        encoding = self.tokenizer(
            tokens,
            truncation=True,
            padding='max_length',
            max_length=max_length,
            return_tensors='pt',
            is_split_into_words=True
        )

        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=-1)

        # Align predictions with tokens
        word_ids = encoding.word_ids(batch_index=0)
        aligned_labels = []
        current_word_idx = None

        for i, word_idx in enumerate(word_ids):
            if word_idx is not None and word_idx != current_word_idx and i < len(predictions[0]):
                label_id = predictions[0][i].item()
                label = self.id2label.get(label_id, 'O')
                aligned_labels.append(label)
                current_word_idx = word_idx

        return list(zip(tokens, aligned_labels[:len(tokens)]))

    def extract_entities(self, predictions):
        entities = []
        current_entity = None

        for i, (token, label) in enumerate(predictions):
            if label.startswith('B-'):
                if current_entity:
                    entities.append(current_entity)
                current_entity = {
                    'text': token,
                    'label': label[2:],
                    'start': i,
                    'end': i
                }
            elif label.startswith('I-') and current_entity and label[2:] == current_entity['label']:
                current_entity['text'] += ' ' + token
                current_entity['end'] = i
            else:
                if current_entity:
                    entities.append(current_entity)
                    current_entity = None

        if current_entity:
            entities.append(current_entity)

        return entities

def compare_models(teacher_model, student_model, test_dataloader):
    """Compare teacher and student model performance"""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def evaluate_model(model, dataloader):
        model.eval() if hasattr(model, 'eval') else model.model.eval()
        inference_times = []

        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(device) for k, v in batch.items()}
                start_time = time.time()
                
                if hasattr(model, 'get_predictions'):
                    logits = model.get_predictions(batch['input_ids'], batch['attention_mask'])
                else:
                    outputs = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'])
                    logits = outputs.logits
                
                inference_times.append(time.time() - start_time)

        return {'avg_inference_time': np.mean(inference_times)}

    teacher_results = evaluate_model(teacher_model, test_dataloader)
    student_results = evaluate_model(student_model, test_dataloader)
    
    speedup = teacher_results['avg_inference_time'] / student_results['avg_inference_time']
    
    teacher_params = sum(p.numel() for p in teacher_model.model.parameters())
    student_params = sum(p.numel() for p in student_model.parameters())
    compression_ratio = teacher_params / student_params

    print(f"Speedup: {speedup:.2f}x")
    print(f"Teacher parameters: {teacher_params:,}")
    print(f"Student parameters: {student_params:,}")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    return {
        'speedup': speedup,
        'compression_ratio': compression_ratio,
        'teacher_params': teacher_params,
        'student_params': student_params
    }

# Enhanced sample data for better demonstration
def create_sample_data():
    """Create more comprehensive sample data for better training demonstration"""
    texts = [
        ["Patient", "presents", "with", "acute", "myocardial", "infarction"],
        ["History", "of", "hypertension", "and", "diabetes", "mellitus"],
        ["Prescribed", "metformin", "500mg", "twice", "daily"],
        ["CT", "scan", "revealed", "pulmonary", "embolism"],
        ["Blood", "pressure", "elevated", "at", "180/100", "mmHg"],
        ["Patient", "diagnosed", "with", "pneumonia", "and", "fever"],
        ["Administered", "aspirin", "and", "lisinopril", "for", "treatment"],
        ["MRI", "shows", "brain", "lesions", "consistent", "with", "stroke"],
        ["Temperature", "recorded", "at", "102.5", "degrees", "Fahrenheit"],
        ["Laboratory", "results", "show", "elevated", "glucose", "levels"]
    ]
    
    labels = [
        ["O", "O", "O", "B-DISEASE", "I-DISEASE", "I-DISEASE"],
        ["O", "O", "B-DISEASE", "O", "B-DISEASE", "I-DISEASE"],
        ["O", "B-MEDICATION", "B-DOSAGE", "B-FREQUENCY", "I-FREQUENCY"],
        ["B-TEST", "I-TEST", "O", "B-DISEASE", "I-DISEASE"],
        ["B-VITAL", "I-VITAL", "O", "O", "B-VALUE", "B-UNIT"],
        ["O", "O", "O", "B-DISEASE", "O", "B-SYMPTOM"],
        ["O", "B-MEDICATION", "O", "B-MEDICATION", "O", "O"],
        ["B-TEST", "O", "B-ANATOMY", "I-ANATOMY", "O", "O", "B-DISEASE"],
        ["B-VITAL", "O", "O", "B-VALUE", "I-VALUE", "I-VALUE"],
        ["B-TEST", "I-TEST", "O", "O", "B-BIOMARKER", "I-BIOMARKER"]
    ]
    
    return texts, labels

# Complete usage example
def main():
    """Complete end-to-end example"""
    print("Medical NER Knowledge Distillation")
    print("=" * 50)
    
    # Create sample data
    texts, labels = create_sample_data()
    print(f"Created {len(texts)} sample sentences")
    
    # Process data
    label2id = DataProcessor.create_label_mappings(labels)
    print(f"Found {len(label2id)} unique labels: {list(label2id.keys())}")
    
    # Initialize models
    print("Initializing teacher and student models...")
    teacher = TeacherModel("microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext", len(label2id))
    student = StudentModel("distilbert-base-uncased", len(label2id))
    
    # Create datasets with better train/val split
    train_texts, train_labels = texts[:8], labels[:8]  # 8 for training
    val_texts, val_labels = texts[8:], labels[8:]      # 2 for validation
    
    train_dataset = MedicalNERDataset(train_texts, train_labels, teacher.tokenizer, 
                                    max_length=64, label2id=label2id)
    val_dataset = MedicalNERDataset(val_texts, val_labels, teacher.tokenizer,
                                  max_length=64, label2id=label2id)
    
    train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Configure and train
    config = {
        'temperature': 4.0,
        'alpha': 0.7,
        'beta': 0.3,
        'learning_rate': 5e-5,
        'num_epochs': 3,  # Increased for better demonstration
        'batch_size': 2,
    }
    
    trainer = DistillationTrainer(teacher, student, teacher.tokenizer, config)
    trainer.label2id = label2id  # Store for saving
    
    try:
        history = trainer.train(train_dataloader, val_dataloader, "./demo_model")
        
        # Plot training history
        trainer.plot_training_history()
        
        # Compare models
        print("\nComparing teacher vs student model performance:")
        comparison = compare_models(teacher, student, val_dataloader)
        
        print(f"\nResults Summary:")
        print(f"Best F1 Score: {trainer.best_val_f1:.4f}")
        print(f"Model Compression: {comparison['compression_ratio']:.1f}x smaller")
        print(f"Speed Improvement: {comparison['speedup']:.1f}x faster")
        
        # Demo inference
        if trainer.best_val_f1 > 0:
            print("\nTesting inference on sample text...")
            inference = ProductionInference("./demo_model/best_model")
            sample_text = ["Patient", "has", "diabetes", "and", "takes", "insulin"]
            predictions = inference.predict(sample_text)
            entities = inference.extract_entities(predictions)
            
            print(f"Sample text: {' '.join(sample_text)}")
            print(f"Predictions: {predictions}")
            print(f"Extracted entities: {entities}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        print("For production use, provide a larger medical NER dataset.")

if __name__ == "__main__":
    main()