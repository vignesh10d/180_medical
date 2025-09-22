import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoTokenizer, AutoModel, AutoConfig,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW  # Fixed import
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report
import warnings
warnings.filterwarnings('ignore')

class DistillationLoss(nn.Module):
    """Knowledge Distillation Loss for transformer models"""
    def __init__(self, alpha=0.7, temperature=4.0):
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Temperature-scaled soft predictions
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss (knowledge transfer)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Standard classification loss
        student_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return total_loss, distill_loss, student_loss

class TeacherModel(nn.Module):
    """Teacher model using BiomedBERT"""
    def __init__(self, model_name="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext", num_classes=2):
        super(TeacherModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.bert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class StudentModel(nn.Module):
    """Student model using DistilBERT"""
    def __init__(self, model_name="distilbert-base-uncased", num_classes=2):
        super(StudentModel, self).__init__()
        self.config = AutoConfig.from_pretrained(model_name)
        self.distilbert = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(self.config.hidden_size, num_classes)
        
    def forward(self, input_ids, attention_mask=None):
        outputs = self.distilbert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Use [CLS] token representation
        pooled_output = outputs.last_hidden_state[:, 0, :]  # [CLS] token
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        
        return logits

class TextDataset(Dataset):
    """Custom dataset for text classification"""
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]
        
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

def prepare_data(dataset_name="imdb", max_samples=1000):
    """Prepare dataset for training"""
    print(f"Loading {dataset_name} dataset...")
    
    if dataset_name == "imdb":
        dataset = load_dataset("imdb")
        
        # Use subset for faster training
        train_texts = dataset['train']['text'][:max_samples]
        train_labels = dataset['train']['label'][:max_samples]
        test_texts = dataset['test']['text'][:max_samples//4]
        test_labels = dataset['test']['label'][:max_samples//4]
        
    elif dataset_name == "medical":
        # You can replace this with a medical text dataset
        # For demo, we'll use a subset of imdb with modified labels
        dataset = load_dataset("imdb")
        train_texts = dataset['train']['text'][:max_samples]
        train_labels = dataset['train']['label'][:max_samples]
        test_texts = dataset['test']['text'][:max_samples//4]
        test_labels = dataset['test']['label'][:max_samples//4]
        
    return train_texts, train_labels, test_texts, test_labels

def train_teacher(model, train_loader, val_loader, epochs=3, lr=2e-5, device='cpu'):
    """Train the teacher model"""
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Teacher Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

def train_with_distillation(teacher_model, student_model, train_loader, val_loader,
                          epochs=5, lr=2e-5, alpha=0.7, temperature=4.0, device='cpu'):
    """Train student model with knowledge distillation"""
    teacher_model.eval()
    teacher_model.to(device)
    student_model.to(device)
    
    optimizer = AdamW(student_model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    train_losses, val_accuracies = [], []
    distill_losses, student_losses = [], []
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        total_distill = 0
        total_student = 0
        
        pbar = tqdm(train_loader, desc=f'Distillation Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Get student predictions
            student_logits = student_model(input_ids=input_ids, attention_mask=attention_mask)
            
            # Calculate distillation loss
            total_loss_batch, distill_loss, student_loss = criterion(
                student_logits, teacher_logits, labels
            )
            
            optimizer.zero_grad()
            total_loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += total_loss_batch.item()
            total_distill += distill_loss.item()
            total_student += student_loss.item()
            
            pbar.set_postfix({
                'Total': f'{total_loss_batch.item():.4f}',
                'Distill': f'{distill_loss.item():.4f}',
                'Student': f'{student_loss.item():.4f}'
            })
        
        # Validation
        val_acc = evaluate_model(student_model, val_loader, device)
        avg_total = total_loss / len(train_loader)
        avg_distill = total_distill / len(train_loader)
        avg_student = total_student / len(train_loader)
        
        train_losses.append(avg_total)
        distill_losses.append(avg_distill)
        student_losses.append(avg_student)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Total: {avg_total:.4f}, '
              f'Distill: {avg_distill:.4f}, Student: {avg_student:.4f}, '
              f'Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies, distill_losses, student_losses

def train_student_baseline(student_model, train_loader, val_loader, 
                          epochs=5, lr=2e-5, device='cpu'):
    """Train student model without distillation"""
    student_model.to(device)
    optimizer = AdamW(student_model.parameters(), lr=lr)
    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=0, num_training_steps=total_steps
    )
    
    criterion = nn.CrossEntropyLoss()
    train_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        student_model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Baseline Epoch {epoch+1}/{epochs}')
        for batch in pbar:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            logits = student_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(logits, labels)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        val_acc = evaluate_model(student_model, val_loader, device)
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy"""
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            logits = model(input_ids=input_ids, attention_mask=attention_mask)
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def main():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Prepare data
    train_texts, train_labels, test_texts, test_labels = prepare_data(
        dataset_name="imdb", max_samples=1000
    )
    
    # Initialize tokenizers
    teacher_tokenizer = AutoTokenizer.from_pretrained(
        "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
    )
    student_tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Create datasets
    train_dataset_teacher = TextDataset(train_texts, train_labels, teacher_tokenizer, max_length=256)
    test_dataset_teacher = TextDataset(test_texts, test_labels, teacher_tokenizer, max_length=256)
    
    train_dataset_student = TextDataset(train_texts, train_labels, student_tokenizer, max_length=256)
    test_dataset_student = TextDataset(test_texts, test_labels, student_tokenizer, max_length=256)
    
    # Create data loaders
    batch_size = 8
    train_loader_teacher = DataLoader(train_dataset_teacher, batch_size=batch_size, shuffle=True)
    test_loader_teacher = DataLoader(test_dataset_teacher, batch_size=batch_size, shuffle=False)
    
    train_loader_student = DataLoader(train_dataset_student, batch_size=batch_size, shuffle=True)
    test_loader_student = DataLoader(test_dataset_student, batch_size=batch_size, shuffle=False)
    
    # Initialize models
    teacher_model = TeacherModel(num_classes=2)
    student_model = StudentModel(num_classes=2)
    student_baseline = StudentModel(num_classes=2)
    
    print(f"Teacher parameters: {count_parameters(teacher_model):,}")
    print(f"Student parameters: {count_parameters(student_model):,}")
    print(f"Compression ratio: {count_parameters(teacher_model) / count_parameters(student_model):.2f}x")
    
    # Train teacher model
    print("\n=== Training Teacher Model (BiomedBERT) ===")
    teacher_losses, teacher_accs = train_teacher(
        teacher_model, train_loader_teacher, test_loader_teacher, epochs=2, device=device
    )
    
    teacher_final_acc = evaluate_model(teacher_model, test_loader_teacher, device)
    print(f"Teacher final accuracy: {teacher_final_acc:.4f}")
    
    # Train student with distillation
    print("\n=== Training Student with Distillation (DistilBERT) ===")
    # Note: Using student tokenizer/loader for student training
    distill_losses, distill_accs, d_losses, s_losses = train_with_distillation(
        teacher_model, student_model, train_loader_student, test_loader_student,
        epochs=3, alpha=0.7, temperature=4.0, device=device
    )
    
    student_distill_acc = evaluate_model(student_model, test_loader_student, device)
    print(f"Student (distilled) final accuracy: {student_distill_acc:.4f}")
    
    # Train student baseline
    print("\n=== Training Student Baseline (DistilBERT) ===")
    baseline_losses, baseline_accs = train_student_baseline(
        student_baseline, train_loader_student, test_loader_student, epochs=3, device=device
    )
    
    baseline_acc = evaluate_model(student_baseline, test_loader_student, device)
    print(f"Student (baseline) final accuracy: {baseline_acc:.4f}")
    
    # Results summary
    print("\n=== Results Summary ===")
    print(f"Teacher (BiomedBERT) accuracy: {teacher_final_acc:.4f}")
    print(f"Student (DistilBERT + KD) accuracy: {student_distill_acc:.4f}")
    print(f"Student (DistilBERT baseline) accuracy: {baseline_acc:.4f}")
    print(f"Distillation improvement: {student_distill_acc - baseline_acc:.4f}")
    print(f"Knowledge transfer efficiency: {student_distill_acc / teacher_final_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    # Pad shorter lists for plotting
    max_epochs = max(len(teacher_accs), len(distill_accs), len(baseline_accs))
    teacher_accs_padded = teacher_accs + [teacher_accs[-1]] * (max_epochs - len(teacher_accs))
    
    plt.subplot(1, 3, 1)
    plt.plot(teacher_accs_padded, label='Teacher (BiomedBERT)', linewidth=2, marker='o')
    plt.plot(distill_accs, label='Student (Distilled)', linewidth=2, marker='s')
    plt.plot(baseline_accs, label='Student (Baseline)', linewidth=2, marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Performance Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(distill_losses, label='Total Loss', linewidth=2, marker='o')
    plt.plot(d_losses, label='Distillation Loss', linewidth=2, marker='s')
    plt.plot(s_losses, label='Student Loss', linewidth=2, marker='^')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Distillation Loss Components')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    model_names = ['Teacher\n(BiomedBERT)', 'Student\n(Distilled)', 'Student\n(Baseline)']
    accuracies = [teacher_final_acc, student_distill_acc, baseline_acc]
    colors = ['#2E86AB', '#A23B72', '#F18F01']
    
    bars = plt.bar(model_names, accuracies, color=colors, alpha=0.8)
    plt.ylabel('Final Accuracy')
    plt.title('Final Model Comparison')
    plt.ylim(0, 1)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('hf_knowledge_distillation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save models
    torch.save(teacher_model.state_dict(), 'teacher_biomedbert.pth')
    torch.save(student_model.state_dict(), 'student_distilbert_distilled.pth')
    torch.save(student_baseline.state_dict(), 'student_distilbert_baseline.pth')
    
    print("Models saved successfully!")

if __name__ == "__main__":
    main()