import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import matplotlib.pyplot as plt

# Knowledge Distillation Loss
class DistillationLoss(nn.Module):
    def __init__(self, alpha=0.7, temperature=4.0):
        """
        Knowledge Distillation Loss Function
        Args:
            alpha: Weight for distillation loss vs student loss
            temperature: Temperature for softening predictions
        """
        super(DistillationLoss, self).__init__()
        self.alpha = alpha
        self.temperature = temperature
        self.kl_div = nn.KLDivLoss(reduction='batchmean')
        self.ce_loss = nn.CrossEntropyLoss()
    
    def forward(self, student_logits, teacher_logits, labels):
        # Soften predictions with temperature
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        
        # Distillation loss (KL divergence)
        distill_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)
        
        # Standard cross-entropy loss
        student_loss = self.ce_loss(student_logits, labels)
        
        # Combined loss
        total_loss = self.alpha * distill_loss + (1 - self.alpha) * student_loss
        
        return total_loss, distill_loss, student_loss

# Teacher Model (Large ResNet)
class TeacherModel(nn.Module):
    def __init__(self, num_classes=10):
        super(TeacherModel, self).__init__()
        self.backbone = torchvision.models.resnet34(pretrained=False)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Linear(512, num_classes)
    
    def forward(self, x):
        return self.backbone(x)

# Student Model (Small CNN)
class StudentModel(nn.Module):
    def __init__(self, num_classes=10):
        super(StudentModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(0.5)
        
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        
        x = x.view(-1, 128 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x

# Training Functions
def train_teacher(model, train_loader, val_loader, epochs=10, lr=0.001, device='cpu'):
    """Train the teacher model"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    model.to(device)
    train_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Teacher Epoch {epoch+1}/{epochs}')
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        val_acc = evaluate_model(model, val_loader, device)
        train_losses.append(running_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

def train_with_distillation(teacher_model, student_model, train_loader, val_loader, 
                          epochs=15, lr=0.001, alpha=0.7, temperature=4.0, device='cpu'):
    """Train student model with knowledge distillation"""
    criterion = DistillationLoss(alpha=alpha, temperature=temperature)
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    
    teacher_model.eval()  # Teacher always in eval mode
    teacher_model.to(device)
    student_model.to(device)
    
    train_losses, val_accuracies = [], []
    distill_losses, student_losses = [], []
    
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        running_distill = 0.0
        running_student = 0.0
        
        pbar = tqdm(train_loader, desc=f'Distillation Epoch {epoch+1}/{epochs}')
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            # Get teacher predictions (no gradients)
            with torch.no_grad():
                teacher_logits = teacher_model(data)
            
            # Get student predictions
            student_logits = student_model(data)
            
            # Calculate losses
            total_loss, distill_loss, student_loss = criterion(
                student_logits, teacher_logits, labels
            )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            running_distill += distill_loss.item()
            running_student += student_loss.item()
            
            pbar.set_postfix({
                'Total': f'{total_loss.item():.4f}',
                'Distill': f'{distill_loss.item():.4f}',
                'Student': f'{student_loss.item():.4f}'
            })
        
        # Validation
        val_acc = evaluate_model(student_model, val_loader, device)
        train_losses.append(running_loss / len(train_loader))
        distill_losses.append(running_distill / len(train_loader))
        student_losses.append(running_student / len(train_loader))
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Total Loss: {train_losses[-1]:.4f}, '
              f'Distill: {distill_losses[-1]:.4f}, Student: {student_losses[-1]:.4f}, '
              f'Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies, distill_losses, student_losses

def train_student_baseline(student_model, train_loader, val_loader, 
                          epochs=15, lr=0.001, device='cpu'):
    """Train student model without distillation (baseline)"""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    
    student_model.to(device)
    train_losses, val_accuracies = [], []
    
    for epoch in range(epochs):
        student_model.train()
        running_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Baseline Epoch {epoch+1}/{epochs}')
        for data, labels in pbar:
            data, labels = data.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = student_model(data)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        # Validation
        val_acc = evaluate_model(student_model, val_loader, device)
        train_losses.append(running_loss / len(train_loader))
        val_accuracies.append(val_acc)
        
        print(f'Epoch {epoch+1}: Train Loss: {train_losses[-1]:.4f}, Val Acc: {val_acc:.4f}')
    
    return train_losses, val_accuracies

def evaluate_model(model, test_loader, device='cpu'):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, labels in test_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return correct / total

def count_parameters(model):
    """Count model parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# Data Loading
def get_cifar10_loaders(batch_size=128):
    """Get CIFAR-10 data loaders"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return trainloader, testloader

# Main Execution
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Get data
    train_loader, test_loader = get_cifar10_loaders(batch_size=128)
    
    # Initialize models
    teacher_model = TeacherModel(num_classes=10)
    student_model = StudentModel(num_classes=10)
    student_baseline = StudentModel(num_classes=10)
    
    print(f"Teacher parameters: {count_parameters(teacher_model):,}")
    print(f"Student parameters: {count_parameters(student_model):,}")
    print(f"Compression ratio: {count_parameters(teacher_model) / count_parameters(student_model):.2f}x")
    
    # Train teacher model
    print("\n=== Training Teacher Model ===")
    teacher_losses, teacher_accs = train_teacher(
        teacher_model, train_loader, test_loader, epochs=10, device=device
    )
    
    teacher_final_acc = evaluate_model(teacher_model, test_loader, device)
    print(f"Teacher final accuracy: {teacher_final_acc:.4f}")
    
    # Train student with distillation
    print("\n=== Training Student with Distillation ===")
    distill_losses, distill_accs, d_losses, s_losses = train_with_distillation(
        teacher_model, student_model, train_loader, test_loader,
        epochs=15, alpha=0.7, temperature=4.0, device=device
    )
    
    student_distill_acc = evaluate_model(student_model, test_loader, device)
    print(f"Student (distilled) final accuracy: {student_distill_acc:.4f}")
    
    # Train student baseline
    print("\n=== Training Student Baseline ===")
    baseline_losses, baseline_accs = train_student_baseline(
        student_baseline, train_loader, test_loader, epochs=15, device=device
    )
    
    baseline_acc = evaluate_model(student_baseline, test_loader, device)
    print(f"Student (baseline) final accuracy: {baseline_acc:.4f}")
    
    # Results summary
    print("\n=== Results Summary ===")
    print(f"Teacher accuracy: {teacher_final_acc:.4f}")
    print(f"Student (distilled) accuracy: {student_distill_acc:.4f}")
    print(f"Student (baseline) accuracy: {baseline_acc:.4f}")
    print(f"Distillation improvement: {student_distill_acc - baseline_acc:.4f}")
    print(f"Knowledge transfer efficiency: {student_distill_acc / teacher_final_acc:.4f}")
    
    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(teacher_accs, label='Teacher', linewidth=2)
    plt.plot(distill_accs, label='Student (Distilled)', linewidth=2)
    plt.plot(baseline_accs, label='Student (Baseline)', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Validation Accuracy')
    plt.title('Model Accuracies')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(distill_losses, label='Total Loss', linewidth=2)
    plt.plot(d_losses, label='Distillation Loss', linewidth=2)
    plt.plot(s_losses, label='Student Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Distillation Loss Components')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 3, 3)
    plt.plot(distill_losses, label='Distilled Student', linewidth=2)
    plt.plot(baseline_losses, label='Baseline Student', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.title('Training Loss Comparison')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('knowledge_distillation_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Save models
    torch.save(teacher_model.state_dict(), 'teacher_model.pth')
    torch.save(student_model.state_dict(), 'student_distilled.pth')
    torch.save(student_baseline.state_dict(), 'student_baseline.pth')
    
    print("Models saved successfully!")