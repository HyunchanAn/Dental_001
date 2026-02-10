
import os
import sys
import time
import copy
import pandas as pd
from datetime import datetime
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from src import config

# --- Configuration V2 ---
DATA_DIR = 'Aariz_CVM_Clean'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
BATCH_SIZE = 32      # RTX 5080(16GB) 최적화: 32로 상향
EPOCHS = 100
LEARNING_RATE = 5e-5 # 정밀 학습을 위해 기존보다 낮은 LR 설정
NUM_CLASSES = 6
IMG_SIZE = 768       # 512 -> 768 상향!!

# Logging
os.makedirs("training_log", exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
LOG_FILE = f"training_log/{TIMESTAMP}_cvm_v2_768px_weighted_log.csv"

# --- Model Definition (EfficientNet with CORAL) ---
class CoralEfficientNet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(CoralEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features) + self.bias
        return logits

def coral_loss(logits, levels):
    val = (-torch.sum((torch.nn.functional.logsigmoid(logits)*levels
                      + (torch.nn.functional.logsigmoid(logits) - logits)*(1-levels)), dim=1))
    return torch.mean(val)

def label_to_levels(label, num_classes):
    levels = torch.zeros(label.size(0), num_classes - 1).to(DEVICE)
    for i in range(label.size(0)):
        if label[i] > 0:
            levels[i, :label[i]] = 1
    return levels

def proba_to_label(logits):
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels

# --- Weight Sampler Logic ---
def get_weighted_sampler(dataset):
    # 클래스별 이미지 개수 카운트
    targets = dataset.targets
    class_count = [0] * NUM_CLASSES
    for t in targets:
        class_count[t] += 1
    
    # 가중치 계산 (역수)
    weights = 1. / torch.tensor(class_count, dtype=torch.float)
    samples_weights = weights[targets]
    
    sampler = WeightedRandomSampler(weights=samples_weights, num_samples=len(samples_weights), replacement=True)
    return sampler

# --- Training Loop ---
def train_model():
    print(f"🚀 Starting CVM V2 Training on {DEVICE}...")
    print(f"Resolution: {IMG_SIZE}px | Batch: {BATCH_SIZE} | LR: {LEARNING_RATE}")
    
    # 1. Data Augmentation (V2: More Geometric)
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # 2. Datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x]) 
                      for x in ['train', 'valid']}
    
    # 학습셋 전용 Sampler 생성 (불균형 해결)
    train_sampler = get_weighted_sampler(image_datasets['train'])
    
    dataloaders = {
        'train': DataLoader(image_datasets['train'], batch_size=BATCH_SIZE, sampler=train_sampler, num_workers=4),
        'valid': DataLoader(image_datasets['valid'], batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    }
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    print(f"Dataset sizes: {dataset_sizes}")

    # 3. Model & Optimizer
    model = CoralEfficientNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=5e-4) # Regularization 강화
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=8)

    # 4. Training
    best_kappa = -1.0
    
    with open(LOG_FILE, 'w') as f:
        f.write("Epoch,Train_Loss,Valid_Loss,Valid_Kappa,Valid_Acc,LR\n")
        
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            all_preds, all_labels = [], []
            
            pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{EPOCHS} [{phase}]", leave=False)
            
            for inputs, labels in pbar:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    levels = label_to_levels(labels, NUM_CLASSES)
                    loss = coral_loss(logits, levels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                preds = proba_to_label(logits)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
            epoch_acc = accuracy_score(all_labels, all_preds)

            if phase == 'train':
                train_loss, train_kappa = epoch_loss, epoch_kappa
            else:
                valid_loss, valid_kappa, valid_acc = epoch_loss, epoch_kappa, epoch_acc
                scheduler.step(valid_loss)
                
                # Save Best (파일 이름 구분)
                if valid_kappa > best_kappa:
                    best_kappa = valid_kappa
                    torch.save(model.state_dict(), f"checkpoints/best_cvm_v2_768px.pth")
                    print(f"  🌟 New Best (Valid Kappa): {best_kappa:.4f} (Saved as v2_768px)")

        print(f"E{epoch+1:02d} | T-Loss: {train_loss:.4f} | V-Loss: {valid_loss:.4f} | V-Kap: {valid_kappa:.4f} | V-Acc: {valid_acc:.4f} | LR: {current_lr:.6f}")
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"{epoch+1},{train_loss},{valid_loss},{valid_kappa},{valid_acc},{current_lr}\n")

    print(f"🎉 V2 Training Complete! Best Valid Kappa: {best_kappa:.4f}")

if __name__ == '__main__':
    train_model()
