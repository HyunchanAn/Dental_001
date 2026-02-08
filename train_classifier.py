
import os
import time
import copy
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score, mean_absolute_error

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

import config

# --- Configuration ---
DATA_DIR = 'Aariz_CVM_Clean'
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BATCH_SIZE = 16
NUM_WORKERS = 4
EPOCHS = 100
LEARNING_RATE = 1e-4 # Fine-tuning LR
NUM_CLASSES = 6
IMG_SIZE = 512

# Logging
os.makedirs("training_log", exist_ok=True)
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M")
LOG_FILE = f"training_log/{TIMESTAMP}_cvm_classifier_log.csv"

# --- Model Definition (EfficientNet with CORAL) ---
class CoralEfficientNet(nn.Module):
    def __init__(self, num_classes=6, pretrained=True):
        super(CoralEfficientNet, self).__init__()
        # Load Pretrained EfficientNet-B0
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT if pretrained else None)
        
        # Replace Classifier Head for CORAL
        # EfficientNet-B0 features: 1280
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # Remove original head
        
        # Binary classifiers for ordinal regression (K-1 output nodes)
        self.fc = nn.Linear(num_features, num_classes - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1)) # Shared bias trick often used, but separate bias is also fine.
        # CORAL paper suggests shared weights but separate bias is simpler for implementation.
        # Actually, standard CORAL implementation uses a single weight vector and K-1 biases.
        # But here we just use K-1 independent outputs for simplicity (similar to previous MIL code).
        # Let's stick to the previous successful implementation: K-1 outputs.
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features) + self.bias
        return logits

def task_importance_weights(label_tensor, num_classes=6):
    # Not used for now, stick to standard CORAL loss logic
    return torch.ones_like(label_tensor, dtype=torch.float32)

def coral_loss(logits, levels, importance_weights=None):
    # levels: (batch_size, num_classes-1) - already encoded
    # logits: (batch_size, num_classes-1)
    val = (-torch.sum((torch.nn.functional.logsigmoid(logits)*levels
                      + (torch.nn.functional.logsigmoid(logits) - logits)*(1-levels)), dim=1))
    return torch.mean(val)

def label_to_levels(label, num_classes):
    # label: 0 to 5
    # levels: [1, 1, 1, 0, 0] for label 3 (class 4)
    # shape: (batch_size, num_classes-1)
    levels = torch.zeros(label.size(0), num_classes - 1).to(DEVICE)
    for i in range(label.size(0)):
        if label[i] > 0:
            levels[i, :label[i]] = 1
    return levels

def proba_to_label(logits):
    # Sum sigmoid > 0.5
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels

# --- Training Loop ---

def train_model():
    print(f"Using device: {DEVICE}")
    
    # 1. Data Transforms
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # ImageNet stats
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
    
    dataloaders = {x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE, 
                                 shuffle=True if x == 'train' else False, 
                                 num_workers=NUM_WORKERS, pin_memory=True) 
                   for x in ['train', 'valid']}
    
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid']}
    class_names = image_datasets['train'].classes
    
    print(f"Dataset sizes: {dataset_sizes}")
    print(f"Classes: {class_names}")

    # 3. Model
    model = CoralEfficientNet(num_classes=NUM_CLASSES).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4) # AdamW for better regularization
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    # 4. Training
    best_model_wts = copy.deepcopy(model.state_dict())
    best_kappa = -1.0
    best_epoch = 0
    
    # Logging Header
    with open(LOG_FILE, 'w') as f:
        f.write("Epoch,Train_Loss,Valid_Loss,Train_Kappa,Valid_Kappa,Valid_Acc,Valid_MAE,LR\n")
        
    print("-" * 60)
    
    for epoch in range(EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            
            running_loss = 0.0
            all_preds = []
            all_labels = []
            
            # Tqdm bar
            pbar = tqdm(dataloaders[phase], desc=f"Epoch {epoch+1}/{EPOCHS} [{phase}]", leave=False)
            
            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE) # 0~5 class index
                
                # ImageFolder reads 1, 2, ... 6 folders as 0, 1, ... 5 labels automatically.
                # We need to make sure this mapping is correct.
                # If folder '1' -> index 0, then label 0 means CVM 1. correct.
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    logits = model(inputs)
                    levels = label_to_levels(labels, NUM_CLASSES)
                    loss = coral_loss(logits, levels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                # Stats
                running_loss += loss.item() * inputs.size(0)
                preds = proba_to_label(logits)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                pbar.set_postfix({'loss': loss.item()})
            
            epoch_loss = running_loss / dataset_sizes[phase]
            
            # Metrics
            if phase == 'train':
                train_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
                train_loss = epoch_loss
            else:
                valid_loss = epoch_loss
                valid_kappa = cohen_kappa_score(all_labels, all_preds, weights='quadratic')
                valid_acc = accuracy_score(all_labels, all_preds)
                valid_mae = mean_absolute_error(all_labels, all_preds)
                
                scheduler.step(valid_loss)
                
                # Save Best
                if valid_kappa > best_kappa:
                    best_kappa = valid_kappa
                    best_epoch = epoch
                    best_model_wts = copy.deepcopy(model.state_dict())
                    torch.save(model.state_dict(), f"checkpoints/best_cvm_classifier.pth")
                    print(f"  --> New Best Kappa: {best_kappa:.4f} (Saved)")

        # Log
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f} Kappa: {train_kappa:.4f} | Valid Loss: {valid_loss:.4f} Kappa: {valid_kappa:.4f} Acc: {valid_acc:.4f}")
        
        with open(LOG_FILE, 'a') as f:
            f.write(f"{epoch+1},{train_loss},{valid_loss},{train_kappa},{valid_kappa},{valid_acc},{valid_mae},{current_lr}\n")

    print(f"Training Complete. Best Validation Kappa: {best_kappa:.4f} at Epoch {best_epoch+1}")
    
if __name__ == '__main__':
    train_model()
