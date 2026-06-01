
import torch
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from dataset_mil import MILDataset
from model_mil import AttentionMIL
import config

# --- Setup ---
DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
CHECKPOINT_PATH = 'checkpoints/best_mil_model.pth' # The best performing model
NUM_CLASSES = 6
PATCH_SIZE = 224
NUM_PATCHES = 16
OUTPUT_DIR = 'attention_analysis'

def get_prediction(logits):
    # CORAL: sum sigmoid > 0.5
    return torch.sum(torch.sigmoid(logits) > 0.5, dim=1).item()

def visualize_attention(model, dataset, num_samples=5):
    model.eval()
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # We need to access individual items to visualize patches
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # 1. Get data
            patches_tensor, label_tensor = dataset[idx]
            
            # 2. Forward pass with attention
            patches_input = patches_tensor.unsqueeze(0).to(DEVICE) # (1, N, C, H, W)
            logits, attention_weights = model(patches_input, return_attention=True)
            
            pred_stage = get_prediction(logits)
            true_stage = label_tensor.item()
            
            # attention_weights shape: (1, num_patches, 1) -> (num_patches,)
            attn = attention_weights.squeeze().cpu().numpy()
            
            # 3. Visualization
            fig, axes = plt.subplots(4, 4, figsize=(12, 12))
            fig.suptitle(f"Sample {idx} | True: {true_stage+1} | Pred: {pred_stage+1}", fontsize=16)
            
            patches_np = patches_tensor.permute(0, 2, 3, 1).numpy()
            patches_np = (patches_np * 0.5) + 0.5 # Un-normalize
            patches_np = np.clip(patches_np, 0, 1)
            
            # Find max attention for highlighting
            max_attn_idx = np.argmax(attn)
            
            for p_idx, ax in enumerate(axes.flat):
                if p_idx < len(patches_np):
                    ax.imshow(patches_np[p_idx])
                    
                    # Highlight border color based on attention weight
                    # Red = Low attention, Green = High attention
                    score = attn[p_idx]
                    color = (1 - score/attn.max(), score/attn.max(), 0) # Simple heatmap color
                    
                    rect_width = 10
                    if p_idx == max_attn_idx:
                        # Draw a thick red border for the most attended patch
                        rect = plt.Rectangle((0,0), PATCH_SIZE, PATCH_SIZE, fill=False, color='red', linewidth=5)
                        ax.add_patch(rect)
                        title_color = 'red'
                        weight_text = "MAX"
                    else:
                        title_color = 'black'
                        weight_text = ""
                    
                    ax.set_title(f"w: {score:.4f} {weight_text}", color=title_color, fontsize=10)
                    ax.axis('off')
            
            plt.tight_layout()
            save_path = os.path.join(OUTPUT_DIR, f"attention_sample_{i}_true{true_stage+1}_pred{pred_stage+1}.png")
            plt.savefig(save_path)
            print(f"Saved attention visualization to {save_path}")
            plt.close()

if __name__ == '__main__':
    print(f"Loading model from {CHECKPOINT_PATH}...")
    
    # Load Model
    model = AttentionMIL(num_classes=NUM_CLASSES).to(DEVICE)
    try:
        model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=DEVICE))
        print("Model loaded successfully.")
    except FileNotFoundError:
        print("Checkpoint not found. Please ensure training has finished and saved the best model.")
        exit()

    # Load Dataset (VALID mode to check generalization)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    val_dataset = MILDataset(mode='VALID', transform=transform, num_patches=NUM_PATCHES, patch_size=PATCH_SIZE)
    
    print("Generating visualizations...")
    visualize_attention(model, val_dataset, num_samples=10)
    print("Done.")
