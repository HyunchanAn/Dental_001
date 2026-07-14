import os
import sys
import torch
import cv2
import numpy as np
import json
import argparse
import albumentations as A

# Add src to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from src.landmark.model import UNetHeatmapModel
from src.config import ANATOMICAL_LANDMARKS, NUM_LANDMARKS
from src.cephalometric_analyzer import CephalometricAnalyzer

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Get the ordered list of symbols based on the config.py dictionary order
# (Dictionary insertion order is preserved in Python 3.7+)
LANDMARK_SYMBOLS = [v["symbol"] for v in ANATOMICAL_LANDMARKS.values()]

def get_coords_from_heatmaps(heatmaps, image_size, heatmap_size):
    """Convert a batch of heatmaps to landmark coordinates in original image scale."""
    batch_size, num_landmarks, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_landmarks, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    y_coords = max_indices // w
    x_coords = max_indices % w
    
    coords = torch.stack([x_coords, y_coords], dim=2).float()
    
    # Scale coordinates back to the original image dimensions
    scale_x = image_size[1] / w
    scale_y = image_size[0] / h
    coords[:, :, 0] *= scale_x
    coords[:, :, 1] *= scale_y
    
    return coords

def main():
    parser = argparse.ArgumentParser(description="Run Cephalometric Analysis on an image")
    parser.add_argument("--image", type=str, required=True, help="Path to lateral cephalogram image")
    parser.add_argument("--model", type=str, default="models/model_heatmap_resnet50_finetuned_mre4.5.pth", help="Path to model weights")
    args = parser.parse_args()

    image_path = args.image
    model_path = os.path.join(BASE_DIR, args.model)
    
    if not os.path.exists(image_path):
        print(f"Error: Image {image_path} not found.")
        return

    # Load image to get original dimensions
    original_img = cv2.imread(image_path)
    if original_img is None:
        print("Error: Could not read image.")
        return
        
    orig_h, orig_w = original_img.shape[:2]
    
    # Preprocessing
    IMAGE_SIZE_PRED = (256, 256)
    HEATMAP_OUTPUT_SIZE_PRED = (64, 64)
    
    transform = A.Compose([
        A.Resize(height=IMAGE_SIZE_PRED[0], width=IMAGE_SIZE_PRED[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ])
    
    img_rgb = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    transformed = transform(image=img_rgb)
    img_tensor = torch.tensor(transformed['image']).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
    
    # Load Model
    print("Loading model...")
    model = UNetHeatmapModel(num_landmarks=NUM_LANDMARKS).to(DEVICE)
    
    if os.path.exists(model_path):
        pretrained_dict = torch.load(model_path, map_location=DEVICE)
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.shape == model_dict[k].shape}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
    else:
        print(f"Warning: Model not found at {model_path}. Proceeding with random weights for testing.")
        
    model.eval()
    
    # Predict
    print("Predicting landmarks...")
    with torch.no_grad():
        heatmaps_pred = model(img_tensor)
        
    # Get coordinates mapped to the RESIZED image size first
    coords_pred = get_coords_from_heatmaps(heatmaps_pred, IMAGE_SIZE_PRED, HEATMAP_OUTPUT_SIZE_PRED).cpu().numpy()[0]
    
    # Scale coordinates back to the ORIGINAL image size so they can be plotted correctly
    scale_x = orig_w / IMAGE_SIZE_PRED[1]
    scale_y = orig_h / IMAGE_SIZE_PRED[0]
    
    landmarks_dict = {}
    for i, symbol in enumerate(LANDMARK_SYMBOLS):
        x = coords_pred[i, 0] * scale_x
        y = coords_pred[i, 1] * scale_y
        landmarks_dict[symbol] = (x, y)
        
    print(f"Mapped {len(landmarks_dict)} landmarks successfully.")
    
    # Run Cephalometric Analyzer
    print("Running Cephalometric Analysis...")
    analyzer = CephalometricAnalyzer(landmarks_dict)
    analysis_results = analyzer.analyze()
    
    print("\n" + "="*40)
    print(" CEPHALOMETRIC ANALYSIS REPORT ")
    print("="*40)
    print(json.dumps(analysis_results, indent=2, ensure_ascii=False))
    print("="*40)

if __name__ == "__main__":
    main()
