
import os
import cv2
import torch
import numpy as np
import json
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
from ultralytics import YOLO
import config

# --- Configuration ---
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
YOLO_PATH = 'yolo_runs/cvm_detector/weights/best.pt'
CLASSIFIER_PATH = 'checkpoints/best_cvm_classifier.pth'
NUM_CLASSES = 6
IMG_SIZE = 512

# Transformation for Classifier
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Model Definition (Simplified for inference) ---
import torch.nn as nn
class CoralEfficientNet(nn.Module):
    def __init__(self, num_classes=6):
        super(CoralEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features) + self.bias
        return logits

def get_prediction(img_path, detector, classifier):
    # Detect
    results = detector.predict(img_path, conf=0.5, verbose=False)
    if len(results[0].boxes) == 0: return None
    
    box = results[0].boxes[0]
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    
    # Crop & Classify
    orig_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Expand slightly for context
    h, w = results[0].orig_shape
    margin = 0.1
    bw, bh = x2 - x1, y2 - y1
    ex1, ey1 = max(0, x1 - bw * margin), max(0, y1 - bh * margin)
    ex2, ey2 = min(w, x2 + bw * margin), min(h, y2 + bh * margin)
    
    crop = rgb_img[int(ey1):int(ey2), int(ex1):int(ex2)]
    pil_crop = Image.fromarray(crop)
    tensor_crop = transform(pil_crop).unsqueeze(0).to(DEVICE)
    
    with torch.no_grad():
        logits = classifier(tensor_crop)
        probas = torch.sigmoid(logits)
        predict_levels = probas > 0.5
        stage_idx = torch.sum(predict_levels, dim=1).item()
        confidence = probas[0, int(stage_idx)-1].item() if stage_idx > 0 else probas[0, 0].item()

    return {
        'bbox': (int(x1), int(y1), int(x2), int(y2)),
        'stage': int(stage_idx) + 1,
        'conf': confidence
    }

def create_pretty_viz():
    print("Creating premium visualization...")
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).parent.parent.parent))
    from src.download_weights import ensure_model_exists
    try:
        ensure_model_exists(os.path.basename(CLASSIFIER_PATH), os.path.dirname(CLASSIFIER_PATH))
    except Exception as e:
        print(f"Failed to check/download model: {e}")
        
    detector = YOLO(YOLO_PATH)
    classifier = CoralEfficientNet(num_classes=NUM_CLASSES).to(DEVICE)
    classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
    classifier.eval()

    # Select a good test image
    test_dir = os.path.join(config.DATASET_PATH, 'test', 'Cephalograms')
    img_list = os.listdir(test_dir)
    # Pick a random one for now, or a specific one if known
    img_name = img_list[10] # Just an example
    img_path = os.path.join(test_dir, img_name)
    
    # Get ground truth
    gt_label = "N/A"
    lbl_path = os.path.join(config.DATASET_PATH, 'test', 'Annotations', 'CVM Stages', os.path.splitext(img_name)[0] + ".json")
    if os.path.exists(lbl_path):
        with open(lbl_path, 'r') as f:
            data = json.load(f)
            gt_label = data['cvm_stage']['value']

    pred = get_prediction(img_path, detector, classifier)
    if not pred:
        print("Failed to detect ROI in sample image.")
        return

    # Load image with PIL for better drawing
    img = Image.open(img_path).convert("RGB")
    draw = ImageDraw.Draw(img, "RGBA")
    
    # 1. Draw ROI Box with soft glow/alpha
    x1, y1, x2, y2 = pred['bbox']
    # Outer glow/border
    for i in range(5):
        draw.rectangle([x1-i, y1-i, x2+i, y2+i], outline=(0, 255, 127, 50 - i*10), width=2)
    draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 127, 200), width=8)

    # 2. Add Overlay Panel at the top
    # Use generic fonts if custom not found
    try:
        font_large = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial Bold.ttf", 80)
        font_small = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 50)
    except:
        font_large = ImageFont.load_default()
        font_small = ImageFont.load_default()

    # Draw semi-transparent background for text
    draw.rectangle([0, 0, 800, 300], fill=(0, 0, 0, 150))
    
    # Text Contents
    draw.text((40, 40), f"CVM Stage Analysis", fill=(0, 255, 127), font=font_large)
    draw.text((40, 140), f"PREDICTED: Stage {pred['stage']} (Conf: {pred['conf']:.2%})", fill=(255, 255, 255), font=font_small)
    draw.text((40, 210), f"GROUND TRUTH: Stage {gt_label}", fill=(200, 200, 200), font=font_small)

    # 3. Label near the box
    draw.rectangle([x1, y1-70, x1+250, y1], fill=(0, 255, 127, 200))
    draw.text((x1+10, y1-65), f"CVM-S{pred['stage']}", fill=(0, 0, 0), font=font_small)

    save_path = "docs/assets/cvm_premium_visualization.png"
    img.save(save_path)
    print(f"Visualization saved to {save_path}")
    os.system(f"open {save_path}")

if __name__ == '__main__':
    create_pretty_viz()
