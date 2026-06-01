
import os
import cv2
import torch
import numpy as np
import glob
import random
import json
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms, models
from ultralytics import YOLO
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src import config
from src.cvm.inverse_filter import apply_equipotential_filter
from src.landmark.model import UNetHeatmapModel

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

# --- Model Definitions ---
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

def proba_to_label(logits):
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels.item()

# --- Inference Tool ---
class CVMInference:
    def __init__(self):
        print("Initializing CVM Inference Pipeline...")
        self.detector = YOLO(YOLO_PATH)
        self.classifier = CoralEfficientNet(num_classes=NUM_CLASSES).to(DEVICE)
        self.classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
        self.classifier.eval()
        
        # Load landmark model for fallback
        self.landmark_model = UNetHeatmapModel(num_landmarks=config.NUM_LANDMARKS).to(DEVICE)
        landmark_weights = 'checkpoints/best_model.pth'
        if os.path.exists(landmark_weights):
            self.landmark_model.load_state_dict(torch.load(landmark_weights, map_location=DEVICE))
        self.landmark_model.eval()
        print("Models loaded successfully.")

    def get_gt_label(self, img_path):
        """Try to find the true label if it exists in the Aariz dataset structure."""
        # This is strictly for evaluating our own test set.
        filename = os.path.basename(img_path)
        # Search in all splits (train/test/valid) for the label file
        for split in ['train', 'test', 'valid']:
            lbl_path = os.path.join(config.DATASET_PATH, split, 'Annotations', 'CVM Stages', os.path.splitext(filename)[0] + ".json")
            if os.path.exists(lbl_path):
                with open(lbl_path, 'r') as f:
                    data = json.load(f)
                    return data['cvm_stage']['value']
        return None

    def predict(self, img_path):
        # 1. Load Image
        orig_img = cv2.imread(img_path)
        if orig_img is None: return None
        rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        
        # Apply equipotential filter (inverse problem approach)
        filtered_img = apply_equipotential_filter(rgb_img)
        
        # 2. Detect ROI (YOLO)
        results = self.detector.predict(filtered_img, conf=0.5, verbose=False)
        
        prediction = {
            'img_path': img_path,
            'roi_detected': False,
            'stage_pred': None,
            'bbox': None,
            'true_label': self.get_gt_label(img_path)
        }

        if len(results[0].boxes) > 0:
            box = results[0].boxes[0]
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        else:
            # Fallback ROI based on Landmarks topology
            input_tensor = cv2.resize(filtered_img, (512, 512))
            input_tensor = transforms.ToTensor()(input_tensor).unsqueeze(0).to(DEVICE)
            with torch.no_grad():
                heatmaps = self.landmark_model(input_tensor)
            
            # Get average location of all landmarks to find the face center
            heatmaps_np = heatmaps.squeeze(0).cpu().numpy()
            y_coords, x_coords = [], []
            for hm in heatmaps_np:
                idx = np.argmax(hm)
                y_coords.append(idx // 512)
                x_coords.append(idx % 512)
            
            face_center_x = np.mean(x_coords) / 512.0
            face_center_y = np.mean(y_coords) / 512.0
            
            h, w = rgb_img.shape[:2]
            # Inverse map: cervical vertebrae are typically right and down from face center
            x1 = int(w * (face_center_x + 0.1))
            y1 = int(h * (face_center_y + 0.1))
            x2 = int(w * (face_center_x + 0.4))
            y2 = int(h * (face_center_y + 0.5))
            
            # Ensure within bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            
        # Common expansion for context
        h, w = rgb_img.shape[:2]
        margin = 0.1
        bw, bh = x2 - x1, y2 - y1
        x1 = max(0, x1 - bw * margin)
        y1 = max(0, y1 - bh * margin)
        x2 = min(w, x2 + bw * margin)
        y2 = min(h, y2 + bh * margin)
        
        prediction['roi_detected'] = True
        prediction['bbox'] = (int(x1), int(y1), int(x2), int(y2))
        
        # 3. Classify (EfficientNet)
        crop = rgb_img[int(y1):int(y2), int(x1):int(x2)]
        pil_crop = Image.fromarray(crop)
        tensor_crop = transform(pil_crop).unsqueeze(0).to(DEVICE)
        
        with torch.no_grad():
            logits = self.classifier(tensor_crop)
            
            # Uncertainty Filter
            probas = torch.sigmoid(logits)
            variance = torch.var(probas, dim=1).item()
            if variance < 0.05: # threshold for uncertainty
                prediction['stage_pred'] = "Uncertain"
            else:
                stage_idx = proba_to_label(logits)
                prediction['stage_pred'] = stage_idx + 1 # 0~5 -> 1~6

        return prediction

    def visualize_results(self, num_samples=10):
        # Pick random test images
        test_images = glob.glob(os.path.join(config.DATASET_PATH, 'test', 'Cephalograms', '*.*'))
        if not test_images:
            print("No test images found.")
            return

        samples = random.sample(test_images, min(len(test_images), num_samples))
        
        plt.figure(figsize=(20, 10))
        for i, img_path in enumerate(samples):
            pred = self.predict(img_path)
            
            orig_img = cv2.imread(img_path)
            orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
            
            plt.subplot(2, 5, i+1)
            
            if pred['roi_detected']:
                x1, y1, x2, y2 = pred['bbox']
                cv2.rectangle(orig_img, (x1, y1), (x2, y2), (0, 255, 0), 15)
                
                title = f"Pred: {pred['stage_pred']} | True: {pred['true_label']}"
                color = 'green' if pred['stage_pred'] == pred['true_label'] else 'red'
                plt.title(title, color=color, fontsize=12, fontweight='bold')
            else:
                plt.title("ROI Not Detected", color='gray')
            
            plt.imshow(orig_img)
            plt.axis('off')
            
        plt.tight_layout()
        output_path = 'docs/assets/inference_results.png'
        plt.savefig(output_path)
        print(f"Results saved to {output_path}")
        return output_path

if __name__ == '__main__':
    engine = CVMInference()
    res_path = engine.visualize_results(10)
    # Automatically open if on Mac
    os.system(f"open {res_path}")
