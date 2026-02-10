
import torch
import numpy as np
import os
import json
import cv2
from torchvision import transforms, models
import torch.nn as nn

# Model Definition (ResNet-18 version)
class AdvancedCephNet(nn.Module):
    def __init__(self, num_landmarks=29, num_classes=6):
        super(AdvancedCephNet, self).__init__()
        self.resnet = models.resnet18(weights=None)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Identity()
        self.landmark_head = nn.Linear(num_features, num_landmarks * 2)
        self.cvm_head = nn.Linear(num_features, num_classes)
    def forward(self, x):
        features = self.resnet(x)
        return self.landmark_head(features), self.cvm_head(features)

def diagnose():
    path = 'checkpoints/best_model.pth'
    device = torch.device('cpu')
    model = AdvancedCephNet(num_landmarks=29)
    model.load_state_dict(torch.load(path, map_location=device), strict=False)
    model.eval()

    img_dir = 'Aariz/test/Cephalograms'
    img_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if not img_list: return
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    for i in range(min(5, len(img_list))):
        img_path = os.path.join(img_dir, img_list[i])
        orig_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_img).unsqueeze(0)
        
        with torch.no_grad():
            lm_preds, _ = model(input_tensor)
            coords = lm_preds.view(29, 2).numpy()[0]
        
        # Get GT from JSON
        json_path = os.path.join('Aariz/test/Annotations/Cephalometric Landmarks/Senior Orthodontists', img_list[i].split('.')[0] + '.json')
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                gt_data = json.load(f)
            gt_sella = gt_data['landmarks'][0]['value']
            
            print(f"\n--- Image {i+1}: {img_list[i]} ---")
            print(f"GT Sella Rel: x={gt_sella['x']/orig_img.shape[1]:.4f}, y={gt_sella['y']/orig_img.shape[0]:.4f}")
            print(f"Pred Sella Rel (to 1024): x={coords[0]/1024:.4f}, y={coords[1]/1024:.4f}")
            print(f"Pred Sella Rel (to 1024, flipped X): x={1 - coords[0]/1024:.4f}")

if __name__ == "__main__":
    diagnose()
