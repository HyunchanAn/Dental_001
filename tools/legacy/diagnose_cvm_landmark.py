
import torch
import numpy as np
import os
import json
import cv2
from torchvision import transforms, models
import torch.nn as nn

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
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    for i in range(min(5, len(img_list))):
        img_path = os.path.join(img_dir, img_list[i])
        orig_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb_img).unsqueeze(0)
        
        with torch.no_grad():
            lm_preds, cvm_logits = model(input_tensor)
            coords = lm_preds.view(1, 29, 2).numpy()[0] # Fix shape
            cvm_probs = torch.softmax(cvm_logits, dim=1).numpy()[0]
            cvm_pred = np.argmax(cvm_probs) + 1
        
        # Get GT CVM
        cvm_json = os.path.join('Aariz/test/Annotations/CVM Stages', img_list[i].split('.')[0] + '.json')
        gt_cvm = "N/A"
        if os.path.exists(cvm_json):
            with open(cvm_json, 'r') as f:
                gt_data = json.load(f)
            gt_cvm = gt_data['cvm_stage']['value']

        print(f"[{img_list[i]}] CVM Pred: {cvm_pred} (Prob: {cvm_probs.max():.2f}), GT: {gt_cvm}")
        print(f"   Landmark Stats: X_Mean={coords[:,0].mean():.1f}, Y_Mean={coords[:,1].mean():.1f}")
        print(f"   Landmark Spread: X_Std={coords[:,0].std():.1f}, Y_Std={coords[:,1].std():.1f}")
        print(f"   Sample L1: ({coords[0,0]:.1f}, {coords[0,1]:.1f})")

if __name__ == "__main__":
    diagnose()
