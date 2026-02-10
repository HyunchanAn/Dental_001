
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
    
    img_path = os.path.join(img_dir, img_list[0])
    orig_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h_orig, w_orig = orig_img.shape[:2]

    # Get GT
    json_path = os.path.join('Aariz/test/Annotations/Cephalometric Landmarks/Senior Orthodontists', img_list[0].split('.')[0] + '.json')
    with open(json_path, 'r') as f:
        gt_data = json.load(f)
    gt_sella = gt_data['landmarks'][0]['value']
    gt_rel_x = gt_sella['x'] / w_orig
    gt_rel_y = gt_sella['y'] / h_orig

    print(f"Target Relative Position (Sella): x={gt_rel_x:.4f}, y={gt_rel_y:.4f}")

    configs = [
        {"size": 256, "norm": "ImageNet"},
        {"size": 256, "norm": "0.5"},
        {"size": 512, "norm": "ImageNet"},
        {"size": 512, "norm": "0.5"},
    ]

    for cfg in configs:
        res = cfg["size"]
        if cfg["norm"] == "ImageNet":
            norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        else:
            norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        
        transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((res, res)),
            transforms.ToTensor(),
            norm
        ])
        
        input_tensor = transform(rgb_img).unsqueeze(0)
        with torch.no_grad():
            lm_preds, _ = model(input_tensor)
            coords = lm_preds.view(29, 2).numpy()[0]
        
        pred_x, pred_y = coords[0], coords[1]
        # Test common scales
        for scale in [256, 512, 1024]:
            rel_x, rel_y = pred_x / scale, pred_y / scale
            err = np.sqrt((rel_x - gt_rel_x)**2 + (rel_y - gt_rel_y)**2)
            if err < 0.1: # Significant match
                print(f"MATCH! Size={res}, Norm={cfg['norm']}, Scale={scale} => Rel: ({rel_x:.3f}, {rel_y:.3f}), Err: {err:.4f}")
            else:
                pass
        
        print(f"Size={res}, Norm={cfg['norm']} => Raw Pred: ({pred_x:.1f}, {pred_y:.1f})")

if __name__ == "__main__":
    diagnose()
