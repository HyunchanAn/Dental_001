
import torch
import os

path = 'checkpoints/best_model.pth'
if os.path.exists(path):
    checkpoint = torch.load(path, map_location='cpu')
    print(f"Total keys: {len(checkpoint.keys())}")
    for k in sorted(checkpoint.keys()):
        print(k)
else:
    print(f"File not found: {path}")
