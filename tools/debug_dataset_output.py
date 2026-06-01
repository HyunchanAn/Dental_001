
import os
import sys
import torch
import cv2
import numpy as np

# Add project root and src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.landmark.dataset import HeatmapDataset
from src.config import DATASET_PATH, IMAGE_SIZE

def debug_dataset():
    # Heatmap size corresponds to model output
    heatmap_size = (256, 256)
    dataset = HeatmapDataset(DATASET_PATH, mode='test', image_size=IMAGE_SIZE, output_size=heatmap_size, sigma=3)
    
    # Get first sample
    image, heatmaps, landmarks = dataset[0]
    
    # Convert image tensor back to numpy (chw -> hwc)
    img_np = image.permute(1, 2, 0).numpy()
    # Unnormalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = (img_np * std + mean) * 255
    img_np = img_np.astype(np.uint8)
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    
    # Draw landmarks (red dots) - These are in IMAGE_SIZE scale (512, 512)
    for i, (x, y) in enumerate(landmarks):
        cv2.circle(img_bgr, (int(x), int(y)), 3, (0, 0, 255), -1)
    
    # Create heatmap overlay (cyan glow)
    # Sum all heatmaps for visualization
    combined_hm = torch.sum(heatmaps, dim=0).numpy()
    combined_hm = np.clip(combined_hm * 255, 0, 255).astype(np.uint8)
    # Resize heatmap to fits image size
    combined_hm_resized = cv2.resize(combined_hm, (IMAGE_SIZE[1], IMAGE_SIZE[0]))
    heatmap_color = cv2.applyColorMap(combined_hm_resized, cv2.COLORMAP_JET)
    
    # Blend image and heatmap
    overlay = cv2.addWeighted(img_bgr, 0.7, heatmap_color, 0.3, 0)
    
    out_path = 'debug_dataset_output.jpg'
    cv2.imwrite(out_path, overlay)
    print(f"Debug image saved to {out_path}")
    print(f"Sample image shape: {img_np.shape}")
    print(f"Heatmap shape: {heatmaps.shape}")
    print(f"First 5 landmarks (512 scale): \n{landmarks[:5]}")

if __name__ == "__main__":
    debug_dataset()
