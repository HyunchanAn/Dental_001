
import os
import sys
import torch
import cv2
import numpy as np
from torchvision import transforms

# Add project root and src to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.landmark.model import UNetHeatmapModel
from src.config import NUM_LANDMARKS

def get_coords_from_heatmaps(heatmaps, image_size, heatmap_size):
    batch_size, num_landmarks, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_landmarks, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    y_coords = max_indices // w
    x_coords = max_indices % w
    coords = torch.stack([x_coords, y_coords], dim=2).float()
    scale_x = image_size[1] / w
    scale_y = image_size[0] / h
    coords[:, :, 0] *= scale_x
    coords[:, :, 1] *= scale_y
    return coords

def visualize():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = 'checkpoints/best_unet_transfer_model_512px.pth'
    
    model = UNetHeatmapModel(num_landmarks=NUM_LANDMARKS).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    img_dir = 'Aariz/test/Cephalograms'
    img_list = sorted([f for f in os.listdir(img_dir) if f.endswith('.png')])
    if not img_list:
        print("No images found.")
        return
    
    img_path = os.path.join(img_dir, img_list[0])
    orig_img = cv2.imread(img_path)
    rgb_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    input_tensor = transform(rgb_img).unsqueeze(0).to(device)
    
    with torch.no_grad():
        heatmaps_pred = model(input_tensor)
        coords_pred = get_coords_from_heatmaps(heatmaps_pred, (512, 512), (256, 256)).cpu().numpy()[0]

    # Visualization
    disp_img = cv2.resize(orig_img, (512, 512))
    for i in range(NUM_LANDMARKS):
        x, y = int(coords_pred[i, 0]), int(coords_pred[i, 1])
        cv2.circle(disp_img, (x, y), 3, (0, 0, 255), -1)
        # Put text for index
        cv2.putText(disp_img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

    save_path = 'interim_result.jpg'
    cv2.imwrite(save_path, disp_img)
    print(f"Result saved to {save_path}")

if __name__ == "__main__":
    visualize()
