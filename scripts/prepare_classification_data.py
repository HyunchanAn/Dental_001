
import os
import shutil
import cv2
import glob
import json
from ultralytics import YOLO
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src import config

# --- Configuration ---
SOURCE_DS_PATH = config.DATASET_PATH # 'Aariz'
OUTPUT_DS_PATH = 'Aariz_CVM_Clean'
YOLO_MODEL_PATH = 'yolo_runs/cvm_detector/weights/best.pt'
IMG_SIZE = 512 # Final classification input size
CONF_THRESHOLD = 0.5 # Confidence threshold

# Load YOLO model
print(f"Loading YOLO model from {YOLO_MODEL_PATH}...")
model = YOLO(YOLO_MODEL_PATH)

def get_cvm_label(json_path):
    """
    Parses the ground truth annotation file (JSON) to get the CVM stage (1-6).
    """
    if not os.path.exists(json_path):
        return None
        
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        # Structure: {"cvm_stage": {"value": 5, ...}}
        if 'cvm_stage' in data and 'value' in data['cvm_stage']:
            return int(data['cvm_stage']['value'])
            
    except Exception as e:
        print(f"Error parsing {json_path}: {e}")
        return None
    
    return None

def process_dataset():
    if os.path.exists(OUTPUT_DS_PATH):
        shutil.rmtree(OUTPUT_DS_PATH)
    
    total_processed = 0
    total_saved = 0
    
    for split in ['train', 'test', 'valid']:
        print(f"Processing {split} set...")
        
        # --- Corrected Path Logic ---
        # Annotations are in 'Annotations/CVM Stages' subdirectory!
        if split == 'train':
            img_dir = os.path.join(SOURCE_DS_PATH, 'train', 'Cephalograms')
            lbl_dir = os.path.join(SOURCE_DS_PATH, 'train', 'Annotations', 'CVM Stages')
        elif split == 'test':
            img_dir = os.path.join(SOURCE_DS_PATH, 'test', 'Cephalograms')
            lbl_dir = os.path.join(SOURCE_DS_PATH, 'test', 'Annotations', 'CVM Stages')
        elif split == 'valid':
            img_dir = os.path.join(SOURCE_DS_PATH, 'valid', 'Cephalograms')
            lbl_dir = os.path.join(SOURCE_DS_PATH, 'valid', 'Annotations', 'CVM Stages')
            
        print(f"  Looking in: {img_dir}")
        print(f"  Looking for labels in: {lbl_dir}")
        
        # Create output dirs
        for i in range(1, 7):
            os.makedirs(os.path.join(OUTPUT_DS_PATH, split, str(i)), exist_ok=True)
            
        # Image extensions
        img_paths = glob.glob(os.path.join(img_dir, "*.bmp")) + \
                    glob.glob(os.path.join(img_dir, "*.jpg")) + \
                    glob.glob(os.path.join(img_dir, "*.png"))
        
        print(f"  Found {len(img_paths)} images.")
        
        for img_path in img_paths:
            total_processed += 1
            filename = os.path.basename(img_path)
            # Label file usually matches image filename but with .json extension
            json_filename = os.path.splitext(filename)[0] + ".json"
            json_path = os.path.join(lbl_dir, json_filename)
            
            # 1. Get GT Label
            label = get_cvm_label(json_path)
            if label is None:
                # print(f"Skipping {filename}: Label not found in {json_path}.")
                continue
                
            # 2. Run YOLO Detection
            results = model.predict(img_path, conf=CONF_THRESHOLD, verbose=False)
            
            if len(results[0].boxes) > 0:
                # Get best box
                box = results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Expand box slightly (context)
                h, w = results[0].orig_shape
                margin = 0.1 # 10% margin
                box_h = y2 - y1
                box_w = x2 - x1
                
                x1 = max(0, x1 - box_w * margin)
                y1 = max(0, y1 - box_h * margin)
                x2 = min(w, x2 + box_w * margin)
                y2 = min(h, y2 + box_h * margin)
                
                # Crop
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                crop = img[int(y1):int(y2), int(x1):int(x2)]
                
                if crop.size == 0:
                    continue
                    
                # Resize
                crop_resized = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
                
                # Save to class folder
                # Convert label to string for folder name
                save_path = os.path.join(OUTPUT_DS_PATH, split, str(label), filename)
                # Convert specific formats to jpg for consistency
                save_path = os.path.splitext(save_path)[0] + ".jpg"
                
                cv2.imwrite(save_path, crop_resized)
                total_saved += 1
            else:
                # print(f"Skipping {filename}: ROI not detected.")
                pass

    print(f"Dataset preparation complete.")
    print(f"Processed: {total_processed}")
    print(f"Saved (ROI detected & Label found): {total_saved}")
    print(f"Saved to {OUTPUT_DS_PATH}")

if __name__ == '__main__':
    process_dataset()
