
import os
import json
import shutil
import random
import glob
from sklearn.model_selection import train_test_split
import config

# --- Configuration ---
SOURCE_IMG_DIR = os.path.join(config.DATASET_PATH, 'train', 'Cephalograms')
SOURCE_LABEL_DIR = os.path.join(config.DATASET_PATH, 'roi_labels')

YOLO_dataset_DIR = os.path.join(config.DATASET_PATH, 'yolo_dataset')
os.makedirs(YOLO_dataset_DIR, exist_ok=True)

# YOLO structure
# images/train, images/val
# labels/train, labels/val
for split in ['train', 'val']:
    os.makedirs(os.path.join(YOLO_dataset_DIR, 'images', split), exist_ok=True)
    os.makedirs(os.path.join(YOLO_dataset_DIR, 'labels', split), exist_ok=True)

def convert_to_yolo_format(bbox, img_width, img_height):
    # bbox: [x, y, w, h] (top-left)
    # YOLO: [x_center, y_center, width, height] (normalized 0~1)
    
    x, y, w, h = bbox
    
    x_center = (x + w / 2) / img_width
    y_center = (y + h / 2) / img_height
    width = w / img_width
    height = h / img_height
    
    return x_center, y_center, width, height

def prepare_yolo_dataset():
    # 1. Get List of Labeled Files
    json_files = glob.glob(os.path.join(SOURCE_LABEL_DIR, "*.json"))
    labeled_filenames = []
    
    data_pairs = [] # (img_path, json_path)
    
    print(f"Found {len(json_files)} labels.")
    
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            
        filename = data['file_name']
        img_path = os.path.join(SOURCE_IMG_DIR, filename)
        
        if os.path.exists(img_path):
            data_pairs.append((img_path, json_file))
        else:
            print(f"Warning: Image not found for {json_file}")

    # 2. Split Train/Val
    train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.2, random_state=42)
    
    print(f"Split: Train {len(train_pairs)}, Val {len(val_pairs)}")
    
    # 3. Copy and Convert
    def process_split(pairs, split_name):
        for img_path, json_path in pairs:
            # Copy Image
            filename = os.path.basename(img_path)
            shutil.copy(img_path, os.path.join(YOLO_dataset_DIR, 'images', split_name, filename))
            
            # Create Label
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            x_c, y_c, w, h = convert_to_yolo_format(data['bbox'], data['image_width'], data['image_height'])
            
            # Write txt file (Class 0 for CVM_ROI)
            label_filename = os.path.splitext(filename)[0] + ".txt"
            label_path = os.path.join(YOLO_dataset_DIR, 'labels', split_name, label_filename)
            
            with open(label_path, 'w') as f:
                f.write(f"0 {x_c:.6f} {y_c:.6f} {w:.6f} {h:.6f}\n")
                
    process_split(train_pairs, 'train')
    process_split(val_pairs, 'val')
    
    # 4. Create data.yaml for YOLO
    yaml_content = f"""
path: {os.path.abspath(YOLO_dataset_DIR)}
train: images/train
val: images/val

nc: 1
names: ['cervical_vertebrae']
    """
    
    with open(os.path.join(YOLO_dataset_DIR, 'data.yaml'), 'w') as f:
        f.write(yaml_content)
        
    print(f"YOLO dataset prepared at {YOLO_dataset_DIR}")

if __name__ == '__main__':
    prepare_yolo_dataset()
