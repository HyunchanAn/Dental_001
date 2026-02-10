
from ultralytics import YOLO
import os
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

from src import config

# --- Configuration ---
DATA_YAML = os.path.join(config.DATASET_PATH, 'yolo_dataset', 'data.yaml')
MODEL_NAME = 'yolov8n.pt'  # Nano model (smallest, fastest)
EPOCHS = 50
IMG_SIZE = 640
BATCH_SIZE = 16
DEVICE = 'mps' # Apple Silicon

def train_yolo():
    # Load model
    print(f"Loading {MODEL_NAME}...")
    model = YOLO(MODEL_NAME) 

    # Train
    print(f"Starting training on {DEVICE}...")
    results = model.train(
        data=DATA_YAML,
        epochs=EPOCHS,
        imgsz=IMG_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        project='yolo_runs',
        name='cvm_detector',
        exist_ok=True, # Overwrite existing run
        plots=True
    )
    
    print("Training complete.")
    print(f"Best model saved at: yolo_runs/cvm_detector/weights/best.pt")

if __name__ == '__main__':
    train_yolo()
