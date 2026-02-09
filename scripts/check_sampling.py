
import cv2
import os
import numpy as np
import json

def check_orientation():
    root = "Aariz/train/Cephalograms"
    images = sorted(os.listdir(root))
    if not images:
        print("No images found.")
        return
    
    img_path = os.path.join(root, images[0])
    img = cv2.imread(img_path)
    h, w, _ = img.shape
    print(f"Image Resolution: {w}x{h}")
    
    # Draw sampling area (lower 50%)
    cv2.rectangle(img, (0, h//2), (w, h), (0, 255, 0), 10)
    
    # Save to check
    cv2.imwrite("sampling_preview.jpg", img)
    print("Saved sampling_preview.jpg")

if __name__ == '__main__':
    check_orientation()
