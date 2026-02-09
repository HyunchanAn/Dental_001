
import cv2
import os
import json
import glob
import numpy as np
import config

# --- Configuration ---
# Aariz 데이터셋 구조 분석 결과: Aariz/train/Cephalograms
IMAGE_DIR = os.path.join(config.DATASET_PATH, 'train', 'Cephalograms') 
LABEL_DIR = os.path.join(config.DATASET_PATH, 'roi_labels')
WINDOW_NAME = 'CVM ROI Labeling Tool'
DISPLAY_HEIGHT = 800  # 맥북 화면 높이를 고려하여 디스플레이용 높이 제한

class LabelingTool:
    def __init__(self):
        # jpg, png, bmp 모두 지원
        exts = ['*.jpg', '*.png', '*.bmp', '*.jpeg']
        self.image_paths = []
        for ext in exts:
            self.image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
        
        self.image_paths = sorted(self.image_paths)
        
        if not self.image_paths:
            print(f"Error: No images found in {IMAGE_DIR}")
            print(f"Current Directory: {os.getcwd()}")
            # 디버깅: 상위 폴더 구조 출력
            if os.path.exists(config.DATASET_PATH):
                print(f"Contents of {config.DATASET_PATH}: {os.listdir(config.DATASET_PATH)}")
            exit()

        os.makedirs(LABEL_DIR, exist_ok=True)
        
        self.current_idx = 0
        self.total_images = len(self.image_paths)
        
        # Mouse state
        self.drawing = False
        self.ix, self.iy = -1, -1
        self.bbox = None # (x, y, w, h) in original scale
        
        # Display scaling
        self.scale_factor = 1.0
        
        # Current image
        self.original_image = None
        self.display_image = None
        
        # UI
        cv2.namedWindow(WINDOW_NAME)
        cv2.setMouseCallback(WINDOW_NAME, self.mouse_callback)
        
        print(f"Loaded {self.total_images} images.")
        print("Controls:")
        print("  [Left Click + Drag] : Draw Bounding Box (C2-C4)")
        print("  [D] or [Right]      : Next Image (Auto Save)")
        print("  [A] or [Left]       : Previous Image")
        print("  [S]                 : Save Manually")
        print("  [R]                 : Reset Box")
        print("  [ESC]               : Quit")

    def load_current_image(self):
        img_path = self.image_paths[self.current_idx]
        self.original_image = cv2.imread(img_path)
        
        if self.original_image is None:
            print(f"Failed to load {img_path}")
            return

        # Calculate scale factor to fit screen
        h, w = self.original_image.shape[:2]
        self.scale_factor = DISPLAY_HEIGHT / float(h)
        new_w, new_h = int(w * self.scale_factor), int(h * self.scale_factor)
        
        # Resize for display
        self.display_image = cv2.resize(self.original_image, (new_w, new_h))
        
        # Load existing label if present
        self.load_label()

    def load_label(self):
        img_path = self.image_paths[self.current_idx]
        filename = os.path.basename(img_path)
        json_path = os.path.join(LABEL_DIR, f"{filename}.json")
        
        self.bbox = None
        if os.path.exists(json_path):
            try:
                with open(json_path, 'r') as f:
                    data = json.load(f)
                    self.bbox = tuple(data['bbox']) # [x, y, w, h]
            except Exception as e:
                print(f"Error loading label: {e}")

    def save_label(self):
        if self.bbox is None:
            return # Don't save empty if nothing drawn

        img_path = self.image_paths[self.current_idx]
        filename = os.path.basename(img_path)
        json_path = os.path.join(LABEL_DIR, f"{filename}.json")
        
        data = {
            "file_name": filename,
            "bbox": list(self.bbox), # [x, y, w, h]
            "image_width": self.original_image.shape[1],
            "image_height": self.original_image.shape[0]
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=4)
        print(f"Saved label for {filename}")

    def mouse_callback(self, event, x, y, flags, param):
        if self.original_image is None:
            return

        # Map display coordinates back to original coordinates
        orig_x = int(x / self.scale_factor)
        orig_y = int(y / self.scale_factor)
        
        h, w = self.original_image.shape[:2]
        orig_x = max(0, min(w, orig_x))
        orig_y = max(0, min(h, orig_y))

        if event == cv2.EVENT_LBUTTONDOWN:
            self.drawing = True
            self.ix, self.iy = orig_x, orig_y
            self.bbox = None # Reset when starting new
            
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.drawing:
                cur_x, cur_y = orig_x, orig_y
                # Temporary bbox for visualization
                x_min = min(self.ix, cur_x)
                y_min = min(self.iy, cur_y)
                w_box = abs(self.ix - cur_x)
                h_box = abs(self.iy - cur_y)
                self.bbox = (x_min, y_min, w_box, h_box)

        elif event == cv2.EVENT_LBUTTONUP:
            self.drawing = False
            cur_x, cur_y = orig_x, orig_y
            x_min = min(self.ix, cur_x)
            y_min = min(self.iy, cur_y)
            w_box = abs(self.ix - cur_x)
            h_box = abs(self.iy - cur_y)
            
            # Minimum size filter (e.g., 10px) to avoid accidental clicks
            if w_box > 10 and h_box > 10:
                self.bbox = (x_min, y_min, w_box, h_box)
                self.save_label() # Auto save on release
            else:
                self.bbox = None

    def draw_ui(self):
        if self.display_image is None:
            return
            
        view_img = self.display_image.copy()
        
        # Draw bbox if exists
        if self.bbox:
            x, y, w, h = self.bbox
            # Scale to display
            dx = int(x * self.scale_factor)
            dy = int(y * self.scale_factor)
            dw = int(w * self.scale_factor)
            dh = int(h * self.scale_factor)
            
            cv2.rectangle(view_img, (dx, dy), (dx+dw, dy+dh), (0, 255, 0), 2)
            cv2.putText(view_img, "C2-C4 ROI", (dx, dy-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # UI Text
        status_text = f"Image: {self.current_idx + 1}/{self.total_images}"
        filename = os.path.basename(self.image_paths[self.current_idx])
        
        cv2.putText(view_img, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(view_img, filename, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Guide
        guide_text = "Draw box around Cervical Vertebrae (C2-C4)"
        cv2.putText(view_img, guide_text, (10, view_img.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        cv2.imshow(WINDOW_NAME, view_img)

    def run(self):
        self.load_current_image()
        
        while True:
            self.draw_ui()
            
            # Wait for key
            key = cv2.waitKey(20) & 0xFF
            
            if key == 27: # ESC
                break
            elif key == 2 or key == 81: # Left Arrow (Mac often sends 2 or 81 depending on backend)
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.load_current_image()
            elif key == 3 or key == 83: # Right Arrow (Mac often sends 3 or 83)
                if self.current_idx < self.total_images - 1:
                    self.current_idx += 1
                    self.load_current_image()
            elif key == ord('s'):
                self.save_label()
                print("Manual Save triggered.")
            elif key == ord('r'):
                self.bbox = None
                print("Box reset.")
            
            # Mac specific check: arrow keys might map differently depending on OpenCV build
            # Usually 0=Up, 1=Down, 2=Left, 3=Right for waitKeyEx or waitKey depending on system
            # To be safe, let's allow 'a'/'d' for navigation too
            if key == ord('a'): # Prev
                if self.current_idx > 0:
                    self.current_idx -= 1
                    self.load_current_image()
            elif key == ord('d'): # Next
                if self.current_idx < self.total_images - 1:
                    self.current_idx += 1
                    self.load_current_image()
                    
        cv2.destroyAllWindows()

if __name__ == '__main__':
    tool = LabelingTool()
    tool.run()
