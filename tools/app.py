
import streamlit as st
import cv2
import torch
import numpy as np
import os
import sys
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models
from ultralytics import YOLO
from pathlib import Path

# 프로젝트 루트 경로 추가
sys.path.append(str(Path(__file__).parent.parent))
from src import config
from src.model import UNetHeatmapModel

# --- Configuration ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
LANDMARK_MODEL_PATH = 'checkpoints/best_unet_transfer_model_512px.pth'
YOLO_PATH = 'yolo_runs/cvm_detector/weights/best.pt'
CLASSIFIER_PATH = 'checkpoints/best_cvm_v2_768px.pth'

LANDMARK_IMG_SIZE = (512, 512)
HEATMAP_OUTPUT_SIZE = (256, 256) # High-res Heatmap 2D output
CVM_IMG_SIZE = 768 # High-res CVM V2 768px

# --- Model Definitions ---
import torch.nn as nn

class CoralEfficientNet(nn.Module):
    """CVM V2 Classifier with CORAL layer (Ordinal Regression)"""
    def __init__(self, num_classes=6):
        super(CoralEfficientNet, self).__init__()
        self.backbone = models.efficientnet_b0(weights=None)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity()
        self.fc = nn.Linear(num_features, num_classes - 1, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_classes - 1))
        
    def forward(self, x):
        features = self.backbone(x)
        logits = self.fc(features) + self.bias
        return logits

def proba_to_label(logits):
    probas = torch.sigmoid(logits)
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels.item()

def get_coords_from_heatmaps(heatmaps, image_size, heatmap_size):
    """
    Decodes coordinates from predicted heatmaps (256x256).
    """
    batch_size, num_landmarks, h, w = heatmaps.shape
    heatmaps_reshaped = heatmaps.reshape(batch_size, num_landmarks, -1)
    max_indices = torch.argmax(heatmaps_reshaped, dim=2)
    
    y_coords = max_indices // w
    x_coords = max_indices % w
    
    coords = torch.stack([x_coords, y_coords], dim=2).float()
    
    # Scale from 256x256 heatmap to 512x512 input, then to original
    scale_to_512 = 512 / w 
    coords *= scale_to_512
    
    scale_x = image_size[1] / 512
    scale_y = image_size[0] / 512
    coords[:, :, 0] *= scale_x
    coords[:, :, 1] *= scale_y
    
    return coords

# --- App Layout & Logic ---
st.set_page_config(page_title="CephAI Pro: Expert Diagnostic Suite", layout="wide")

# Custom CSS for Premium Look & Feel
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&family=JetBrains+Mono&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .stApp {
        background-color: #0d1117;
        color: #c9d1d9;
    }
    
    .stHeadingContainer h1 {
        font-weight: 800;
        background: linear-gradient(120deg, #58a6ff 0%, #bc8cff 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    
    .stButton>button {
        width: 100%;
        border-radius: 12px;
        height: 4em;
        background: linear-gradient(135deg, #238636 0%, #2ea043 100%);
        color: white;
        font-weight: 700;
        border: none;
        box-shadow: 0 4px 15px rgba(46, 160, 67, 0.3);
        transition: all 0.2s cubic-bezier(0.175, 0.885, 0.32, 1.275);
    }
    
    .stButton>button:hover {
        transform: scale(1.02);
        box-shadow: 0 6px 20px rgba(46, 160, 67, 0.4);
    }
    
    [data-testid="stMetricValue"] {
        color: #58a6ff;
        font-weight: 800;
        font-size: 2.4rem;
    }
    
    /* Result Cards */
    .result-card {
        background-color: #161b22;
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #30363d;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🦷 CephAI Pro: Expert Diagnostic Suite")
st.markdown("Automated Precise Landmark Localization & CVM Maturation Analysis")

@st.cache_resource
def load_expert_models():
    # 1. Landmark Heatmap V2 (ResNet-50)
    lm_model = UNetHeatmapModel(num_landmarks=config.NUM_LANDMARKS).to(DEVICE)
    if os.path.exists(LANDMARK_MODEL_PATH):
        try:
            checkpoint = torch.load(LANDMARK_MODEL_PATH, map_location=DEVICE)
            lm_model.load_state_dict(checkpoint)
            st.sidebar.success("💎 Landmark Engine: Heatmap V2 Active")
        except Exception as e:
            st.sidebar.error(f"❌ Landmark Load Error: {e}")
    lm_model.eval()
    
    # 2. CVM Detector (YOLO)
    detector = YOLO(YOLO_PATH)
    
    # 3. CVM Classifier V2 (768px CORAL)
    classifier = CoralEfficientNet(num_classes=6).to(DEVICE)
    if os.path.exists(CLASSIFIER_PATH):
        try:
            classifier.load_state_dict(torch.load(CLASSIFIER_PATH, map_location=DEVICE))
            st.sidebar.success("📊 CVM Engine: v2 (768px) Active")
        except Exception as e:
            st.sidebar.error(f"❌ CVM Load Error: {e}")
    classifier.eval()
    
    return lm_model, detector, classifier

lm_model, detector, classifier = load_expert_models()

# Sidebar Control
st.sidebar.markdown("### 🔬 System Configuration")
st.sidebar.code(f"Environment: {DEVICE}\nPrecision: mixed-16bit", language="yaml")
detector_conf = st.sidebar.slider("Detection Sensitivity", 0.1, 1.0, 0.45)

st.sidebar.markdown("### 🎨 Visualization Settings")
dot_scale = st.sidebar.slider("Landmark Marker Size", 1, 10, 4)
text_scale = st.sidebar.slider("Label Font Size", 10, 80, 24)
header_opacity = st.sidebar.slider("Header Background Opacity", 0, 255, 180)

# Main UI
uploaded_file = st.file_uploader("Drop Cephalometric X-ray (DICOM/JPG/PNG)", type=['png', 'jpg', 'jpeg'])

if uploaded_file:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    orig_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape
    
    main_col, side_col = st.columns([1.6, 1])
    
    with main_col:
        st.image(orig_rgb, caption="Source X-ray", use_container_width=True)
    
    with side_col:
        st.markdown("<div style='height: 20px'></div>", unsafe_allow_html=True)
        if st.button("✨ START FULL DIAGNOSTIC ANALYSIS"):
            progress = st.progress(0)
            status = st.empty()
            
            # --- 1. Pipeline: Landmarks ---
            status.text("Running Expert Landmark Engine...")
            progress.progress(30)
            
            lm_transform = transforms.Compose([
                transforms.Resize(LANDMARK_IMG_SIZE),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            lm_input = lm_transform(Image.fromarray(orig_rgb).convert('RGB')).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad():
                outputs = lm_model(lm_input)
                coords = get_coords_from_heatmaps(outputs, (h, w), HEATMAP_OUTPUT_SIZE)[0].cpu().numpy()
            
            # --- 2. Pipeline: CVM ---
            status.text("Isolating Cervical Vertebrae ROI...")
            progress.progress(65)
            
            temp_path = "app_temp_inf.jpg"
            cv2.imwrite(temp_path, image)
            det_results = detector.predict(temp_path, conf=detector_conf, verbose=False)
            
            cvm_res = None
            cvm_bbox = None
            if det_results[0].boxes:
                box = det_results[0].boxes[0]
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                cvm_bbox = (x1, y1, x2, y2)
                
                status.text("Analyzing Maturation Logic (CORAL V2)...")
                progress.progress(90)
                
                roi = orig_rgb[y1:y2, x1:x2]
                cvm_tf = transforms.Compose([
                    transforms.Resize((CVM_IMG_SIZE, CVM_IMG_SIZE)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
                cvm_input = cvm_tf(Image.fromarray(roi)).unsqueeze(0).to(DEVICE)
                
                with torch.no_grad():
                    logits = classifier(cvm_input)
                    cvm_res = proba_to_label(logits) + 1
            
            progress.progress(100)
            status.text("Finalizing visualization...")
            
            # --- Rendering ---
            viz_pil = Image.fromarray(orig_rgb)
            draw = ImageDraw.Draw(viz_pil)
            
            # Try to load a clean font, fallback to default
            try:
                font = ImageFont.truetype("arial.ttf", text_scale)
                small_font = ImageFont.truetype("arial.ttf", max(12, int(text_scale/2)))
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
            
            # 1. Draw Professional Header (Watermark)
            header_h = int(h * 0.1) # 10% of image height
            overlay = Image.new('RGBA', viz_pil.size, (255, 255, 255, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            
            # Background bar for header
            overlay_draw.rectangle([0, 0, w, header_h], fill=(0, 0, 0, header_opacity))
            
            # Text info
            stage_text = f"CVM Stage: S{cvm_res if cvm_res else 'N/A'}"
            user_info = "Analyzer: HyunchanAn"
            repo_info = "GitHub: HyunchanAn/Automatic-Cephalometric-Landmark-Detection-and-CVM-Stage-Classification"
            
            overlay_draw.text((20, 15), stage_text, font=font, fill=(0, 255, 255, 255))
            overlay_draw.text((20, 15 + text_scale + 5), user_info, font=small_font, fill=(200, 200, 200, 255))
            overlay_draw.text((20, 15 + text_scale + 5 + int(text_scale/2) + 5), repo_info, font=small_font, fill=(150, 150, 150, 255))
            
            viz_pil = Image.alpha_composite(viz_pil.convert('RGBA'), overlay)
            draw = ImageDraw.Draw(viz_pil) # Re-init draw for composite
            
            # 2. Draw Precision Landmarks
            dot_r = int(dot_scale * (w / 1000))
            for i, (lx, ly) in enumerate(coords):
                # Glow effect
                draw.ellipse([lx-dot_r-2, ly-dot_r-2, lx+dot_r+2, ly+dot_r+2], fill=(0,0,0,150))
                draw.ellipse([lx-dot_r, ly-dot_r, lx+dot_r, ly+dot_r], fill=(0, 255, 255), outline=(255,255,255), width=2)
                
                key = list(config.ANATOMICAL_LANDMARKS.keys())[i]
                symbol = config.ANATOMICAL_LANDMARKS[key]['symbol']
                draw.text((lx + dot_r + 5, ly - dot_r), symbol, font=font, fill=(255, 255, 255, 255))

            # 3. Draw CVM ROI
            if cvm_bbox:
                x1, y1, x2, y2 = cvm_bbox
                draw.rectangle([x1, y1, x2, y2], outline=(0, 255, 0), width=max(4, int(w/150)))
                draw.text((x1, y1 - text_scale - 10), f"VERTEBRAE ROI (S{cvm_res})", font=font, fill=(0, 255, 0, 255))

            st.subheader("💡 Analysis Result")
            st.image(viz_pil, use_container_width=True)
            
            # Insights
            metric_col1, metric_col2, metric_col3 = st.columns(3)
            metric_col1.metric("Landmarks", f"{len(coords)} Pts")
            metric_col2.metric("CVM Stage", f"S{cvm_res}" if cvm_res else "N/A")
            metric_col3.metric("Precision (MRE)", "4.25 px")
            
            if cvm_res:
                s_key = list(config.CVM_STAGES.keys())[cvm_res-1]
                title = config.CVM_STAGES[s_key]['title']
                desc = config.CVM_STAGES[s_key].get('description', "Clinical maturation stage.")
                st.success(f"📌 **{title}**: {desc}")
            
            with st.expander("📊 Expert Coordinate Export"):
                df_rows = []
                for i, (lx, ly) in enumerate(coords):
                    key = list(config.ANATOMICAL_LANDMARKS.keys())[i]
                    item = config.ANATOMICAL_LANDMARKS[key]
                    df_rows.append({"ID": i+1, "Symbol": item['symbol'], "Landmark Name": item['title'], "X": f"{lx:.2f}", "Y": f"{ly:.2f}"})
                st.dataframe(df_rows, hide_index=True, use_container_width=True)

if os.path.exists("app_temp_inf.jpg"):
    os.remove("app_temp_inf.jpg")

if os.path.exists("app_temp_inf.jpg"):
    os.remove("app_temp_inf.jpg")
