import json
import cv2
import numpy as np
import torch
from src.dataset import HeatmapDataset
from src.config import NUM_LANDMARKS

def test_heatmap_dataset_loading(tmp_path):
    # Setup the required folder structure in tmp_path
    mode = "train"
    train_dir = tmp_path / mode
    ceph_dir = train_dir / "Cephalograms"
    annotations_dir = train_dir / "Annotations" / "Cephalometric Landmarks"
    senior_dir = annotations_dir / "Senior Orthodontists"
    junior_dir = annotations_dir / "Junior Orthodontists"

    # Create directories
    ceph_dir.mkdir(parents=True, exist_ok=True)
    senior_dir.mkdir(parents=True, exist_ok=True)
    junior_dir.mkdir(parents=True, exist_ok=True)

    # Create a dummy image (e.g. 800x600 gray/color image)
    img_name = "test_image.png"
    dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
    cv2.imwrite(str(ceph_dir / img_name), dummy_image)

    # Create dummy landmarks (29 points)
    # The coordinate values must be within the image bounds
    landmarks_data = []
    for i in range(NUM_LANDMARKS):
        # Place landmarks at varying positions
        landmarks_data.append({
            "value": {
                "x": float(50 + i * 10),
                "y": float(100 + i * 15)
            }
        })

    annotation_content = {"landmarks": landmarks_data}

    # Write senior and junior annotation JSON files
    json_name = "test_image.json"
    with open(senior_dir / json_name, "w") as f:
        json.dump(annotation_content, f)
    with open(junior_dir / json_name, "w") as f:
        json.dump(annotation_content, f)

    # Instantiate HeatmapDataset
    image_size = (512, 512)
    output_size = (64, 64)
    dataset = HeatmapDataset(
        dataset_folder_path=str(tmp_path),
        mode="TRAIN",
        image_size=image_size,
        output_size=output_size,
        sigma=2
    )

    # Assert dataset length is correct
    assert len(dataset) == 1

    # Load the item
    image, heatmaps, landmarks = dataset[0]

    # Check return types
    assert isinstance(image, torch.Tensor)
    assert isinstance(heatmaps, torch.Tensor)
    assert isinstance(landmarks, torch.Tensor)

    # Check shapes
    # Albumentations Compose with ToTensorV2 converts HWC image to CHW tensor: (3, 512, 512)
    assert image.shape == (3, 512, 512)
    # Heatmaps should be (NUM_LANDMARKS, output_height, output_width)
    assert heatmaps.shape == (NUM_LANDMARKS, 64, 64)
    # Landmarks should be (NUM_LANDMARKS, 2)
    assert landmarks.shape == (NUM_LANDMARKS, 2)
