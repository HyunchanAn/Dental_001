import io
from unittest.mock import MagicMock
from fastapi.testclient import TestClient
import numpy as np
import cv2
import torch

# Add project root path to sys.path to allow importing tools
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))

# Import tools.api for mocking
import tools.api

def test_api_health():
    """Verifies that the API health check endpoint returns 200 and expected status."""
    client = TestClient(tools.api.app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_api_predict():
    """Runs a simulated prediction on the FastAPI endpoint.

    Mocks the Landmark model, YOLO detector, and CVM classifier to test the entire
    FastAPI controller, pre-processing, cropping, and response serialization logic.
    """
    client = TestClient(tools.api.app)

    # 1. Mock Landmark model output: (1, 29, 256, 256)
    mock_lm_model = MagicMock()
    dummy_heatmap = torch.zeros(1, 29, 256, 256)
    for i in range(29):
        # Peak coordinates simulated on diagonal to guarantee deterministic scaling coords
        dummy_heatmap[0, i, min(i * 5, 255), min(i * 5, 255)] = 1.0
    mock_lm_model.return_value = dummy_heatmap

    # 2. Mock YOLO Detector: xyxy coordinates for crop
    mock_detector = MagicMock()
    mock_box = MagicMock()
    mock_box.xyxy = torch.tensor([[10.0, 20.0, 100.0, 200.0]])
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]
    mock_detector.predict.return_value = [mock_result]

    # 3. Mock CVM EfficientNet Classifier: logits shape (1, 5)
    mock_classifier = MagicMock()
    # 2 sigmoid thresholds are > 0.5 (logit > 0), so proba_to_label + 1 -> 3
    mock_classifier.return_value = torch.tensor([[2.0, 1.5, -1.0, -2.0, -3.0]])

    # Inject mocks globally into tools.api module
    tools.api.lm_model = mock_lm_model
    tools.api.detector = mock_detector
    tools.api.classifier = mock_classifier

    # Create a dummy image (e.g. 600x800 black image)
    dummy_image = np.zeros((600, 800, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', dummy_image)
    img_bytes = img_encoded.tobytes()

    # Call the /predict endpoint
    response = client.post(
        "/predict",
        files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    )

    # Assertions
    assert response.status_code == 200, f"Expected 200, but got {response.status_code}: {response.text}"

    res_data = response.json()
    assert "landmarks" in res_data
    assert len(res_data["landmarks"]) == 29

    # Check shape of landmark coordinates
    first_landmark = res_data["landmarks"][0]
    assert "symbol" in first_landmark
    assert "name" in first_landmark
    assert "x" in first_landmark
    assert "y" in first_landmark

    # Check CVM predictions
    assert "cvm" in res_data
    assert res_data["cvm"]["stage"] == 3
    assert res_data["cvm"]["title"] == "CVM-S3"
    assert res_data["cvm"]["bbox"] == [10, 20, 100, 200]

    # Check latency and compliance keys
    assert "latency_ms" in res_data
    assert res_data["latency_ms"] > 0
    assert res_data["compliance"]["metadata_stripped"] is True
    assert res_data["compliance"]["zero_storage_active"] is True
