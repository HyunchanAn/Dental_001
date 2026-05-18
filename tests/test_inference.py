import torch
import numpy as np
import sys
from pathlib import Path

# Add project root path to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from src.model import UNetHeatmapModel
from tools.api import CoralEfficientNet, proba_to_label, get_coords_from_heatmaps
from src.config import NUM_LANDMARKS

def set_deterministic_seeds(seed=42):
    """Sets identical seeds across PyTorch and NumPy for deterministic runs."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    # Ensure determinism in PyTorch operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def test_landmark_model_reproducibility():
    """Verifies that the UNet Landmark Heatmap model produces identical outputs
    given the same dummy input tensor, ensuring deterministic neural network forward pass.
    """
    set_deterministic_seeds(42)

    # Initialize landmark model with pretrained=False
    model = UNetHeatmapModel(num_landmarks=NUM_LANDMARKS, pretrained=False)
    model.eval()

    # Generate dummy input (batch_size=1, channels=3, height=512, width=512)
    dummy_input = torch.randn(1, 3, 512, 512)

    # First inference run
    with torch.no_grad():
        output_run_1 = model(dummy_input)

    # Second inference run with same inputs
    with torch.no_grad():
        output_run_2 = model(dummy_input)

    # Verify outputs are exactly identical down to precision limits
    assert torch.allclose(output_run_1, output_run_2, atol=1e-6), (
        "Landmark model forward pass is non-deterministic. Outputs differ across identical runs."
    )

    # Validate landmark coordinate decoding determinism
    coords_1 = get_coords_from_heatmaps(output_run_1, (600, 800), (256, 256))
    coords_2 = get_coords_from_heatmaps(output_run_2, (600, 800), (256, 256))

    assert torch.allclose(coords_1, coords_2, atol=1e-4), (
        "Heatmap coordinate decoding algorithm is non-deterministic."
    )

def test_cvm_classifier_reproducibility():
    """Verifies that the Coral Classifier model produces identical logits and
    final ordinal stage classifications given the same inputs.
    """
    set_deterministic_seeds(100)

    # Initialize classifier
    classifier = CoralEfficientNet(num_classes=6)
    classifier.eval()

    # CVM images are sized to 768px in the revised configuration
    dummy_input = torch.randn(1, 3, 768, 768)

    with torch.no_grad():
        logits_run_1 = classifier(dummy_input)

    with torch.no_grad():
        logits_run_2 = classifier(dummy_input)

    # Assert logits match
    assert torch.allclose(logits_run_1, logits_run_2, atol=1e-6), (
        "CVM Coral classifier logits differ across identical runs."
    )

    # Validate label decoding logic determinism
    stage_run_1 = proba_to_label(logits_run_1) + 1
    stage_run_2 = proba_to_label(logits_run_2) + 1

    assert stage_run_1 == stage_run_2, (
        "CVM label decoding logic output differs despite identical logits."
    )
