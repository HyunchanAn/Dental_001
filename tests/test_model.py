import torch
from src.model import UNetHeatmapModel, HeatmapModel
from src.config import NUM_LANDMARKS

def test_unet_heatmap_model_shape():
    # Instantiate with pretrained=False to avoid downloading ResNet-50 weights in CI
    model = UNetHeatmapModel(num_landmarks=NUM_LANDMARKS, pretrained=False)
    model.eval()

    # Create dummy tensor of shape (batch_size, channels, height, width)
    # The models are designed for 512x512 images
    dummy_input = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        output = model(dummy_input)

    # The output heatmap is expected to have shape (batch_size, num_landmarks, output_h, output_w)
    # UNetHeatmapModel extra upsampling block (up_conv1) gives 1/2 size of input (256x256)
    expected_shape = (2, NUM_LANDMARKS, 256, 256)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"

def test_heatmap_model_shape():
    # Instantiate with pretrained=False
    model = HeatmapModel(num_landmarks=NUM_LANDMARKS, pretrained=False)
    model.eval()

    dummy_input = torch.randn(2, 3, 512, 512)

    with torch.no_grad():
        output = model(dummy_input)

    # HeatmapModel downsamples to 1/32 (16x16) and has 3 transpose conv blocks (upsampling 2x each)
    # 16 -> 32 -> 64 -> 128
    # So the output size is 128x128
    expected_shape = (2, NUM_LANDMARKS, 128, 128)
    assert output.shape == expected_shape, f"Expected shape {expected_shape}, but got {output.shape}"
