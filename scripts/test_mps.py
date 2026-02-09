
import torch
import time
from torch.utils.data import DataLoader
from torchvision import transforms
from dataset_mil import MILDataset
from model_mil import AttentionMIL
import config

def test_mps_speed():
    DEVICE = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Testing speed on: {DEVICE}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    dataset = MILDataset(mode='TRAIN', transform=transform, num_patches=16, patch_size=128)
    # Using a smaller batch for quick test
    loader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=2)

    model = AttentionMIL(num_classes=6).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss() # Simplified for speed test

    model.train()
    start_time = time.time()
    num_batches = 10
    
    print(f"Starting {num_batches} iterations...")
    for i, (patches, labels) in enumerate(loader):
        if i >= num_batches:
            break
        
        patches = patches.to(DEVICE)
        # Simplified label for ordinal loss testing
        dummy_labels = torch.zeros(patches.size(0), 5).to(DEVICE) 
        
        optimizer.zero_grad()
        logits = model(patches)
        loss = criterion(logits, dummy_labels)
        loss.backward()
        optimizer.step()
        
        if (i+1) % 2 == 0:
            print(f"Iteration {i+1}/{num_batches}")

    end_time = time.time()
    avg_time = (end_time - start_time) / num_batches
    print(f"Average time per batch (BS=8): {avg_time:.4f} seconds")
    print(f"Total time for {num_batches} batches: {end_time - start_time:.4f} seconds")

if __name__ == '__main__':
    test_mps_speed()
