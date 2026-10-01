"""Train the brain-extraction U-Net (PyTorch).

Expects:
  <image_dir>/*.jpg                         - MRI slices
  <mask_dir>/<image_base>_mask_pred.png      - matching brain masks
"""
import os

import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from dataset import BrainSegmentationDataset
from model import UNet, dice_score

image_dir = "Train-Input"
mask_dir = "final_train_input"
num_epochs = 20
output_weights_path = "unet_brain.pth"

if __name__ == "__main__":
    all_images = [f for f in os.listdir(image_dir) if f.endswith(".jpg")]

    # Keep only images that have a corresponding mask
    image_mask_pairs = []
    for img in all_images:
        base = os.path.splitext(img)[0]
        mask_name = f"{base}_mask_pred.png"
        if os.path.exists(os.path.join(mask_dir, mask_name)):
            image_mask_pairs.append(img)

    train_files, test_files = train_test_split(image_mask_pairs, test_size=0.2, random_state=42)

    train_dataset = BrainSegmentationDataset(image_dir, mask_dir, train_files)
    test_dataset = BrainSegmentationDataset(image_dir, mask_dir, test_files)

    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=4, shuffle=False)

    model = UNet()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for images, masks in train_loader:
            preds = model(images)
            loss = criterion(preds, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        print(f"Epoch {epoch + 1}, Training Loss: {train_loss:.4f}")

    model.eval()
    total_dice = 0
    with torch.no_grad():
        for images, masks in test_loader:
            preds = model(images)
            total_dice += dice_score(preds, masks)

    avg_dice = total_dice / len(test_loader)
    print(f"Validation Dice Score: {avg_dice:.4f}")

    torch.save(model.state_dict(), output_weights_path)
