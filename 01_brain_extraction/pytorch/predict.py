"""Run a trained brain-extraction U-Net over a folder of MRI slices and save predicted masks."""
import os

import cv2
import numpy as np
import torch

from model import UNet

weights_path = "unet_brain.pth"
image_dir = "Train-Input"
output_dir = "final"

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet()
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if not fname.endswith(".jpg"):
            continue
        image = cv2.imread(os.path.join(image_dir, fname), cv2.IMREAD_GRAYSCALE)
        resized = cv2.resize(image, (256, 256)) / 255.0
        input_tensor = torch.tensor(resized).unsqueeze(0).unsqueeze(0).float()

        with torch.no_grad():
            pred = model(input_tensor)
            pred_mask = torch.sigmoid(pred).squeeze().numpy()
            pred_mask = (pred_mask > 0.5).astype(np.uint8) * 255

        mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]))
        save_path = os.path.join(output_dir, fname.replace(".jpg", "_mask_pred.png"))
        cv2.imwrite(save_path, mask_resized)
