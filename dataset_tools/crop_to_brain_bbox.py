"""Crop each MRI slice to the bounding box of its predicted brain mask."""
import os

import cv2

image_dir = "Images1"
mask_dir = "output_masks1"
output_dir = "cropped_brains"

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    for fname in os.listdir(image_dir):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(image_dir, fname)
        mask_path = os.path.join(mask_dir, fname.replace(".jpg", "_mask_pred.png"))

        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if image is None or mask is None:
            print(f"Skipping {fname} due to missing image or mask.")
            continue

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            print(f"No brain region found in {fname}, skipping.")
            continue

        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped = image[y:y + h, x:x + w]

        output_path = os.path.join(output_dir, fname.replace(".jpg", "_cropped.png"))
        cv2.imwrite(output_path, cropped)
