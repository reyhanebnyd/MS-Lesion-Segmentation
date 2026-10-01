"""Delete (image, mask) pairs whose mask is entirely empty (no plaque)."""
import glob
import os

import cv2
import numpy as np

train_image_dir = "Train-Input"
train_mask_dir = "Train-Mask"
log_path = "deleted_files_log.txt"


def is_mask_empty(mask_path):
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load mask {mask_path}")
        return False
    return np.all(mask == 0)


def delete_pair(image_path, mask_path, log_list):
    try:
        if os.path.exists(image_path) and os.path.exists(mask_path):
            os.remove(image_path)
            os.remove(mask_path)
            log_list.append((os.path.basename(image_path), os.path.basename(mask_path)))
            print(f"Deleted: {image_path} and {mask_path}")
        else:
            print(f"Warning: File not found - Image: {image_path}, Mask: {mask_path}")
    except Exception as e:
        print(f"Error deleting {image_path} or {mask_path}: {e}")


def process_dataset(image_dir, mask_dir, dataset_name):
    deleted_files = []

    image_files = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.png")))

    min_len = min(len(image_files), len(mask_files))
    image_files = image_files[:min_len]
    mask_files = mask_files[:min_len]

    print(f"Processing {dataset_name}: {min_len} paired files found")

    for img_path, mask_path in zip(image_files, mask_files):
        if is_mask_empty(mask_path):
            delete_pair(img_path, mask_path, deleted_files)

    return deleted_files


if __name__ == "__main__":
    print("Starting deletion of empty mask/image pairs...")
    train_deleted = process_dataset(train_image_dir, train_mask_dir, "Training set")

    total_deleted = len(train_deleted)
    print(f"\nSummary: Deleted {total_deleted} pairs")

    with open(log_path, "w") as f:
        f.write(f"Deleted {total_deleted} pairs\n")
        if total_deleted > 0:
            f.write("\nDeleted files (image, mask):\n")
            for img_name, mask_name in train_deleted:
                f.write(f"{img_name}, {mask_name}\n")

    print(f"Log saved to {log_path}")
