"""PyTorch Dataset for (MRI slice, brain mask) pairs.

Expects a mask directory containing files named "<image_base>_mask_pred.png"
for each "<image_base>.jpg" in the image directory.
"""
import os

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class BrainSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_filenames):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.image_filenames = image_filenames

        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        image_name = self.image_filenames[idx]
        base = os.path.splitext(image_name)[0]
        mask_name = f"{base}_mask_pred.png"

        image_path = os.path.join(self.image_dir, image_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(image_path).convert("L")
        mask = Image.open(mask_path).convert("L")

        image = self.transform(image)
        mask = self.transform(mask)

        return image, mask
