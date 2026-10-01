"""Batch-run the stage-1 threshold mask (see initial_threshold_mask.py) over a
numbered sequence of files named "A (<i>).jpg", e.g. an exported slice range.
"""
import os

import cv2

from initial_threshold_mask import make_initial_mask

input_dir = "Input"
output_dir = "masks"
start_index = 1
end_index = 159  # exclusive

if __name__ == "__main__":
    os.makedirs(output_dir, exist_ok=True)

    for i in range(start_index, end_index):
        filename = os.path.join(input_dir, f"A ({i}).jpg")
        output_path = os.path.join(output_dir, f"A_{i}_mask.png")

        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if image is None:
            print(f"Could not load {filename}. Skipping.")
            continue

        brain_mask = make_initial_mask(image)
        if brain_mask.max() == 0:
            print(f"No brain region found in {filename}")
            continue

        cv2.imwrite(output_path, brain_mask)
        print(f"Saved: {output_path}")
