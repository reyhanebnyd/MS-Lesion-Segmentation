"""Convert a folder of "...(<i>)..." mask images to PNG, renumbered and
renamed to the "A (<i+1>)_mask_pred.png" convention expected by the
preprocessing/training scripts.
"""
import os
import re

from PIL import Image

source_dir = "./new-aida-mask"
dest_dir = "./correct-mask"
index_offset = 1  # added to the number extracted from each source filename

if __name__ == "__main__":
    os.makedirs(dest_dir, exist_ok=True)

    for filename in os.listdir(source_dir):
        match = re.search(r"\((\d+)\)", filename)
        if not match:
            continue
        try:
            i_part = int(match.group(1)) + index_offset
            new_filename = f"A ({i_part})_mask_pred.png"

            img_path = os.path.join(source_dir, filename)
            img = Image.open(img_path)
            new_img_path = os.path.join(dest_dir, new_filename)
            img.save(new_img_path, "PNG")

            print(f"Converted and renamed: {filename} -> {new_filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")
