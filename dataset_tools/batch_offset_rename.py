"""Shift the numeric index of "aug_<NNNN>.png" files by a fixed offset
(useful when merging numbered batches of generated images without
clobbering filenames) and convert them to .jpg.
"""
import os

folder = "final-mask3"
offset = 1000

if __name__ == "__main__":
    for filename in os.listdir(folder):
        if filename.startswith("aug_") and filename.endswith(".png"):
            num_str = filename[4:8]  # aug_0000 -> "0000"
            new_num = int(num_str) + offset
            new_filename = f"aug_{new_num:04d}.jpg"

            old_path = os.path.join(folder, filename)
            new_path = os.path.join(folder, new_filename)
            os.rename(old_path, new_path)

    print("Renaming completed.")
