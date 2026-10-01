# Dataset Tools

Small, standalone housekeeping scripts used while assembling the dataset.
Not part of the core pipeline — run as needed.

- **`remove_empty_pairs.py`** — deletes (image, mask) pairs whose mask is
  entirely blank (no plaque), and logs what was deleted.
- **`rename_masks.py`** — converts/renames a folder of mask images to the
  `A (<i>)_mask_pred.png` naming convention the other scripts expect.
- **`batch_offset_rename.py`** — shifts the numeric index of `aug_<NNNN>.png`
  files by a fixed offset, for merging batches of generated images without
  filename collisions.
- **`crop_to_brain_bbox.py`** — crops each MRI slice to the bounding box of
  its predicted brain mask.

Edit the path constants near the top of each file before running.
