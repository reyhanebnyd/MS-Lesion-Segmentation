# Stage 4 — Segmentation Model Training

This stage searches for and trains the model that separates MS plaques
from brain tissue (report section 9, "Train کردن مدل"). Three different
attempts exist in the working files; here's how they relate, based on
cross-checking each notebook's architecture/output against the figures
actually embedded in `../docs/MSProject.pdf`.

## `keras_architecture_search/` — the model search described in the report

- **`deeplabv3plus_resnet50_attempt.ipynb`** (from `model1.ipynb`) —
  DeepLabV3+ with a ResNet50 encoder + ASPP module. The report (section
  9.2) says DeepLabV3+-ResNet50 and PSPNet were tried first and ran into
  overfitting / high computational cost — this notebook's function is
  literally named `DeepLabV3Plus_resnet50`, an exact match. Abandoned.

- **`unet_lightweight_final.ipynb`** (from `model2.ipynb`) — a smaller
  custom U-Net (352×352×3 input, 32→256 channels, with dropout), trained
  with Keras `ImageDataGenerator` augmentation and a class-weighted
  focal+Dice loss. **This is almost certainly the exact model behind the
  report's results**: its `model.summary()` output
  (`../docs/assets/final_unet_architecture_part{1,2}.png`) and its
  training curves (`../docs/assets/final_unet_training_*_curve.png`) match
  the figures embedded in the report pixel-for-pixel in shape, and the
  30-epoch curves line up with `../results/metrics_log.csv` /
  `unet_metrics_log.csv` (same file saved under two names) almost exactly
  — e.g. `val_dice_score` ends at 0.7824 at epoch 30, matching the report's
  Dice-score figure. Its second cell loads the saved `best_unet_model.h5`
  and evaluates it on the test set (this is where the report's test-set
  table, reproduced in the root README, comes from). That second cell has
  messy duplicated imports/definitions — left as-is since it wasn't
  possible to execute/verify a rewrite here.

## Separate experiment: baseline vs. augmented comparison

- **`train_with_augmentation.ipynb`** (from `model_augment.ipynb`) — a
  **different, deeper** U-Net (256×256×1 input, 64→1024 channels — the
  same architecture as `../01_brain_extraction/pytorch/model.py`, as a
  Keras model instead). It trains this architecture twice in one run: once
  on the original labeled data only, once on original+synthetic data
  (`final_generated_input` / `final_generated_mask`, 1500 images total),
  then explicitly prints `Improvement: combined_val_dice - baseline_val_dice`.
  This is the notebook that actually demonstrates the headline claim of
  the report (synthetic data improves plaque detection) — but note its
  architecture does **not** match the one shown in the report's own
  figures, so it reads as a separate validation experiment rather than the
  source of the report's published numbers.

- **`train_baseline.ipynb`** (from `train_unet.ipynb`) — an earlier,
  **unfinished draft** of the same idea: it fully trains and reports a
  baseline-only model, but the "Combined Training" section at the end is
  just a header print with no code after it. Kept for reference.

Edit the path constants near the top of each notebook to point at your own
local data folders.

## What the numbers actually say

**Validation set, final epoch (30), from `../results/metrics_log.csv`** —
the most reliable source since it's unambiguous numeric data, not text
read out of the PDF:

| Metric | Value |
|---|---|
| val_loss | 0.250 |
| val_accuracy | 0.996 |
| val_dice_score | 0.782 |
| val_iou_score | 0.643 |
| val_precision | 0.777 |
| val_recall | 0.788 |

**Test set** (from the report's own table, section 9.6 — clean decimal
digits, high confidence):

| Metric | Value |
|---|---|
| Test Loss | 0.2960 |
| Test Dice Score | 0.7040 |
| Test IoU Score | 0.5432 |
| Test Precision | 0.3775 |
| Test Recall | 0.3466 |
| Test Accuracy | 0.9985 |

Test precision/recall are notably lower than validation — worth keeping in
mind given the test set is only 35 images. See the root README's Results
section for the full writeup; this file is implementation detail.
