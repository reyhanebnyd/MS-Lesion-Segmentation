# Stage 1 — Brain Extraction

Before any plaque analysis, the brain region has to be isolated from skull,
scalp and eyes ("brain extraction" / "skull stripping"). This went through
a multi-step loop:

1. **`preprocessing/`** — a fast classical approach: Gaussian blur → binary
   threshold → morphological close/open → largest connected component →
   hole-filling. Good enough to bootstrap labels, but noisy around the skull
   boundary (see `../docs/assets/brain_extraction_failure_case.png` for a
   failure case on a lower, eye-level slice). `manual_mask_editor.py` is a
   small OpenCV GUI used to hand-correct these initial masks before using
   them as training labels.

2. **`pytorch/`** — a standard U-Net (64→1024 channel encoder/decoder)
   trained on the (manually corrected) masks from step 1, and then used in
   a correct-and-retrain loop (predict → manually fix the worst masks →
   retrain). `train.py` trains and reports a validation Dice score;
   `predict.py` runs a saved model over a folder of slices.

**Correction vs. an earlier version of this README:** this folder
previously also contained two Keras notebooks (`model1.ipynb`,
`model2.ipynb`) labeled as a second brain-extraction attempt. That was a
mistake — cross-checking their `model.summary()` output and training
curves against the figures embedded in the report (`docs/MSProject.pdf`)
showed they're actually the **plaque-segmentation** model search (report
section 9), not brain extraction. They've been moved to
`../04_model_training/keras_architecture_search/`.

See `../docs/assets/brain_extraction_example_*.png` for example
original/mask/segmented triptychs from the threshold-based stage, and
`../docs/assets/unet_brain_mask_example.png` for an example mask produced
by the trained model (as shown in the report).
