# Stage 2 — Plaque Analysis

`plaque_feature_analysis.ipynb` analyzes the real (training-set) plaque
masks to characterize what synthetic plaques should imitate:

- **Count distribution** — how many plaques appear per image (mean 4.37,
  std 2.99, skewness 2.60 — as reported), used in stage 3 to sample how
  many synthetic plaques to place per generated image.
  (`../docs/assets/plaque_count_histogram.png`, from the report.)
- **Spatial distribution** — plaque centroids clustered with a 6-component
  Gaussian Mixture Model (`../docs/assets/plaque_spatial_gmm_clusters.png`,
  from the report), and a smoothed 2D heatmap of plaque locations (the same
  kind of heatmap used to guide synthetic plaque placement in
  `../03_synthetic_data_generation` — see
  `../docs/assets/plaque_placement_heatmap.png`, reconstructed here from a
  leftover debug image rather than regenerated from the dataset).
- **Texture features** — per-plaque intensity mean/std/skewness, plus
  GLCM (gray-level co-occurrence matrix) contrast, homogeneity, energy and
  correlation, to describe what plaque texture looks like vs. surrounding
  brain tissue (`../docs/assets/plaque_texture_contrast_vs_correlation.png`,
  from the report). Saved to `plaque_texture_features.csv` when run (not
  included here — wasn't present in the original working folder).

**Known quirk, not changed:** the plaque-count cell and the texture-feature
cell match mask files ending in `.png`, `.jpg`, or `.jpeg`, but the spatial
distribution / GMM-clustering cell only matches `.png`. If any masks in your
dataset are `.jpg`, that cell silently analyzes a smaller/different subset
than the other two. Left as-is rather than "fixed" since changing it would
change already-reported numbers without your say-so — worth checking your
mask file extensions before trusting the spatial/cluster figures.
