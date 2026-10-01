"""Generate synthetic MS-plaque training data.

Pipeline (see the report, section 7, for the full write-up):
  1. Extract real plaques from the labeled training set and give each one a
     randomized Bezier-curve radial deformation so repeated placements of the
     same plaque don't look identical.
  2. Build a heatmap of where plaques statistically occur (from plaque
     centroids), used to sample biologically plausible placement locations.
  3. For each synthetic image: pick a number of plaques (sampled from the
     real per-image plaque-count distribution), randomly rotate/scale/
     elastically-deform each one, place it at a heatmap-and-brain-mask-guided
     location, and blend it into the target MRI with a soft (Gaussian-
     feathered) mask.

Originally developed in Colab; the `!pip install`, Drive-mount and other
notebook-only lines have been removed here. Point the paths below at your
own local copies of the data before running.
"""
import os
import random

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
from skimage.measure import label, regionprops

# -------------------- Paths (edit these for your local setup) --------------------
input_folder = "Train-Input"
mask_folder = "Train-Mask"

out_input_folder = "output/synthetic-input"
out_mask_folder = "output/synthetic-mask"
debug_folder = "output/debug"

# A single real MRI slice + its (manually corrected) brain mask used as the
# background that synthetic plaques get composited onto.
target_file = "sample_data/A (15).jpg"
brain_mask_file = "sample_data/A (15)_mask_edited.png"

os.makedirs(out_input_folder, exist_ok=True)
os.makedirs(out_mask_folder, exist_ok=True)
os.makedirs(debug_folder, exist_ok=True)

# -------------------- Parameters --------------------
aug_total = 500  # number of synthetic images to generate
heatmap_sigma = 20.0  # smoothing applied to the plaque-location heatmap
feather_sigma = 6.0  # softness of a placed plaque's blended edge
shape_factor = 0.12  # strength of the Bezier radial deformation
n_anchors = 8  # control points for the Bezier deformation
n_angles = 360  # angular resolution of the radial profile

random_seed = 42
random.seed(random_seed)
np.random.seed(random_seed)

valid_exts = (".png", ".jpg", ".jpeg")


# -------------------- Utilities --------------------

def ensure_uint8(img):
    if img.dtype != np.uint8:
        return np.clip(img, 0, 255).astype(np.uint8)
    return img


def centroid_from_mask(mask):
    m = cv2.moments((mask > 127).astype(np.uint8))
    if m["m00"] == 0:
        h, w = mask.shape[:2]
        return (w / 2.0, h / 2.0)
    cx = m["m10"] / m["m00"]
    cy = m["m01"] / m["m00"]
    return (cx, cy)


def radial_profile_from_mask(mask, n_angles=360):
    mask_u8 = (mask > 127).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) == 0:
        raise ValueError("No contour found in mask")
    contour = max(contours, key=lambda c: cv2.contourArea(c)).reshape(-1, 2)
    cx, cy = centroid_from_mask(mask)
    xs = contour[:, 0].astype(np.float32)
    ys = contour[:, 1].astype(np.float32)
    dx = xs - cx
    dy = ys - cy
    angs = np.mod(np.arctan2(dy, dx), 2 * np.pi)
    rads = np.sqrt(dx * dx + dy * dy)
    order = np.argsort(angs)
    angs_s = angs[order]
    rads_s = rads[order]
    angs_ext = np.concatenate([angs_s, angs_s[:1] + 2 * np.pi])
    rads_ext = np.concatenate([rads_s, rads_s[:1]])
    sample_angles = np.linspace(0.0, 2 * np.pi, n_angles, endpoint=False)
    r_sample = np.interp(sample_angles, angs_ext, rads_ext)
    return sample_angles, r_sample, (cx, cy), contour


def cubic_bezier_scalar(p0, p1, p2, p3, t):
    t = np.asarray(t)
    u = 1.0 - t
    return (u ** 3) * p0 + 3 * (u ** 2) * t * p1 + 3 * u * (t ** 2) * p2 + (t ** 3) * p3


def bezier_radial_deform(angles, r_orig, n_anchors=8, shape_factor=0.12, random_state=None):
    if random_state is None:
        rng = np.random.RandomState()
    elif isinstance(random_state, int):
        rng = np.random.RandomState(random_state)
    else:
        rng = random_state

    anchors = np.linspace(0.0, 2 * np.pi, n_anchors, endpoint=False)
    base_r = np.interp(anchors, angles, r_orig)
    offsets = rng.uniform(-shape_factor, shape_factor, size=n_anchors) * base_r
    anchor_r = base_r + offsets

    r_new = np.zeros_like(r_orig)
    for i in range(n_anchors):
        j = (i + 1) % n_anchors
        a0 = anchors[i]
        a3 = anchors[j] + (0 if j > i else 2 * np.pi)
        angs_ext_wrap = np.where(angles < a0, angles + 2 * np.pi, angles)
        mask_interval = (angs_ext_wrap >= a0) & (angs_ext_wrap <= a3)
        idxs = np.where(mask_interval)[0]
        if idxs.size == 0:
            continue
        t_local = (angs_ext_wrap[idxs] - a0) / float(a3 - a0)
        p0 = anchor_r[i]
        p3 = anchor_r[j]
        p1 = p0 + (p3 - p0) * 0.3
        p2 = p0 + (p3 - p0) * 0.7
        r_new[idxs] = cubic_bezier_scalar(p0, p1, p2, p3, t_local)

    zero_mask = r_new <= 0
    if np.any(zero_mask):
        r_new[zero_mask] = r_orig[zero_mask]
    return r_new


def deform_plaque_shape_gray(plaque_img, plaque_mask, shape_factor_local=0.12,
                              n_anchors_local=8, n_angles_local=360, random_state_local=None):
    plaque_img = ensure_uint8(plaque_img)
    plaque_mask_bin = (plaque_mask > 0).astype(np.uint8) * 255
    if plaque_mask_bin.sum() == 0:
        return None, None
    try:
        angles, r_orig, centroid, _ = radial_profile_from_mask(plaque_mask_bin, n_angles=n_angles_local)
        r_new = bezier_radial_deform(angles, r_orig, n_anchors=n_anchors_local,
                                      shape_factor=shape_factor_local, random_state=random_state_local)

        Hs, Ws = plaque_mask_bin.shape
        ys, xs = np.mgrid[0:Hs, 0:Ws]
        cx, cy = centroid
        dx = xs - cx
        dy = ys - cy
        r_pixel = np.sqrt(dx * dx + dy * dy)
        theta = np.mod(np.arctan2(dy, dx), 2 * np.pi)

        r_new_theta = np.interp(theta.flatten(), angles, r_new).reshape(theta.shape)
        r_orig_theta = np.interp(theta.flatten(), angles, r_orig).reshape(theta.shape)

        eps = 1e-6
        denom = r_new_theta.copy()
        denom[denom <= 0] = eps
        frac_clipped = np.clip(r_pixel / denom, 0.0, 1.0)
        r_src = frac_clipped * r_orig_theta

        xs_src = np.clip(cx + r_src * np.cos(theta), 0, Ws - 1).flatten()
        ys_src = np.clip(cy + r_src * np.sin(theta), 0, Hs - 1).flatten()

        src = plaque_img.astype(np.float32)
        sampled = src[ys_src.astype(int), xs_src.astype(int)].reshape(Hs, Ws)
        sampled_u8 = np.clip(sampled, 0, 255).astype(np.uint8)

        new_mask = (r_pixel <= r_new_theta).astype(np.uint8)
        if new_mask.sum() == 0:
            return None, None
        return sampled_u8, new_mask
    except Exception:
        return None, None


def rotate_scale_image(image, mask, angle, scale):
    """Rotate and scale a plaque crop and its mask together."""
    h, w = mask.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_img = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    rotated_mask = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rotated_img, rotated_mask


def elastic_displacement_field(shape, alpha=2, sigma=1, random_state=None):
    """Generate a random displacement field for elastic_transform."""
    if random_state is None:
        random_state = np.random.RandomState(None)
    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    return dx, dy


def elastic_transform(image, displacement_field):
    """Apply a precomputed elastic displacement field to make plaques look
    more organic. The same field must be reused for a plaque's image and its
    mask (see the main loop) so the two stay aligned after warping.
    """
    dx, dy = displacement_field
    shape = image.shape
    x, y = np.meshgrid(np.arange(shape[1]), np.arange(shape[0]))
    indices = np.reshape(y + dy, (-1, 1)), np.reshape(x + dx, (-1, 1))
    return map_coordinates(image, indices, order=1, mode="reflect").reshape(shape)


def random_irregular_mask(mask, iterations=3):
    """Turn a (roughly) convex mask into an irregular lesion-like blob."""
    for _ in range(iterations):
        ksize = np.random.randint(3, 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
        if np.random.rand() > 0.5:
            mask = cv2.dilate(mask, kernel, iterations=1)
        else:
            mask = cv2.erode(mask, kernel, iterations=1)
    return (mask > 0).astype(np.uint8)


def main():
    # -------------------- Step 0: Per-image plaque-count distribution --------------------
    plaque_counts = []
    for mask_file in os.listdir(mask_folder):
        if mask_file.lower().endswith(valid_exts):
            mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
            _, thresh = cv2.threshold(mask, 10, 1, cv2.THRESH_BINARY)
            labeled_array = label(thresh, connectivity=2)
            plaque_counts.append(labeled_array.max())
    plaque_counts = np.array(plaque_counts)

    # -------------------- Step 1: Build the plaque-location heatmap --------------------
    first_mask_file = next(f for f in os.listdir(mask_folder) if f.lower().endswith(valid_exts))
    sample_mask = cv2.imread(os.path.join(mask_folder, first_mask_file), cv2.IMREAD_GRAYSCALE)
    H, W = sample_mask.shape
    heatmap = np.zeros((H, W), dtype=np.float32)

    for mask_file in os.listdir(mask_folder):
        if not mask_file.lower().endswith(valid_exts):
            continue
        mask = cv2.imread(os.path.join(mask_folder, mask_file), cv2.IMREAD_GRAYSCALE)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        binary = (mask > 10).astype(np.uint8)
        labeled = label(binary, connectivity=2)
        for reg in regionprops(labeled):
            cy, cx = reg.centroid
            iy, ix = int(round(cy)), int(round(cx))
            if 0 <= iy < H and 0 <= ix < W:
                heatmap[iy, ix] += 1.0

    heatmap_smooth = gaussian_filter(heatmap, sigma=heatmap_sigma)
    prob_map = heatmap_smooth if heatmap_smooth.sum() > 0 else np.ones_like(heatmap_smooth)
    prob_map = prob_map.astype(np.float64)
    prob_map /= prob_map.sum()

    # -------------------- Step 2: Extract + deform all plaques --------------------
    plaques = []
    for mask_file in os.listdir(mask_folder):
        if not mask_file.lower().endswith(valid_exts):
            continue
        mask_path = os.path.join(mask_folder, mask_file)
        base, _ = os.path.splitext(mask_file)
        input_path = None
        for ext in valid_exts:
            candidate = os.path.join(input_folder, base + ext)
            if os.path.exists(candidate):
                input_path = candidate
                break
        if input_path is None:
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
        if mask.shape != (H, W):
            mask = cv2.resize(mask, (W, H), interpolation=cv2.INTER_NEAREST)
        if img.shape != (H, W):
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        binary = (mask > 10).astype(np.uint8)
        labeled = label(binary, connectivity=2)
        for reg in regionprops(labeled):
            if reg.area < 10:
                continue
            minr, minc, maxr, maxc = reg.bbox
            plaque_mask = (labeled == reg.label).astype(np.uint8)[minr:maxr, minc:maxc]
            plaque_crop = img[minr:maxr, minc:maxc].copy()
            new_crop, new_mask = deform_plaque_shape_gray(plaque_crop, plaque_mask, shape_factor_local=shape_factor)
            if new_crop is not None and new_mask is not None:
                plaques.append((new_crop, new_mask))
    print(f"{len(plaques)} plaques extracted.")

    # -------------------- Step 3: Load the target background MRI + brain mask --------------------
    target_img = cv2.imread(target_file, cv2.IMREAD_GRAYSCALE)
    brain_mask = cv2.imread(brain_mask_file, cv2.IMREAD_GRAYSCALE)
    brain_mask = (brain_mask > 0).astype(np.uint8)

    if target_img.shape != (H, W):
        target_img = cv2.resize(target_img, (W, H))
    if brain_mask.shape != (H, W):
        brain_mask = cv2.resize(brain_mask, (W, H), interpolation=cv2.INTER_NEAREST)

    # Plaque placement probability = location heatmap restricted to the brain.
    # Fixed for the whole run, so compute it once instead of per-plaque.
    combined_prob = prob_map * brain_mask
    if combined_prob.sum() == 0:
        combined_prob = np.ones_like(prob_map)
    combined_prob /= combined_prob.sum()
    combined_flat = combined_prob.flatten()
    coords_flat = np.arange(H * W)

    # -------------------- Step 4: Generate synthetic images --------------------
    for i in range(aug_total):
        aug_img = target_img.copy()
        aug_mask = np.zeros_like(target_img, dtype=np.uint8)
        n_plaques = int(np.random.choice(plaque_counts))

        for _ in range(n_plaques):
            placed = False
            for _attempt in range(20):  # retry a few times if placement fails
                plaque_crop, plaque_mask = random.choice(plaques)

                scale_factor = np.random.uniform(0.7, 1.3)
                angle = np.random.uniform(-45, 45)
                plaque_aug, mask_aug = rotate_scale_image(plaque_crop, plaque_mask, angle, scale_factor)

                # Same displacement field for the image and its mask, or the
                # plaque texture and its mask boundary drift apart.
                displacement_field = elastic_displacement_field(plaque_aug.shape, alpha=2, sigma=1)
                plaque_aug = elastic_transform(plaque_aug, displacement_field)
                mask_aug = elastic_transform(mask_aug.astype(np.float32), displacement_field) > 0.5
                mask_aug = mask_aug.astype(np.uint8)

                mask_aug = random_irregular_mask(mask_aug, iterations=np.random.randint(2, 5))

                soft_mask = gaussian_filter(mask_aug.astype(np.float32), sigma=feather_sigma)
                if soft_mask.max() > 0:
                    soft_mask /= soft_mask.max()

                plaque_aug = cv2.GaussianBlur(plaque_aug, (3, 3), 0)

                ph, pw = mask_aug.shape
                idx = np.random.choice(coords_flat, p=combined_flat)
                cy, cx = divmod(idx, W)
                new_minr, new_minc = int(round(cy - ph / 2)), int(round(cx - pw / 2))
                new_maxr, new_maxc = new_minr + ph, new_minc + pw

                if new_minr < 0 or new_minc < 0 or new_maxr > H or new_maxc > W:
                    continue

                brain_region = brain_mask[new_minr:new_maxr, new_minc:new_maxc]
                if (mask_aug * brain_region).sum() / (mask_aug.sum() + 1e-6) < 0.5:
                    continue

                target_region = aug_img[new_minr:new_maxr, new_minc:new_maxc]
                blended = soft_mask * plaque_aug + (1 - soft_mask) * target_region
                blended = np.clip(blended, 0, 255).astype(np.uint8)

                aug_img[new_minr:new_maxr, new_minc:new_maxc] = blended
                aug_mask[new_minr:new_maxr, new_minc:new_maxc][mask_aug > 0] = 255

                placed = True
                break

            if not placed:
                continue

        out_name = f"aug_{i:04d}.png"
        cv2.imwrite(os.path.join(out_input_folder, out_name), aug_img)
        cv2.imwrite(os.path.join(out_mask_folder, out_name), aug_mask)
        print(f"{i + 1}/{aug_total} augmented images created.")


if __name__ == "__main__":
    main()
