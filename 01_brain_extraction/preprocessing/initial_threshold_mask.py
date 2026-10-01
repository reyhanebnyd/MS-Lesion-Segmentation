"""Quick-and-dirty brain mask via Gaussian blur + Otsu-style threshold + largest
connected component + hole filling.

This was the first-pass brain-extraction method (stage 1 of the pipeline
described in the report): fast, but noisy around the skull boundary and the
reason a U-Net (see ../pytorch and ../keras) was trained afterward to refine
these masks. Kept here for reference / as a fallback when no trained model
is available yet.
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt

image_path = "A (7).jpg"
threshold_value = 70
show_plot = True


def make_initial_mask(image, threshold_value=70):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, threshold_value, 255, cv2.THRESH_BINARY)

    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(opened)
    if num_labels <= 1:
        return np.zeros_like(image)
    largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    brain_mask = (labels == largest_label).astype(np.uint8) * 255

    # Fill holes fully enclosed by the brain contour (e.g. ventricles)
    contours, hierarchy = cv2.findContours(brain_mask.copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    for i in range(len(contours)):
        if hierarchy[0][i][3] != -1:
            cv2.drawContours(brain_mask, contours, i, 255, -1)

    return brain_mask


if __name__ == "__main__":
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    brain_mask = make_initial_mask(image, threshold_value)
    segmented_brain = cv2.bitwise_and(image, image, mask=brain_mask)

    if show_plot:
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 3, 1)
        plt.title("Original")
        plt.imshow(image, cmap="gray")
        plt.subplot(1, 3, 2)
        plt.title("Brain Mask")
        plt.imshow(brain_mask, cmap="gray")
        plt.subplot(1, 3, 3)
        plt.title("Segmented Brain")
        plt.imshow(segmented_brain, cmap="gray")
        plt.show()
