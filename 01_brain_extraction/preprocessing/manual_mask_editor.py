"""Tiny OpenCV GUI for manually correcting a predicted brain mask.

Controls:
  d        switch to DRAW mode (add to mask)
  e        switch to ERASE mode (remove from mask)
  +  / -   grow / shrink the brush
  s        save the edited mask
  q        quit
"""
import cv2

image_path = "Images1/A (212).jpg"
predicted_mask_path = "output_masks1/A (212)_mask_pred.png"
edited_mask_path = "editmask1/A (212)_mask_edited.png"
brush_radius = 8

drawing = False
mode = "draw"  # 'draw' or 'erase'


def draw(event, x, y, flags, param):
    global drawing, mask, mode
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        color = 255 if mode == "draw" else 0
        cv2.circle(mask, (x, y), brush_radius, color, -1)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False


if __name__ == "__main__":
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    mask = cv2.imread(predicted_mask_path, cv2.IMREAD_GRAYSCALE)

    if image.shape != mask.shape:
        mask = cv2.resize(mask, (image.shape[1], image.shape[0]))

    cv2.namedWindow("Edit Mask")
    cv2.setMouseCallback("Edit Mask", draw)

    print("Instructions:")
    print(" - Press 'd' to switch to DRAW mode (add to mask).")
    print(" - Press 'e' to switch to ERASE mode (remove from mask).")
    print(" - Press 's' to save the edited mask.")
    print(" - Press 'q' to quit.")

    while True:
        blended = cv2.addWeighted(image, 0.7, mask, 0.3, 0)
        display = cv2.cvtColor(blended, cv2.COLOR_GRAY2BGR)

        label = "Mode: DRAW" if mode == "draw" else "Mode: ERASE"
        color = (0, 255, 0) if mode == "draw" else (0, 0, 255)
        cv2.putText(display, label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        cv2.imshow("Edit Mask", display)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("d"):
            mode = "draw"
        elif key == ord("e"):
            mode = "erase"
        elif key in (ord("+"), ord("=")):
            brush_radius += 1
            print(f"Brush size increased to {brush_radius}")
        elif key == ord("-") and brush_radius > 1:
            brush_radius -= 1
            print(f"Brush size decreased to {brush_radius}")
        elif key == ord("s"):
            cv2.imwrite(edited_mask_path, mask)
            print(f"Mask saved as '{edited_mask_path}'")
        elif key == ord("q"):
            break

    cv2.destroyAllWindows()
