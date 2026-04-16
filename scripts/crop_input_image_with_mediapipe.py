"""
crop_input_image_with_mediapipe.py
Crops a single test image using MediaPipe hand detection and saves the result.
"""

import sys
import cv2
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.hand_cropper import HandCropper

# ---- Project layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
IMAGE_DIR = PROJECT_ROOT / "test_images"
OUT_DIR   = PROJECT_ROOT / "cropped_images"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_IMAGE = "sample.jpg"


def main():
    img_name = input(f"Enter image name (default: {DEFAULT_IMAGE}): ").strip() or DEFAULT_IMAGE
    img_path = IMAGE_DIR / img_name

    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Image not found: {img_path}")

    with HandCropper(static_image_mode=True, min_detection_confidence=0.2) as cropper:
        result = cropper.crop(img_bgr)

    if result.used_mediapipe:
        print("✅ MediaPipe detected hand")
        x1, y1, x2, y2 = result.bounding_box
        cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
    else:
        print("⚠️  MediaPipe failed — using fallback crop")

    out_path = OUT_DIR / f"cropped_{img_name}"
    cv2.imwrite(str(out_path), result.image_bgr)
    print(f"✅ Saved cropped hand to: {out_path}")

    cv2.imshow("Hand Crop Saved", img_bgr)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
