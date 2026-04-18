"""
crop_existing_dataset_with_mediapipe.py
Walks data/{train,val,test} and writes a MediaPipe-cropped copy to
data_mp/{train,val,test}.
"""

import sys
import os
from pathlib import Path

from tqdm import tqdm
import cv2

from src.core.hand_cropper import HandCropper

# ---- Project layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "data"
DST_ROOT = PROJECT_ROOT / "data_mp"
OUTPUT_SIZE = (128, 128)


def process_split(split_name: str, cropper: HandCropper):
    src_split = SRC_ROOT / split_name
    dst_split = DST_ROOT / split_name
    dst_split.mkdir(parents=True, exist_ok=True)

    for label in os.listdir(src_split):
        src_label_dir = src_split / label
        if not src_label_dir.is_dir():
            continue

        dst_label_dir = dst_split / label
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for fname in tqdm(os.listdir(src_label_dir), desc=f"{split_name}/{label}"):
            img = cv2.imread(str(src_label_dir / fname))
            if img is None:
                continue
            result = cropper.crop(img)
            cv2.imwrite(str(dst_label_dir / fname), result.image_bgr)


def main():
    # static_image_mode=True, lower confidence to catch tricky signs (R, S, T)
    with HandCropper(
        static_image_mode=True,
        min_detection_confidence=0.2,
        output_size=OUTPUT_SIZE,
    ) as cropper:
        for split in ("train", "val", "test"):
            print(f"\n🔄 Processing split: {split}")
            process_split(split, cropper)

    print(f"\n🎉 MediaPipe-cropped dataset saved to: {DST_ROOT}")


if __name__ == "__main__":
    main()
