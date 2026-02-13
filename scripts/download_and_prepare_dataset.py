# Install Kaggle CLI
# Get Kaggle API Token (https://www.kaggle.com/settings) and place it in ~/.kaggle/kaggle.json
# Run python scripts/download_and_prepare_dataset.py

import os
import shutil
import random
import subprocess
from pathlib import Path

# -------- Resolve project root safely --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Config
DATASET = "grassknoted/asl-alphabet"
RAW_DIR = PROJECT_ROOT / "data_raw"
OUT_DIR = PROJECT_ROOT / "data"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test

def download_dataset():
    print("📥 Downloading ASL Alphabet dataset from Kaggle...")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET,
        "-p", str(RAW_DIR),
        "--unzip"
    ], check=True)

def organize_dataset():
    src_root = RAW_DIR / "asl_alphabet_train" / "asl_alphabet_train"

    for split in ["train", "val", "test"]:
        (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

    for label in os.listdir(src_root):
        label_path = src_root / label
        if not label_path.is_dir():
            continue

        images = os.listdir(label_path)
        random.shuffle(images)

        n = len(images)
        n_train = int(SPLIT[0] * n)
        n_val = int(SPLIT[1] * n)

        train_imgs = images[:n_train]
        val_imgs = images[n_train:n_train + n_val]
        test_imgs = images[n_train + n_val:]

        for split_name, split_imgs in zip(
            ["train", "val", "test"],
            [train_imgs, val_imgs, test_imgs]
        ):
            split_label_dir = OUT_DIR / split_name / label
            split_label_dir.mkdir(parents=True, exist_ok=True)

            for img in split_imgs:
                src = label_path / img
                dst = split_label_dir / img
                shutil.copy(str(src), str(dst))

        print(f"✅ {label}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

def main():
    download_dataset()
    organize_dataset()
    print(f"\n🎉 Dataset ready at: {OUT_DIR}/train, {OUT_DIR}/val, {OUT_DIR}/test")

if __name__ == "__main__":
    main()
