# Install Kaggle CLI
# Get Kaggle API Token (https://www.kaggle.com/settings) and place it in ~/.kaggle/kaggle.json
# Run python scripts/download_and_prepare_dataset.py

import os
import shutil
import zipfile
import random
import subprocess

# Config
DATASET = "grassknoted/asl-alphabet"
RAW_DIR = "data_raw"
OUT_DIR = "data"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test

def download_dataset():
    print("📥 Downloading ASL Alphabet dataset from Kaggle...")
    os.makedirs(RAW_DIR, exist_ok=True)
    subprocess.run([
        "kaggle", "datasets", "download",
        "-d", DATASET,
        "-p", RAW_DIR,
        "--unzip"
    ], check=True)

def organize_dataset():
    src_root = os.path.join(RAW_DIR, "asl_alphabet_train", "asl_alphabet_train")

    for split in ["train", "val", "test"]:
        os.makedirs(os.path.join(OUT_DIR, split), exist_ok=True)

    for label in os.listdir(src_root):
        label_path = os.path.join(src_root, label)
        if not os.path.isdir(label_path):
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
            split_label_dir = os.path.join(OUT_DIR, split_name, label)
            os.makedirs(split_label_dir, exist_ok=True)

            for img in split_imgs:
                src = os.path.join(label_path, img)
                dst = os.path.join(split_label_dir, img)
                shutil.copy(src, dst)

        print(f"✅ {label}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")

def main():
    download_dataset()
    organize_dataset()
    print("\n🎉 Dataset ready at: data/train, data/val, data/test")

if __name__ == "__main__":
    main()
