"""
download_and_prepare_dataset.py
Downloads the Kaggle ASL Alphabet dataset and organises it into
train / val / test splits.

Install Kaggle CLI and place your API token at ~/.kaggle/kaggle.json.
"""

import sys
from pathlib import Path

from src.core.dataset import DatasetDownloader, DatasetOrganizer

# ---- Project layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR  = PROJECT_ROOT / "data_raw"
OUT_DIR  = PROJECT_ROOT / "data"

# ---- Config ----
KAGGLE_DATASET = "grassknoted/asl-alphabet"
SPLIT = (0.7, 0.15, 0.15)  # train, val, test


def main():
    downloader = DatasetDownloader(dataset_slug=KAGGLE_DATASET, raw_dir=RAW_DIR)
    downloader.download()

    src_root = RAW_DIR / "asl_alphabet_train" / "asl_alphabet_train"
    organizer = DatasetOrganizer(src_root=src_root, out_dir=OUT_DIR, split=SPLIT)
    organizer.organize()

    print(f"\n🎉 Dataset ready at: {OUT_DIR}/{{train,val,test}}")


if __name__ == "__main__":
    main()
