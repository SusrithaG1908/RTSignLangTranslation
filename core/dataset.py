"""
dataset.py
Classes for downloading the Kaggle ASL dataset and organising it into
train / val / test splits.
"""

import os
import shutil
import random
import subprocess
from pathlib import Path
from typing import Tuple


class DatasetDownloader:
    """
    Downloads a Kaggle dataset using the Kaggle CLI.

    Requires ~/.kaggle/kaggle.json to be present.

    Parameters
    ----------
    dataset_slug : str
        Kaggle dataset identifier, e.g. "grassknoted/asl-alphabet".
    raw_dir : Path
        Destination folder for the downloaded + unzipped files.
    """

    def __init__(self, dataset_slug: str, raw_dir: Path):
        self.dataset_slug = dataset_slug
        self.raw_dir = raw_dir

    def download(self):
        print(f"📥 Downloading dataset '{self.dataset_slug}' from Kaggle…")
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                "kaggle", "datasets", "download",
                "-d", self.dataset_slug,
                "-p", str(self.raw_dir),
                "--unzip",
            ],
            check=True,
        )
        print(f"✅ Download complete → {self.raw_dir}")


class DatasetOrganizer:
    """
    Splits a flat per-class image folder into train / val / test directories.

    Parameters
    ----------
    src_root : Path
        Root of the raw dataset (one subfolder per class label).
    out_dir : Path
        Where to write the split dataset.
    split : (float, float, float)
        Fractions for (train, val, test).  Must sum to ≤ 1.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        src_root: Path,
        out_dir: Path,
        split: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ):
        self.src_root = src_root
        self.out_dir = out_dir
        self.split = split
        self.seed = seed

    def organize(self):
        random.seed(self.seed)

        for split_name in ("train", "val", "test"):
            (self.out_dir / split_name).mkdir(parents=True, exist_ok=True)

        for label in os.listdir(self.src_root):
            label_path = self.src_root / label
            if not label_path.is_dir():
                continue

            images = os.listdir(label_path)
            random.shuffle(images)

            n = len(images)
            n_train = int(self.split[0] * n)
            n_val = int(self.split[1] * n)

            splits = {
                "train": images[:n_train],
                "val":   images[n_train: n_train + n_val],
                "test":  images[n_train + n_val:],
            }

            for split_name, imgs in splits.items():
                dst_dir = self.out_dir / split_name / label
                dst_dir.mkdir(parents=True, exist_ok=True)
                for img in imgs:
                    shutil.copy(str(label_path / img), str(dst_dir / img))

            print(
                f"  {label}: "
                f"train={len(splits['train'])}, "
                f"val={len(splits['val'])}, "
                f"test={len(splits['test'])}"
            )

        print(f"\n🎉 Dataset ready at: {self.out_dir}")
