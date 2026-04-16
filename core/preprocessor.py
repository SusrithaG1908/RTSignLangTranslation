"""
preprocessor.py
Converts a raw BGR image (after optional cropping) into a model-ready
numpy batch.  Keeps all resize / normalise logic in one place.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class PreprocessResult:
    """Batch-ready array plus any metadata the caller might need."""
    batch: np.ndarray          # shape (1, H, W, 3)
    img_size: Tuple[int, int]  # (width, height) that was used


class ImagePreprocessor:
    """
    Resizes an image and applies the appropriate normalisation.

    Parameters
    ----------
    img_size : (width, height)
    mode : "cnn" | "mobilenet"
        "cnn"       → divide by 255.0
        "mobilenet" → tf.keras MobileNetV2 preprocess_input  (scales to [-1, 1])
    """

    def __init__(self, img_size: Tuple[int, int], mode: str = "cnn"):
        self.img_size = img_size
        self.mode = mode.lower()
        if self.mode not in ("cnn", "mobilenet"):
            raise ValueError(f"mode must be 'cnn' or 'mobilenet', got '{mode}'")

        if self.mode == "mobilenet":
            # Import lazily so the class can be imported without TF when only
            # using other parts of the module.
            from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
            self._mobilenet_preprocess = preprocess_input

    # ------------------------------------------------------------------
    def process(self, img_bgr: np.ndarray) -> PreprocessResult:
        """
        Convert *img_bgr* → (1, H, W, 3) batch ready for model.predict().

        Always converts BGR → RGB before any further processing.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, self.img_size)

        if self.mode == "mobilenet":
            arr = self._mobilenet_preprocess(img_resized.astype(np.float32))
        else:
            arr = img_resized / 255.0

        return PreprocessResult(
            batch=np.expand_dims(arr, axis=0),
            img_size=self.img_size,
        )
