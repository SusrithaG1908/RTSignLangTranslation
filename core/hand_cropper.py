"""
hand_cropper.py
Wraps MediaPipe hand detection and bounding-box crop logic so every
script uses exactly the same crop strategy.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class CropResult:
    """Result returned by HandCropper.crop()."""
    image_bgr: np.ndarray          # cropped (and optionally resized) BGR image
    used_mediapipe: bool           # True  → landmarks found; False → fallback crop
    bounding_box: Optional[Tuple[int, int, int, int]] = None  # (x1,y1,x2,y2) or None
    landmarks: object = None       # raw MediaPipe hand landmarks object, if detected


class HandCropper:
    """
    Detects a single hand with MediaPipe and returns a tight crop.

    Parameters
    ----------
    static_image_mode : bool
        True for still images (dataset processing / single-image predict).
        False for live video (webcam).
    min_detection_confidence : float
        Lower values catch tricky signs (R, S, T) at the cost of false positives.
    min_tracking_confidence : float
        Only used in video mode (static_image_mode=False).
    pad : int
        Pixel padding added around the detected bounding box.
    fallback_crop : tuple[float, float, float, float]
        (y_start%, y_end%, x_start%, x_end%) fractions used when MediaPipe
        fails to detect a hand.
    output_size : tuple[int,int] or None
        If given, the crop is resized to (width, height) before returning.
    """

    def __init__(
        self,
        static_image_mode: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        pad: int = 30,
        fallback_crop: Tuple[float, float, float, float] = (0.2, 0.9, 0.2, 0.9),
        output_size: Optional[Tuple[int, int]] = None,
    ):
        self.pad = pad
        self.fallback_crop = fallback_crop
        self.output_size = output_size

        mp_hands = mp.solutions.hands
        self._hands = mp_hands.Hands(
            static_image_mode=static_image_mode,
            max_num_hands=1,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

    # ------------------------------------------------------------------
    def crop(self, img_bgr: np.ndarray) -> CropResult:
        """
        Detect the hand in *img_bgr* and return a CropResult.

        The returned image is always BGR.  If *output_size* was set it is
        resized; otherwise it keeps its natural crop dimensions.
        """
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        results = self._hands.process(img_rgb)
        h, w = img_bgr.shape[:2]

        if results.multi_hand_landmarks:
            xs, ys = [], []
            for lm in results.multi_hand_landmarks[0].landmark:
                xs.append(int(lm.x * w))
                ys.append(int(lm.y * h))

            x1 = max(min(xs) - self.pad, 0)
            y1 = max(min(ys) - self.pad, 0)
            x2 = min(max(xs) + self.pad, w)
            y2 = min(max(ys) + self.pad, h)
            crop_bgr = img_bgr[y1:y2, x1:x2]
            bbox = (x1, y1, x2, y2)
            used_mp = True
            landmarks = results.multi_hand_landmarks[0]
        else:
            fy0, fy1, fx0, fx1 = self.fallback_crop
            crop_bgr = img_bgr[int(fy0 * h):int(fy1 * h), int(fx0 * w):int(fx1 * w)]
            bbox = None
            used_mp = False
            landmarks = None

        if self.output_size is not None:
            crop_bgr = cv2.resize(crop_bgr, self.output_size)

        return CropResult(image_bgr=crop_bgr, used_mediapipe=used_mp, bounding_box=bbox, landmarks=landmarks)

    # ------------------------------------------------------------------
    def close(self):
        """Release MediaPipe resources."""
        self._hands.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()