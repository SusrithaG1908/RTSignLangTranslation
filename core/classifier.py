"""
classifier.py
Loads a Keras model + class-label JSON and exposes a single predict()
method used by every inference script.
"""

import json
import numpy as np
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union


@dataclass
class Prediction:
    label: str
    confidence: float
    class_index: int


class SignClassifier:
    """
    Wraps a Keras `.h5` model and its matching class-label JSON file.

    Parameters
    ----------
    model_path : path-like
        Path to the saved Keras model (.h5).
    labels_path : path-like
        Path to the JSON file produced by ImageDataGenerator
        (maps label_name → integer index).
    """

    def __init__(self, model_path: Union[str, Path], labels_path: Union[str, Path]):
        from tensorflow.keras.models import load_model  # lazy import

        self.model_path = Path(model_path)
        self.labels_path = Path(labels_path)

        print(f"📦 Loading model: {self.model_path.name}")
        self._model = load_model(str(self.model_path))

        with open(self.labels_path, "r") as f:
            class_indices: Dict[str, int] = json.load(f)

        # Invert: int index → label string
        self._labels: Dict[int, str] = {int(v): k for k, v in class_indices.items()}

    # ------------------------------------------------------------------
    @property
    def num_classes(self) -> int:
        return len(self._labels)

    @property
    def labels_sorted(self):
        """List of label strings ordered by class index (useful for confusion matrices)."""
        return [self._labels[i] for i in range(self.num_classes)]

    # ------------------------------------------------------------------
    def predict(self, batch: np.ndarray) -> Prediction:
        """
        Run inference on a pre-processed batch (shape (1, H, W, 3)).

        Returns a Prediction with the top-1 label, confidence, and index.
        """
        preds = self._model.predict(batch, verbose=0)
        class_index = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = self._labels.get(class_index, "Unknown")
        return Prediction(label=label, confidence=confidence, class_index=class_index)

    def predict_all_scores(self, batch: np.ndarray) -> np.ndarray:
        """Return the raw softmax array (shape (1, num_classes))."""
        return self._model.predict(batch, verbose=0)
