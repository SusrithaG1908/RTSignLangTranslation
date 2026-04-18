"""
train_pipelines.py
Trains all four model pipelines (CNN raw, CNN+MP, MobileNet 10 %, MobileNet 25 %).
"""

import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from pathlib import Path

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
from pathlib import Path
from dataclasses import dataclass
from typing import Callable, Optional, Tuple


# ---- Project layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_TRAIN = PROJECT_ROOT / "data"    / "train"
DATA_RAW_VAL   = PROJECT_ROOT / "data"    / "val"
DATA_MP_TRAIN  = PROJECT_ROOT / "data_mp" / "train"
DATA_MP_VAL    = PROJECT_ROOT / "data_mp" / "val"
MODELS_DIR     = PROJECT_ROOT / "models"
LOGS_DIR       = PROJECT_ROOT / "train_logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE_CNN       = (128, 128)
IMG_SIZE_MOBILENET = (224, 224)
BATCH_SIZE         = 32
EPOCHS             = 30
FINE_TUNE_EPOCHS   = 10


# =============================================================================
# Data generators
# =============================================================================

class AugmentedDataGenerators:
    """
    Builds ImageDataGenerator train + val generators for a given directory pair.

    Parameters
    ----------
    train_dir, val_dir : Path
    img_size : (width, height)
    use_mobilenet_preprocess : bool
        True → MobileNetV2 preprocess_input; False → rescale 1/255.
    batch_size : int
    """

    def __init__(
        self,
        train_dir: Path,
        val_dir: Path,
        img_size: Tuple[int, int],
        use_mobilenet_preprocess: bool = False,
        batch_size: int = BATCH_SIZE,
    ):
        if use_mobilenet_preprocess:
            train_datagen = ImageDataGenerator(
                preprocessing_function=preprocess_input,
                brightness_range=(0.7, 1.3),
                zoom_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=10,
                shear_range=0.1,
            )
            val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
        else:
            train_datagen = ImageDataGenerator(
                rescale=1.0 / 255,
                brightness_range=(0.7, 1.3),
                zoom_range=0.2,
                width_shift_range=0.1,
                height_shift_range=0.1,
                rotation_range=10,
                shear_range=0.1,
            )
            val_datagen = ImageDataGenerator(rescale=1.0 / 255)

        self.train = train_datagen.flow_from_directory(
            str(train_dir), target_size=img_size,
            batch_size=batch_size, class_mode="categorical",
        )
        self.val = val_datagen.flow_from_directory(
            str(val_dir), target_size=img_size,
            batch_size=batch_size, class_mode="categorical", shuffle=False,
        )

    @property
    def num_classes(self) -> int:
        return self.train.num_classes

    @property
    def labels(self):
        return list(self.train.class_indices.keys())

    @property
    def class_indices(self):
        return self.train.class_indices


# =============================================================================
# Model builders
# =============================================================================

class CNNModelBuilder:
    """Builds the custom CNN used for raw and MP-crop pipelines."""

    @staticmethod
    def build(num_classes: int, input_shape: Tuple[int, int, int] = (128, 128, 3)) -> models.Sequential:
        return models.Sequential([
            layers.Input(shape=input_shape),
            layers.Conv2D(32,  3, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.MaxPool2D(),
            layers.Conv2D(64,  3, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.MaxPool2D(),
            layers.Conv2D(128, 3, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.MaxPool2D(),
            layers.Flatten(),
            layers.Dense(256, activation="relu", kernel_regularizer=regularizers.l2(1e-4)),
            layers.Dropout(0.6),
            layers.Dense(num_classes, activation="softmax"),
        ])


class MobileNetModelBuilder:
    """Builds a MobileNetV2 transfer-learning model."""

    @staticmethod
    def build(num_classes: int):
        base = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
        base.trainable = False
        x = layers.GlobalAveragePooling2D()(base.output)
        x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
        x = layers.Dropout(0.5)(x)
        outputs = layers.Dense(num_classes, activation="softmax")(x)
        return models.Model(inputs=base.input, outputs=outputs), base


# =============================================================================
# Trainer
# =============================================================================

class ModelTrainer:
    """
    Trains a model, optionally fine-tunes the MobileNet base, saves artefacts.

    Parameters
    ----------
    name : str
        Used for file naming (model .h5, labels .json, confusion matrix .png).
    generators : AugmentedDataGenerators
    model : Keras model
    base_model : optional Keras model
        The frozen base; required when fine_tune=True.
    fine_tune : bool
    fine_tune_ratio : int
        Percentage of base layers to unfreeze for fine-tuning.
    """

    def __init__(
        self,
        name: str,
        generators: AugmentedDataGenerators,
        model,
        base_model=None,
        fine_tune: bool = False,
        fine_tune_ratio: int = 25,
    ):
        self.name = name
        self.gen = generators
        self.model = model
        self.base_model = base_model
        self.fine_tune = fine_tune
        self.fine_tune_ratio = fine_tune_ratio

    def _callbacks(self):
        return [
            EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
            ModelCheckpoint(
                str(MODELS_DIR / f"{self.name}_best.h5"),
                monitor="val_loss", save_best_only=True,
            ),
        ]

    def train(self):
        print(f"\n🚀 Training: {self.name}")
        self.model.compile(
            optimizer="adam",
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(
            self.gen.train,
            validation_data=self.gen.val,
            epochs=EPOCHS,
            callbacks=self._callbacks(),
        )

        if self.fine_tune and self.base_model is not None:
            self._fine_tune()

        self._save()
        self._plot_confusion_matrix()

    def _fine_tune(self):
        print("🔧 Fine-tuning MobileNet…")
        n_layers = int(len(self.base_model.layers) * self.fine_tune_ratio / 100)
        for layer in self.base_model.layers[-n_layers:]:
            layer.trainable = True

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        self.model.fit(
            self.gen.train,
            validation_data=self.gen.val,
            epochs=FINE_TUNE_EPOCHS,
            callbacks=self._callbacks(),
        )

    def _save(self):
        model_path = MODELS_DIR / f"{self.name}.h5"
        self.model.save(str(model_path))
        print(f"✅ Saved model: {model_path}")

        labels_path = MODELS_DIR / f"class_labels_{self.name}.json"
        with open(labels_path, "w") as f:
            json.dump(self.gen.class_indices, f)
        print(f"✅ Saved labels: {labels_path}")

    def _plot_confusion_matrix(self):
        preds  = self.model.predict(self.gen.val, verbose=1)
        y_pred = np.argmax(preds, axis=1)
        y_true = self.gen.val.classes
        labels = self.gen.labels

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues")
        plt.title(f"Confusion Matrix – {self.name}")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        out_path = LOGS_DIR / f"{self.name}_confusion_matrix.png"
        plt.savefig(out_path)
        plt.close()
        print(f"📊 Saved confusion matrix: {out_path}")


# =============================================================================
# Entry point
# =============================================================================

def run_all_pipelines():
    # 1. CNN – raw images
    gen = AugmentedDataGenerators(DATA_RAW_TRAIN, DATA_RAW_VAL, IMG_SIZE_CNN)
    ModelTrainer(
        name="cnn_raw_v2",
        generators=gen,
        model=CNNModelBuilder.build(gen.num_classes),
    ).train()

    # 2. CNN – MediaPipe-cropped images
    gen = AugmentedDataGenerators(DATA_MP_TRAIN, DATA_MP_VAL, IMG_SIZE_CNN)
    ModelTrainer(
        name="cnn_mp_v2",
        generators=gen,
        model=CNNModelBuilder.build(gen.num_classes),
    ).train()

    # 3. MobileNet TL – 10 % fine-tune
    gen = AugmentedDataGenerators(DATA_MP_TRAIN, DATA_MP_VAL, IMG_SIZE_MOBILENET, use_mobilenet_preprocess=True)
    model, base = MobileNetModelBuilder.build(gen.num_classes)
    ModelTrainer(
        name="mobilenet_mp_10%_v2",
        generators=gen,
        model=model,
        base_model=base,
        fine_tune=True,
        fine_tune_ratio=10,
    ).train()

    # 4. MobileNet TL – 25 % fine-tune
    gen = AugmentedDataGenerators(DATA_MP_TRAIN, DATA_MP_VAL, IMG_SIZE_MOBILENET, use_mobilenet_preprocess=True)
    model, base = MobileNetModelBuilder.build(gen.num_classes)
    ModelTrainer(
        name="mobilenet_mp_25%_v2",
        generators=gen,
        model=model,
        base_model=base,
        fine_tune=True,
        fine_tune_ratio=25,
    ).train()


if __name__ == "__main__":
    run_all_pipelines()
