import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import numpy as np

# ---- Config ----
IMG_SIZE_CNN = (128, 128)
IMG_SIZE_MOBILENET = (224, 224)
BATCH_SIZE = 32
EPOCHS = 30
FINE_TUNE_EPOCHS = 10

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_TRAIN = PROJECT_ROOT / "data" / "train"
DATA_RAW_VAL   = PROJECT_ROOT / "data" / "val"
DATA_MP_TRAIN  = PROJECT_ROOT / "data_mp" / "train"
DATA_MP_VAL    = PROJECT_ROOT / "data_mp" / "val"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "train_logs"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# NEW v2 (ROBUST)
# =========================
def get_generators_v2(data_dir, val_dir, img_size, use_mobilenet_preprocess=False):
    if use_mobilenet_preprocess:
        train_datagen = ImageDataGenerator(
            preprocessing_function=preprocess_input,
            brightness_range=(0.7, 1.3),
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            shear_range=0.1
        )
        val_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)
    else:
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            brightness_range=(0.7, 1.3),
            zoom_range=0.2,
            width_shift_range=0.1,
            height_shift_range=0.1,
            rotation_range=10,
            shear_range=0.1
        )
        val_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(data_dir), target_size=img_size, batch_size=BATCH_SIZE, class_mode='categorical'
    )
    val_gen = val_datagen.flow_from_directory(
        str(val_dir), target_size=img_size, batch_size=BATCH_SIZE, class_mode='categorical', shuffle=False
    )
    return train_gen, val_gen

def build_cnn_v2(num_classes, input_shape=(128,128,3)):
    return models.Sequential([
        layers.Input(shape=input_shape),
        layers.Conv2D(32, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPool2D(),
        layers.Conv2D(64, 3, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPool2D(),
        layers.Conv2D(128,3, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.MaxPool2D(),
        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.Dropout(0.6),
        layers.Dense(num_classes, activation='softmax')
    ])

def build_mobilenet_v2(num_classes):
    base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights="imagenet")
    base.trainable = False
    x = base.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation="relu", kernel_regularizer=regularizers.l2(1e-4))(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inputs=base.input, outputs=outputs), base

# =========================
# HELPERS (UNCHANGED)
# =========================
def evaluate_and_plot_cm(model, val_gen, labels, name):
    preds = model.predict(val_gen, verbose=1)
    y_pred = np.argmax(preds, axis=1)
    y_true = val_gen.classes

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    out_path = LOGS_DIR / f"{name}_confusion_matrix.png"
    plt.savefig(out_path)
    plt.close()
    print(f"📊 Saved confusion matrix: {out_path}")

# =========================
# TRAIN PIPELINE (EXTENDED)
# =========================
def train_pipeline(name, train_dir, val_dir, build_fn, img_size,
                   use_mobilenet_preprocess=False, fine_tune=False, fine_tune_ratio=25,
                   use_v2=False):

    print(f"\n🚀 Training pipeline: {name}")

    train_gen, val_gen = get_generators_v2(train_dir, val_dir, img_size, use_mobilenet_preprocess)

    num_classes = train_gen.num_classes
    labels = list(train_gen.class_indices.keys())

    if name.startswith("mobilenet"):
        model, base_model = build_fn(num_classes)
    else:
        model = build_fn(num_classes)

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ModelCheckpoint(str(MODELS_DIR / f"{name}_best.h5"), monitor="val_loss", save_best_only=True)
    ]

    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS, callbacks=callbacks)

    if name.startswith("mobilenet") and fine_tune:
        print("🔧 Fine-tuning MobileNet...")
        N = int(len(base_model.layers) * fine_tune_ratio / 100)
        for layer in base_model.layers[-N:]:
            layer.trainable = True

        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-5),
            loss="categorical_crossentropy",
            metrics=["accuracy"]
        )
        model.fit(train_gen, validation_data=val_gen, epochs=FINE_TUNE_EPOCHS, callbacks=callbacks)

    final_model_path = MODELS_DIR / f"{name}.h5"
    model.save(str(final_model_path))
    print(f"✅ Saved final model: {final_model_path}")

    with open(MODELS_DIR / f"class_labels_{name}.json", "w") as f:
        json.dump(train_gen.class_indices, f)

    evaluate_and_plot_cm(model, val_gen, labels, name)

# =========================
# MAIN
# =========================
if __name__ == "__main__":
    # New robust v2 pipelines
    train_pipeline("cnn_raw_v2", DATA_RAW_TRAIN, DATA_RAW_VAL, build_cnn_v2,
                IMG_SIZE_CNN, use_v2=True)

    train_pipeline("cnn_mp_v2", DATA_MP_TRAIN, DATA_MP_VAL, build_cnn_v2,
                IMG_SIZE_CNN, use_v2=True)

    train_pipeline("mobilenet_mp_10%_v2", DATA_MP_TRAIN, DATA_MP_VAL, build_mobilenet_v2,
                IMG_SIZE_MOBILENET, use_mobilenet_preprocess=True, fine_tune=True, fine_tune_ratio=10, use_v2=True)

    train_pipeline("mobilenet_mp_25%_v2", DATA_MP_TRAIN, DATA_MP_VAL, build_mobilenet_v2,
                IMG_SIZE_MOBILENET, use_mobilenet_preprocess=True, fine_tune=True, fine_tune_ratio=25, use_v2=True)