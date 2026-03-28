import os
import time
import json
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# -------- Config --------
IMG_SIZE_CNN = (128, 128)
IMG_SIZE_MOBILENET = (224, 224)

# ---- Resolve project root safely ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_RAW_TEST = PROJECT_ROOT / "data" / "test"
DATA_MP_TEST  = PROJECT_ROOT / "data_mp" / "test"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "benchmark_logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

GLOBAL_SUMMARY_CSV = LOGS_DIR / "benchmark_summary.csv"

# ---- MediaPipe setup ----
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.5)

def crop_with_mediapipe(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    h, w, _ = img_rgb.shape

    if results.multi_hand_landmarks:
        xs, ys = [], []
        for lm in results.multi_hand_landmarks[0].landmark:
            xs.append(int(lm.x * w))
            ys.append(int(lm.y * h))
        pad = 30
        x1, y1 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
        x2, y2 = min(max(xs) + pad, w), min(max(ys) + pad, h)
        crop_rgb = img_rgb[y1:y2, x1:x2]
        used_mp = True
    else:
        crop_rgb = img_rgb[int(0.2*h):int(0.9*h), int(0.2*w):int(0.9*w)]
        used_mp = False

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    return crop_bgr, used_mp

# ---- Preprocessors ----
def preprocess_raw(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE_CNN)
    return np.expand_dims(img_resized / 255.0, axis=0), {"used_mediapipe": False}

def preprocess_mediapipe_cnn(img_bgr):
    cropped, used_mp = crop_with_mediapipe(img_bgr)
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE_CNN)
    return np.expand_dims(img_resized / 255.0, axis=0), {"used_mediapipe": used_mp}

def preprocess_mediapipe_mobilenet(img_bgr):
    cropped, used_mp = crop_with_mediapipe(img_bgr)
    img_rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, IMG_SIZE_MOBILENET)
    img_pre = preprocess_input(img_resized.astype(np.float32))
    return np.expand_dims(img_pre, axis=0), {"used_mediapipe": used_mp}

# ---- Pipelines ----
PIPELINES = {
    # Existing
    #"CNN_Raw": {
    #    "model_path": MODELS_DIR / "cnn_raw.h5",
    #    "labels_path": MODELS_DIR / "class_labels_cnn_raw.json",
    #    "test_dir": DATA_RAW_TEST,
    #    "preprocess_fn": preprocess_raw
    #},
    #"CNN_MediaPipeCrop": {
    #    "model_path": MODELS_DIR / "cnn_mp.h5",
    #    "labels_path": MODELS_DIR / "class_labels_cnn_mp.json",
    #    "test_dir": DATA_MP_TEST,
    #    "preprocess_fn": preprocess_mediapipe_cnn
    #},
    #"MobileNet_TL_10%": {
    #    "model_path": MODELS_DIR / "mobilenet_mp_10%.h5",
    #    "labels_path": MODELS_DIR / "class_labels_mobilenet_mp_10%.json",
    #    "test_dir": DATA_MP_TEST,
    #    "preprocess_fn": preprocess_mediapipe_mobilenet
    #},
    #"MobileNet_TL_25%": {
    #    "model_path": MODELS_DIR / "mobilenet_mp_25%.h5",
    #    "labels_path": MODELS_DIR / "class_labels_mobilenet_mp_25%.json",
    #    "test_dir": DATA_MP_TEST,
    #    "preprocess_fn": preprocess_mediapipe_mobilenet
    #},

    # New v2 (robust)
    "CNN_Raw_v2": {
        "model_path": MODELS_DIR / "cnn_raw_v2.h5",
        "labels_path": MODELS_DIR / "class_labels_cnn_raw_v2.json",
        "test_dir": DATA_RAW_TEST,
        "preprocess_fn": preprocess_raw
    },
    "CNN_MediaPipeCrop_v2": {
        "model_path": MODELS_DIR / "cnn_mp_v2.h5",
        "labels_path": MODELS_DIR / "class_labels_cnn_mp_v2.json",
        "test_dir": DATA_MP_TEST,
        "preprocess_fn": preprocess_mediapipe_cnn
    },
    "MobileNet_TL_10%_v2": {
        "model_path": MODELS_DIR / "mobilenet_mp_10%_v2.h5",
        "labels_path": MODELS_DIR / "class_labels_mobilenet_mp_10%_v2.json",
        "test_dir": DATA_MP_TEST,
        "preprocess_fn": preprocess_mediapipe_mobilenet
    },
    "MobileNet_TL_25%_v2": {
        "model_path": MODELS_DIR / "mobilenet_mp_25%_v2.h5",
        "labels_path": MODELS_DIR / "class_labels_mobilenet_mp_25%_v2.json",
        "test_dir": DATA_MP_TEST,
        "preprocess_fn": preprocess_mediapipe_mobilenet
    }
}

# ---- Load models + labels ----
loaded_models = {}
loaded_labels = {}

for name, cfg in PIPELINES.items():
    print(f"📦 Loading {name}")
    loaded_models[name] = load_model(str(cfg["model_path"]))
    with open(cfg["labels_path"], "r") as f:
        ci = json.load(f)
    loaded_labels[name] = {int(v): k for k, v in ci.items()}

# ---- Benchmark ----
global_summary = []

for name, cfg in PIPELINES.items():
    model = loaded_models[name]
    preprocess_fn = cfg["preprocess_fn"]
    test_root = cfg["test_dir"]
    class_labels = loaded_labels[name]
    labels_sorted = [class_labels[i] for i in range(len(class_labels))]

    y_true, y_pred = [], []
    correct, total = 0, 0
    total_latency = 0.0
    confidences = []
    per_image_logs = []

    print(f"\n🚀 Benchmarking on TEST set: {name}")

    for label in os.listdir(test_root):
        label_dir = test_root / label
        if not label_dir.is_dir():
            continue

        for fname in os.listdir(label_dir):
            img_path = label_dir / fname
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue

            img_input, meta = preprocess_fn(img_bgr)

            start = time.perf_counter()
            preds = model.predict(img_input, verbose=0)
            latency_ms = (time.perf_counter() - start) * 1000

            pred_class = int(np.argmax(preds))
            confidence = float(np.max(preds))
            pred_label = class_labels[pred_class]

            total += 1
            total_latency += latency_ms
            confidences.append(confidence)

            is_correct = (pred_label == label)
            if is_correct:
                correct += 1

            y_true.append(label)
            y_pred.append(pred_label)

            per_image_logs.append({
                "model": name,
                "image_path": str(img_path.relative_to(PROJECT_ROOT)),
                "true_label": label,
                "pred_label": pred_label,
                "confidence": round(confidence, 4),
                "latency_ms": round(latency_ms, 2),
                "correct": is_correct,
                **meta
            })

    # ---- Save per-image logs ----
    per_image_df = pd.DataFrame(per_image_logs)
    per_image_csv = LOGS_DIR / f"{name}_per_image_results.csv"
    per_image_df.to_csv(per_image_csv, index=False)

    # ---- Confusion Matrix ----
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, xticklabels=labels_sorted, yticklabels=labels_sorted, cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = LOGS_DIR / f"{name}_confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    print(f"📊 Confusion matrix saved: {cm_path}")

    # ---- Global summary ----
    acc = correct / total if total else 0
    avg_latency = total_latency / total if total else 0
    avg_conf = sum(confidences) / len(confidences) if confidences else 0

    global_summary.append({
        "Model": name,
        "Accuracy_%": round(acc * 100, 2),
        "Avg_Confidence": round(avg_conf, 3),
        "Avg_Latency_ms": round(avg_latency, 2),
        "Samples": total
    })

# ---- Save global summary ----
summary_df = pd.DataFrame(global_summary)
summary_df.to_csv(GLOBAL_SUMMARY_CSV, index=False)

print("\n📊 Global Benchmark Summary (TEST set)\n")
print(summary_df)

# ---- Plot global summary ----
plt.figure(figsize=(10, 5))
plt.bar(summary_df["Model"], summary_df["Accuracy_%"])
plt.ylabel("Accuracy (%)")
plt.title("Model Accuracy Comparison (TEST Set)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(LOGS_DIR / "benchmark_accuracy_comparison.png")
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(summary_df["Model"], summary_df["Avg_Latency_ms"])
plt.ylabel("Avg Latency (ms)")
plt.title("Model Latency Comparison (TEST Set)")
plt.xticks(rotation=20)
plt.tight_layout()
plt.savefig(LOGS_DIR / "benchmark_latency_comparison.png")
plt.show()

print(f"\n✅ Global summary saved: {GLOBAL_SUMMARY_CSV}")
print(f"📈 Plots saved in: {LOGS_DIR}")