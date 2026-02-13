import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
import os

# Config
IMG_SIZE = (128, 128)
MODEL_PATH = "../models/sign_model.h5"
LABELS_PATH = "../models/class_labels.json"
IMAGE_DIR = "../test_images"
DEFAULT_IMAGE = "sample.jpg"

IMAGE_DIR = "../cropped_images"
DEFAULT_IMAGE = "cropped_sample.jpg"

# 👉 Prompt user for image name
img_name = input(f"Enter image name (default: {DEFAULT_IMAGE}): ").strip()

if img_name == "":
    img_name = DEFAULT_IMAGE

IMAGE_PATH = os.path.join(IMAGE_DIR, img_name)

# Load model
model = load_model(MODEL_PATH)

# Load class labels
with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

class_labels = {int(v): k for k, v in class_indices.items()}

# Read image
img_bgr = cv2.imread(IMAGE_PATH)

if img_bgr is None:
    raise FileNotFoundError(f"Image not found at path: {IMAGE_PATH}")

# Convert BGR to RGB (if trained on RGB)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

# Resize
img_resized = cv2.resize(img_rgb, IMG_SIZE)

# Visualize the input image
roi_bgr = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
cv2.imshow("Model Input (128x128)", roi_bgr)
cv2.waitKey(0)

# Normalize
img_norm = img_resized / 255.0
img_input = np.expand_dims(img_norm, axis=0)

# Predict
preds = model.predict(img_input, verbose=0)
pred_class = int(np.argmax(preds))
confidence = float(np.max(preds))

label = class_labels[pred_class]

# Print result
print(f"Predicted Sign: {label}")
print(f"Confidence: {confidence:.4f}")

# Show image with prediction (optional)
cv2.putText(img_bgr, f"{label} ({confidence:.2f})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1,
            (0, 255, 0), 2)

cv2.imshow("Prediction", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()