import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
import time
from pathlib import Path

# -------- Resolve project root safely --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

SAVE_DIR = PROJECT_ROOT / "captured_frames"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

IMG_SIZE = (128, 128)

MODEL_PATH = PROJECT_ROOT / "models" / "sign_model.h5"
LABELS_PATH = PROJECT_ROOT / "models" / "class_labels.json"

model = load_model(str(MODEL_PATH))

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

class_labels = {int(v): k for k, v in class_indices.items()}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to read from webcam")
        break

    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 50, 50, 350, 350
    roi_wlabel = frame[0:y2, x1:x2]   # ROI including label area
    roi = frame[y1:y2, x1:x2]         # pure ROI

    # Convert BGR → RGB
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    img = cv2.resize(roi_rgb, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    label = class_labels.get(pred_class, "Unknown")

    if confidence < 0.6:
        label = "Uncertain"

    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('q'):
        break

    if key == ord('s'):
        ts = int(time.time())
        full_path = SAVE_DIR / f"frame_{ts}_with_label.jpg"
        roi_path = SAVE_DIR / f"frame_{ts}_onlyROI.jpg"

        cv2.imwrite(str(full_path), roi_wlabel)
        cv2.imwrite(str(roi_path), roi)

        print(f"✅ Saved:")
        print(f"   - With label: {full_path}")
        print(f"   - ROI only:   {roi_path}")

cap.release()
cv2.destroyAllWindows()
