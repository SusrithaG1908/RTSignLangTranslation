import cv2
import numpy as np
import json
from tensorflow.keras.models import load_model
import os
import time

save_dir = "../captured_frames"
os.makedirs(save_dir, exist_ok=True)

IMG_SIZE = (128, 128)

model = load_model("../models/sign_model.h5")

with open("../models/class_labels.json", "r") as f:
    class_indices = json.load(f)

# Map class index to label
# class_labels = {v: k for k, v in model.class_indices.items()} if hasattr(model, "class_indices") else None
class_labels = {int(v): k for k, v in class_indices.items()}

# If class_indices not saved in model, define manually:
# class_labels = {0: 'A', 1: 'B', 2: 'C', ...}

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)

    x1, y1, x2, y2 = 50, 50, 350, 350
    roi_wlabel = frame[0:y2, x1:x2]
    roi = frame[y1:y2, x1:x2]

    # ✅ Convert BGR → RGB (CRITICAL FIX)
    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

    img = cv2.resize(roi_rgb, IMG_SIZE)
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))

    label = class_labels[pred_class]

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

    # inside while loop, after reading frame
    if key == ord('s'):
        filename = f"frame_{int(time.time())}.jpg"
        #cv2.imwrite(os.path.join(save_dir, filename), frame)
        cv2.imwrite(os.path.join(save_dir, filename), roi_wlabel)
        filename = f"frame_{int(time.time())}_onlyROI.jpg"
        cv2.imwrite(os.path.join(save_dir, filename), roi)
        print(f"Saved full frame: {filename}")

cap.release()
cv2.destroyAllWindows()

