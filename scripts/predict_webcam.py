# import cv2
# import numpy as np
# import json
# import time
# from pathlib import Path
# from tensorflow.keras.models import load_model
# from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# import mediapipe as mp
# # import mediapipe.python.solutions as mp
# from collections import deque
# import pyttsx3
# import threading

# # -------- Resolve project root safely --------
# PROJECT_ROOT = Path(__file__).resolve().parents[1]

# SAVE_DIR = PROJECT_ROOT / "captured_frames"
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# IMG_SIZE_CNN = (128, 128)
# IMG_SIZE_MOBILENET = (224, 224)

# MODELS_DIR = PROJECT_ROOT / "models"

# # -------- Pipeline registry --------
# PIPELINES = {
#     "1": {
#         "name": "CNN_Raw_v2",
#         "model_path": MODELS_DIR / "cnn_raw_v2.h5",
#         "labels_path": MODELS_DIR / "class_labels_cnn_raw_v2.json",
#         "img_size": IMG_SIZE_CNN,
#         "use_mediapipe": False,
#         "is_mobilenet": False
#     },
#     "2": {
#         "name": "CNN_MediaPipeCrop_v2",
#         "model_path": MODELS_DIR / "cnn_mp_v2.h5",
#         "labels_path": MODELS_DIR / "class_labels_cnn_mp_v2.json",
#         "img_size": IMG_SIZE_CNN,
#         "use_mediapipe": True,
#         "is_mobilenet": False
#     },
#     "3": {
#         "name": "MobileNet_TL_10%_v2",
#         "model_path": MODELS_DIR / "mobilenet_mp_10%_v2.h5",
#         "labels_path": MODELS_DIR / "class_labels_mobilenet_mp_10%_v2.json",
#         "img_size": IMG_SIZE_MOBILENET,
#         "use_mediapipe": True,
#         "is_mobilenet": True
#     },
#     "4": {
#         "name": "MobileNet_TL_25%_v2",
#         "model_path": MODELS_DIR / "mobilenet_mp_25%_v2_best.h5",
#         "labels_path": MODELS_DIR / "class_labels_mobilenet_mp_25%_v2.json",
#         "img_size": IMG_SIZE_MOBILENET,
#         "use_mediapipe": True,
#         "is_mobilenet": True
#     }
# }

# # -------- Prompt user --------
# print("\nSelect pipeline:")
# print("1️⃣  CNN (Raw ROI)")
# print("2️⃣  CNN + MediaPipe Crop")
# print("3️⃣  MobileNet(10%) + MediaPipe Crop")
# print("4️⃣  MobileNet(25%) + MediaPipe Crop")

# choice = input("Enter choice (1-4): ").strip()
# if choice not in PIPELINES:
#     raise ValueError("❌ Invalid pipeline selection")

# cfg = PIPELINES[choice]
# print(f"\n🚀 Using pipeline: {cfg['name']}")

# # -------- Load model + labels --------
# # model = load_model(str(cfg["model_path"]))
# model = load_model(str(cfg["model_path"]), compile=False)

# with open(cfg["labels_path"], "r") as f:
#     class_indices = json.load(f)
# class_labels = {int(v): k for k, v in class_indices.items()}

# # -------- MediaPipe setup --------
# hands = None
# if cfg["use_mediapipe"]:
#     mp_hands = mp.solutions.hands
#     hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# def crop_with_mediapipe(img_bgr):
#     img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
#     results = hands.process(img_rgb)
#     h, w, _ = img_rgb.shape

#     if results.multi_hand_landmarks:
#         xs, ys = [], []
#         for lm in results.multi_hand_landmarks[0].landmark:
#             xs.append(int(lm.x * w))
#             ys.append(int(lm.y * h))
#         pad = 30
#         x1 = max(min(xs) - pad, 0)
#         y1 = max(min(ys) - pad, 0)
#         x2 = min(max(xs) + pad, w)
#         y2 = min(max(ys) + pad, h)
#         crop_rgb = img_rgb[y1:y2, x1:x2]
#         used_mp = True
#         mp_box = (x1, y1, x2, y2)
#     else:
#         crop_rgb = img_rgb[int(0.2*h):int(0.9*h), int(0.2*w):int(0.9*w)]
#         used_mp = False
#         mp_box = None

#     crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
#     return crop_bgr, used_mp, mp_box

# # -------- TTS setup (FIXED) --------
# engine = pyttsx3.init()
# engine.setProperty('rate', 160)
# engine.setProperty('volume', 1.0)

# tts_lock = threading.Lock()

# def speak_async(text):
#     def _speak():
#         with tts_lock:  # ✅ prevents concurrent run loops
#             engine.say(text)
#             engine.runAndWait()
#     threading.Thread(target=_speak, daemon=True).start()

# last_spoken_time = 0
# SPEAK_COOLDOWN = 0.6

# cap = cv2.VideoCapture(0)

# # -------- Word builder state --------
# last_char = None
# stable_buffer = deque(maxlen=6)
# current_word = ""

# print("\n📝 Live recognized word: ")

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         print("❌ Failed to read from webcam")
#         break

#     frame = cv2.flip(frame, 1)

#     x1, y1, x2, y2 = 50, 50, 350, 350
#     roi_fallback = frame[y1:y2, x1:x2]

#     if cfg["use_mediapipe"]:
#         roi_used, used_mp, mp_box = crop_with_mediapipe(frame)
#         if not used_mp:
#             roi_used = roi_fallback
#     else:
#         roi_used = roi_fallback
#         mp_box = None

#     roi_rgb = cv2.cvtColor(roi_used, cv2.COLOR_BGR2RGB)
#     img = cv2.resize(roi_rgb, cfg["img_size"])

#     if cfg["is_mobilenet"]:
#         img = preprocess_input(img.astype(np.float32))
#     else:
#         img = img / 255.0

#     img = np.expand_dims(img, axis=0)

#     preds = model.predict(img, verbose=0)[0]
#     pred_class = int(np.argmax(preds))
#     confidence = float(np.max(preds))
#     label = class_labels.get(pred_class, "Unknown")

#     if confidence >= 0.8:
#         stable_buffer.append(label)
#     else:
#         stable_buffer.clear()

#     if len(stable_buffer) == stable_buffer.maxlen:
#         stable_label = max(set(stable_buffer), key=stable_buffer.count)

#         if stable_label != last_char:
#             last_char = stable_label
#             speak_text = stable_label

#             if stable_label.lower() == "space":
#                 stable_label = " "
#                 speak_text = "space"
#             if stable_label.lower() == "nothing":
#                 stable_label = ""
#                 speak_text = ""

#             current_word += stable_label
#             print(f"\r📝 Word: {current_word}", end="", flush=True)

#             now = time.time()
#             if now - last_spoken_time > SPEAK_COOLDOWN:
#                 speak_async(speak_text)
#                 last_spoken_time = now

#     if mp_box:
#         cv2.rectangle(frame, (mp_box[0], mp_box[1]), (mp_box[2], mp_box[3]), (255, 0, 0), 2)
#     else:
#         cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

#     tag = f"{label} ({confidence:.2f}) | {cfg['name']}"
#     cv2.putText(frame, tag, (20, 40),
#                 cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

#     cv2.imshow("Sign Language Translator (Word Builder + Audio)", frame)

#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('q'):
#         print("\n👋 Exiting...")
#         break
#     if key == ord('r'):
#         current_word = ""
#         last_char = None
#         stable_buffer.clear()
#         print("\n🔄 Word reset")

# cap.release()
# cv2.destroyAllWindows()





import cv2
import numpy as np
import json
import time
from pathlib import Path
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import mediapipe as mp
from collections import deque
import pyttsx3
import threading

# -------- Paths --------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = PROJECT_ROOT / "models"

# -------- Model config (ONLY ONE) --------
MODEL_PATH = MODELS_DIR / "mobilenet_mp_25%_v2_best.h5"
LABELS_PATH = MODELS_DIR / "class_labels_mobilenet_mp_25%_v2.json"
IMG_SIZE = (224, 224)

print("\n🚀 Using MobileNet 25% model")

# -------- Load model --------
model = load_model(str(MODEL_PATH), compile=False)

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)
class_labels = {int(v): k for k, v in class_indices.items()}

# -------- MediaPipe --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5)

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
        x1 = max(min(xs) - pad, 0)
        y1 = max(min(ys) - pad, 0)
        x2 = min(max(xs) + pad, w)
        y2 = min(max(ys) + pad, h)
        crop_rgb = img_rgb[y1:y2, x1:x2]
        mp_box = (x1, y1, x2, y2)
    else:
        crop_rgb = img_rgb[int(0.2*h):int(0.9*h), int(0.2*w):int(0.9*w)]
        mp_box = None

    return cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR), mp_box

# -------- TTS --------
engine = pyttsx3.init()
tts_lock = threading.Lock()

def speak_async(text):
    def _speak():
        with tts_lock:
            engine.say(text)
            engine.runAndWait()
    threading.Thread(target=_speak, daemon=True).start()

# -------- Webcam --------
cap = cv2.VideoCapture(0)

stable_buffer = deque(maxlen=6)
last_char = None
current_word = ""

print("\n📝 Live recognized word:")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    roi, mp_box = crop_with_mediapipe(frame)

    roi_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    img = cv2.resize(roi_rgb, IMG_SIZE)
    img = preprocess_input(img.astype(np.float32))
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img, verbose=0)[0]
    pred_class = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = class_labels.get(pred_class, "Unknown")

    if confidence >= 0.8:
        stable_buffer.append(label)
    else:
        stable_buffer.clear()

    if len(stable_buffer) == stable_buffer.maxlen:
        stable_label = max(set(stable_buffer), key=stable_buffer.count)

        if stable_label != last_char:
            last_char = stable_label

            if stable_label.lower() == "space":
                stable_label = " "
            if stable_label.lower() == "nothing":
                stable_label = ""

            current_word += stable_label
            print(f"\r📝 Word: {current_word}", end="", flush=True)
            speak_async(stable_label)

    if mp_box:
        cv2.rectangle(frame, (mp_box[0], mp_box[1]),
                      (mp_box[2], mp_box[3]), (255, 0, 0), 2)

    cv2.putText(frame, f"{label} ({confidence:.2f})",
                (20, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (0, 255, 0), 2)

    cv2.imshow("Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()