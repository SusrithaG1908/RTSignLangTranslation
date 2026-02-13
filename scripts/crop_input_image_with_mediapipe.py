import cv2
import mediapipe as mp
import os

IMAGE_DIR = "../test_images"
OUT_DIR = "../cropped_images"
DEFAULT_IMAGE = "sample.jpg"

os.makedirs(OUT_DIR, exist_ok=True)

img_name = input(f"Enter image name (default: {DEFAULT_IMAGE}): ").strip() or DEFAULT_IMAGE
IMAGE_PATH = os.path.join(IMAGE_DIR, img_name)

img_bgr = cv2.imread(IMAGE_PATH)
if img_bgr is None:
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.2
)

results = hands.process(img_rgb)

used_mediapipe = False

if results.multi_hand_landmarks:
    h, w, _ = img_rgb.shape
    xs, ys = [], []
    for lm in results.multi_hand_landmarks[0].landmark:
        xs.append(int(lm.x * w))
        ys.append(int(lm.y * h))

    pad = 30
    x1, y1 = max(min(xs) - pad, 0), max(min(ys) - pad, 0)
    x2, y2 = min(max(xs) + pad, w), min(max(ys) + pad, h)
    hand_rgb = img_rgb[y1:y2, x1:x2]
    used_mediapipe = True
    print("✅ MediaPipe detected hand")
else:
    print("⚠️ MediaPipe failed. Using fallback crop.")
    h, w, _ = img_rgb.shape
    hand_rgb = img_rgb[int(0.2*h):int(0.9*h), int(0.2*w):int(0.9*w)]

hand_bgr = cv2.cvtColor(hand_rgb, cv2.COLOR_RGB2BGR)

out_path = os.path.join(OUT_DIR, f"cropped_{img_name}")
cv2.imwrite(out_path, hand_bgr)

# Draw bounding box only if MediaPipe succeeded
if used_mediapipe:
    cv2.rectangle(img_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)

cv2.imshow("Hand Crop Saved", img_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(f"✅ Saved cropped hand to: {out_path}")
