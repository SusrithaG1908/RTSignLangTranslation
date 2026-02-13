import os
import cv2
import mediapipe as mp
from tqdm import tqdm

# -------- Config --------
SRC_ROOT = "data"        # existing organized dataset
DST_ROOT = "data_mp"    # new cropped dataset
IMG_SIZE = (128, 128)

# -------- MediaPipe Setup --------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.2  # lower for tricky signs (R, S, T)
)

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
    else:
        # 🔁 Fallback: center crop (keeps pipeline robust)
        crop_rgb = img_rgb[int(0.2*h):int(0.9*h), int(0.2*w):int(0.9*w)]

    crop_bgr = cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2BGR)
    crop_resized = cv2.resize(crop_bgr, IMG_SIZE)
    return crop_resized

def process_split(split_name):
    src_split = os.path.join(SRC_ROOT, split_name)
    dst_split = os.path.join(DST_ROOT, split_name)
    os.makedirs(dst_split, exist_ok=True)

    for label in os.listdir(src_split):
        src_label_dir = os.path.join(src_split, label)
        dst_label_dir = os.path.join(dst_split, label)
        os.makedirs(dst_label_dir, exist_ok=True)

        if not os.path.isdir(src_label_dir):
            continue

        for fname in tqdm(os.listdir(src_label_dir), desc=f"{split_name}/{label}"):
            src_path = os.path.join(src_label_dir, fname)
            dst_path = os.path.join(dst_label_dir, fname)

            img = cv2.imread(src_path)
            if img is None:
                continue

            cropped = crop_with_mediapipe(img)
            cv2.imwrite(dst_path, cropped)

def main():
    for split in ["train", "val", "test"]:
        print(f"\n🔄 Processing split: {split}")
        process_split(split)

    print("\n🎉 Done! MediaPipe-cropped dataset saved to:", DST_ROOT)

if __name__ == "__main__":
    main()
