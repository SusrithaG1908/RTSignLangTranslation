import cv2
import os
from pathlib import Path

label = "A"  # Change this for each sign

# ---- Resolve project root safely ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw" / label
DATA_DIR.mkdir(parents=True, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("❌ Failed to capture frame from webcam")
        break

    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture - Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        img_path = DATA_DIR / f"{count}.jpg"
        cv2.imwrite(str(img_path), frame)
        print(f"✅ Saved {img_path}")
        count += 1

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
