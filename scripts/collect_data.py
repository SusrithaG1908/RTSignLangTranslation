import cv2
import os

label = "A"  # Change this for each sign
save_dir = f"../data/raw/{label}"
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

while True:
    ret, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2.imshow("Capture - Press 's' to save, 'q' to quit", frame)

    key = cv2.waitKey(1)
    if key == ord('s'):
        cv2.imwrite(f"{save_dir}/{count}.jpg", frame)
        print(f"Saved {count}")
        count += 1

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
