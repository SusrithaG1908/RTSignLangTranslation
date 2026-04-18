"""
predict_webcam.py
Real-time ASL sign-language translation via webcam.

Uses:  HandCropper · ImagePreprocessor · SignClassifier
       WordBuilder  · TTSSpeaker        · PipelineRegistry
"""

import sys
import cv2
import numpy as np
from pathlib import Path

from src.core.pipeline_config import PipelineRegistry
from src.core.hand_cropper import HandCropper
from src.core.preprocessor import ImagePreprocessor
from src.core.classifier import SignClassifier
from src.core.word_builder import WordBuilder
from src.core.tts_speaker import TTSSpeaker

# ---- Project layout ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAVE_DIR = PROJECT_ROOT / "captured_frames"
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# Fallback ROI when MediaPipe is disabled or fails
FALLBACK_ROI = (50, 350, 50, 350)   # y1, y2, x1, x2


def main():
    registry = PipelineRegistry(PROJECT_ROOT)
    registry.print_menu()

    choice = input("Enter choice (1-4): ").strip()
    cfg = registry.get_by_number(choice)
    print(f"\n🚀 Using pipeline: {cfg.name}")

    # ---- Build pipeline components ----
    classifier   = SignClassifier(cfg.model_path, cfg.labels_path)
    preprocessor = ImagePreprocessor(img_size=cfg.img_size, mode=cfg.preprocess_mode)
    word_builder = WordBuilder(buffer_size=6, min_confidence=0.9)
    speaker      = TTSSpeaker()

    cropper = None
    if cfg.use_mediapipe:
        cropper = HandCropper(static_image_mode=False, min_detection_confidence=0.5)

    cap = cv2.VideoCapture(0)
    y1_fb, y2_fb, x1_fb, x2_fb = FALLBACK_ROI
    print("\n📝 Live recognised word: ")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to read from webcam")
                break

            frame = cv2.flip(frame, 1)
            bbox = None

            if cropper is not None:
                crop_result = cropper.crop(frame)
                roi_bgr = crop_result.image_bgr
                bbox = crop_result.bounding_box
                if not crop_result.used_mediapipe:
                    roi_bgr = frame[y1_fb:y2_fb, x1_fb:x2_fb]
            else:
                roi_bgr = frame[y1_fb:y2_fb, x1_fb:x2_fb]

            prep   = preprocessor.process(roi_bgr)
            pred   = classifier.predict(prep.batch)
            state  = word_builder.update(pred.label, pred.confidence)

            if state.new_char_committed:
                committed = state.last_committed_char
                if committed.lower() == "space":
                    if state.completed_word:
                        speaker.speak(state.completed_word)
                else:
                    speaker.speak(committed)
                print(f"\r📝 Word: {state.current_word}", end="", flush=True)

            # ---- Draw overlays ----
            if bbox:
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            else:
                cv2.rectangle(frame, (x1_fb, y1_fb), (x2_fb, y2_fb), (0, 255, 0), 2)

            tag = f"{pred.label} ({pred.confidence:.2f}) | {cfg.name}"
            cv2.putText(frame, tag, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            cv2.imshow("Sign Language Translator", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("\n👋 Exiting…")
                break
            if key == ord("r"):
                word_builder.reset()
                print("\n🔄 Word reset")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        speaker.stop()
        if cropper:
            cropper.close()


if __name__ == "__main__":
    main()