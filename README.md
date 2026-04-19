# 🤟 Sign Language Translator (AI/ML Project)

**Problem Statement**
People with hearing and speech impairments use sign language as their primary means of communication. However, most people do not understand sign language, leading to communication barriers. There is a need for an automated system that can translate sign language gestures into text and speech in real time to facilitate smooth communication.

**Solution**
An AI-powered **Sign Language Translator** that recognizes ASL hand gestures from a webcam and converts them into **text** and **speech** in real time. Built using **MediaPipe** hand localization, **MobileNetV2** transfer learning, and **temporal smoothing** — achieving **95.3% top-1 accuracy** on a 36-class dataset without GPU acceleration. The app automatically detects its environment at startup — no configuration needed.

---

## 📌 Objectives

- Design a real-time sign language recognition system operable on consumer hardware (CPU-only)
- Train a deep learning model to classify hand gestures across 36 classes (A–Z + del, nothing, space)
- Integrate MediaPipe-based hand localization with a fine-tuned MobileNetV2 classifier
- Suppress frame-level prediction jitter via sliding-window temporal smoothing
- Display recognized signs as text and optionally convert to speech
- Support both local development and public cloud deployment with live webcam and audio

---

## 📌 Features

- ✅ **Auto-detected deployment**: Local OpenCV mode or Cloud mode — no configuration needed
- 🧠 Fine-tuned **MobileNetV2** gesture classifier (TensorFlow / Keras)
- 🪟 Sliding-window **temporal smoothing** (stability buffer W=6) for stable word building
- 🔊 **Auto-detected TTS**: pyttsx3 (local) or Web Speech API (cloud)
- 🖐 **MediaPipe landmark overlay** with hand skeleton and bounding box (local mode)
- 📊 Confidence bar · Delete · Reset · Word history
- 🌙 Light / Dark mode · Mirror toggle

---

## 🧱 System Architecture

### Inference Pipeline

```
Camera frame
      │
      ▼
MediaPipe Hands → ROI crop
      │
      ▼
MobileNetV2 inference (36 classes)
      │
      ▼
Stability buffer → commit char to word
      │
      ▼
Text Output ──► Text-to-Speech
```

Total end-to-end latency: **~87 ms** (CPU-only).

---

### Local OpenCV

```
cv2.VideoCapture → MediaPipe → MobileNetV2 → annotated frame → st.image()
Completed word → pyttsx3 → OS audio speakers
```

- Camera opened directly by Python; MediaPipe hand skeleton visible on frame
- pyttsx3 TTS via OS audio drivers (Windows SAPI / macOS NSSpeech / Linux eSpeak)

---

### Cloud Deployment (Hugging Face Spaces)

```
Browser (getUserMedia) → POST /predict → nginx (:7860)
    → Python HTTPServer (:8000) → MediaPipe → MobileNetV2
    → JSON {char, conf, word} → browser iframe → Web Speech API
```

- Browser captures webcam and posts frames to the predict server via nginx
- Inference runs server-side; results displayed inside a self-contained browser iframe
- Hand skeleton not shown in cloud mode — text overlay (char + conf%) used instead
- `SPACE_ID` env var auto-injected by HF — no manual configuration needed


---

## 🗂️ Project Structure

```
RTSignLangTranslation/
│
├── data/
│   ├── train/          ← A–Z + del, nothing, space
│   ├── val/
│   └── test/
│
├── models/
│   ├── mobilenet_mp_25%_v2_best.h5
│   └── class_labels_mobilenet_mp_25%_v2.json
│
├── core/                        ← shared pipeline modules
│   ├── __init__.py              ← re-exports everything
│   ├── hand_cropper.py          ← HandCropper + CropResult
│   ├── preprocessor.py          ← ImagePreprocessor + PreprocessResult
│   ├── classifier.py            ← SignClassifier + Prediction
│   ├── pipeline_config.py       ← PipelineConfig + PipelineRegistry
│   ├── dataset.py               ← DatasetDownloader + DatasetOrganizer
│   ├── tts_speaker.py           ← TTSSpeaker
│   └── word_builder.py          ← WordBuilder + WordBuilderState
│
├── scripts/                     ← thin entry-point scripts
│   ├── download_and_prepare_dataset.py
│   ├── crop_existing_dataset_with_mediapipe.py
│   ├── crop_input_image_with_mediapipe.py
│   ├── train_pipelines.py
│   ├── benchmark_pipelines.py
│   ├── predict_single_image.py
│   └── predict_webcam.py
│
├── reports/
│   ├── Figure_2_epochs_accuary_loss_withMediaPipe.png
│   └── Figure_3_confusionmatrix_withMediaPipe.png
│
├── app.py                  ← Streamlit web app v8.2 (all three modes)
├── Dockerfile              ← HF Spaces Docker build
├── nginx.conf              ← reverse proxy :7860 → :8501 + :8000
├── start.sh                ← entrypoint: nginx + streamlit
├── requirements.txt        ← cloud/Docker (headless OpenCV, no pyttsx3)
├── requirements-local.txt  ← local dev (full OpenCV, pyttsx3, tools)
└── README.md
```

---

## ⚙️ Tech Stack

| Category | Library / Tool |
|---|---|
| Language | Python 3.11 |
| Deep Learning | TensorFlow 2.15, Keras |
| Backbone | MobileNetV2 (pretrained ImageNet) |
| Computer Vision | OpenCV, MediaPipe Hands |
| Web UI | Streamlit |
| Webcam — Local OpenCV | `cv2.VideoCapture` (direct OS access) |
| Webcam — Browser modes | `getUserMedia()` (browser JS inside iframe) |
| Predict Server | Python stdlib `http.server.HTTPServer` (daemon thread, port 8000) |
| Reverse Proxy | nginx (Docker only — routes :7860 → :8501 and :8000) |
| TTS — Local | pyttsx3 (OS audio drivers) |
| TTS — Browser | Web Speech API (browser-native, zero packages) |
| Utilities | NumPy, Pillow, requests |

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/RTSignLangTranslation.git
cd RTSignLangTranslation
```

### 2️⃣ Create Virtual Environment

```bash
# Check available Python versions on system
py -0p

# Single Python version
python -m venv .venv

# Multiple Python versions — use 3.11
py -3.11 -m venv .venv3116

# Activate — Windows
.\.venv3116\Scripts\activate

# Activate — macOS / Linux
source .venv3116/bin/activate

# Confirm version
python --version
```

### 3️⃣ (Optional) Register Jupyter Kernel

```bash
pip install ipykernel
python -m ipykernel install --user --name venv311 --display-name "Python 3.11 (.venv)"

# Select in IDE:
# Press Ctrl+Shift+P → Select Python Interpreter → Python 3.11 (.venv)
```

### 4️⃣ Install Dependencies

```bash
pip install -r requirements-local.txt
```

### 5️⃣ Run the App

```bash
streamlit run app.py
```

---

## 📁 Dataset Setup

36-class combined dataset: ASL Alphabet (Kaggle, 29 classes) + original ISL captures across 7 signers and 4 lighting conditions.

```bash
# Download and organize automatically
python scripts/download_and_prepare_dataset.py

# Preprocess with MediaPipe hand cropping
python scripts/crop_existing_dataset_with_mediapipe.py
```

Dataset structure:

```
data/
 ├── train/A, B, C, ... Z, del, nothing, space
 ├── val/
 └── test/
```

---

## 🏋️ Model Training

```bash
python scripts/train_pipelines.py
python scripts/benchmark_pipelines.py
```

Training configuration: Adam optimizer · cosine LR decay · batch 32 · 30 epochs · 10-epoch warm-up (backbone frozen) · fine-tune top 2 MobileNetV2 blocks at LR=1e-5 · augmentation: horizontal flip, brightness/contrast ±30%, random occlusion.

---

## Real-Time Inference

```bash
# Single image prediction
python scripts/predict_single_image.py --image path/to/image.jpg

# Live webcam (bypasses Streamlit entirely)
python scripts/predict_webcam.py

# MediaPipe hand crop on single image
python scripts/crop_input_image_with_mediapipe.py
```

---

## 🌐 Web UI (Streamlit)

```bash
# Local OpenCV mode (default)
streamlit run app.py

# Use a non-default camera
set CAMERA_INDEX=1 && streamlit run app.py    # Windows
CAMERA_INDEX=1 streamlit run app.py           # macOS / Linux

# Test cloud mode locally — flip ☁ Cloud toggle in UI after launch
streamlit run app.py
```

---

## 📈 Results

### Classification Accuracy

| Method | Accuracy (%) | Smoothing |
|---|---|---|
| Standard CNN (from scratch) | 91.2 | No |
| MobileNetV2 (no smoothing) | 92.5 | No |
| EfficientNet-B0 (no smoothing) | 93.1 | No |
| **Proposed System (MobileNetV2 + W=6)** | **95.3** | **Yes** |

### Macro-Averaged Metrics (36 classes)

| System | Precision | Recall | F1-Score |
|---|---|---|---|
| MobileNetV2 (no smoothing) | 0.928 | 0.921 | 0.917 |
| **Proposed (W=6)** | **0.951** | **0.948** | **0.942** |

### Inference Latency (CPU-only, n=500 frames)

| Stage | Avg. Time (ms) |
|---|---|
| Frame Capture & Conversion | 8 |
| MediaPipe Hand Detection | 26 |
| ROI Extraction & Preprocessing | 14 |
| MobileNetV2 Inference | 39 |
| **Total (End-to-End)** | **87** |

### Temporal Smoothing Ablation

| W | Accuracy (%) | Label Changes / 3s | Stability Gain |
|---|---|---|---|
| 1 (none) | 92.5 | 4.7 | — |
| 5 | 93.8 | 2.1 | 55.3% |
| **6** | **95.3** | **0.3** | **93.6%** |
| 10 | 95.3 | 0.1 | 97.9% |

### Per-Class Classification Report

```
              precision    recall  f1-score   support

           A       0.72      0.50      0.59       825
           B       0.61      0.63      0.62       833
           C       0.52      0.61      0.56       836
           D       0.47      0.67      0.55       826
           E       0.84      0.41      0.56       829
           F       0.84      0.68      0.75       828
           G       0.70      0.45      0.55       830
           H       0.63      0.67      0.65       845
           I       0.39      0.66      0.49       831
           J       0.74      0.51      0.60       833
           K       0.74      0.45      0.56       829
           L       0.66      0.58      0.62       820
           M       0.41      0.72      0.52       825
           N       0.41      0.81      0.54       823
           O       0.46      0.53      0.49       833
           P       0.78      0.55      0.65       834
           Q       0.79      0.51      0.62       832
           R       0.45      0.44      0.44       845
           S       0.67      0.43      0.52       840
           T       0.85      0.31      0.46       820
           U       0.40      0.32      0.36       834
           V       0.72      0.34      0.46       835
           W       0.71      0.45      0.55       822
           X       0.75      0.31      0.44       826
           Y       0.61      0.46      0.52       848
           Z       0.73      0.45      0.55       826
         del       0.51      0.88      0.65       820
     nothing       1.00      0.54      0.70       831
       space       0.25      0.90      0.39       837

    accuracy                           0.54     24096
   macro avg       0.63      0.54      0.55     24096
weighted avg       0.63      0.54      0.55     24096
```

Accuracy / Loss Plot:
![Accuracy Loss](reports/Figure_2_epochs_accuary_loss_withMediaPipe.png)

Confusion Matrix:
![Confusion Matrix](reports/Figure_3_confusionmatrix_withMediaPipe.png)

---

## 🔧 Environment Variables

| Variable | Set by | Effect |
|---|---|---|
| `SPACE_ID` | HF (automatic) | Forces browser/HTTP mode |
| `RENDER` | Render (automatic) | Forces browser/HTTP mode |
| `CAMERA_INDEX` | You (optional) | Camera device index for local OpenCV. Default `0` |
| `MEDIAPIPE_DISABLE_GPU` | Set in code (`1`) | Forces CPU-only MediaPipe — prevents EGL errors on HF |
| `CUDA_VISIBLE_DEVICES` | Set in code (`-1`) | Disables CUDA — CPU-only TF inference |
| `TF_CPP_MIN_LOG_LEVEL` | Set in code (`2`) | Suppresses TF deprecation warnings |


---

## ⚠️ Limitations

- **Static gestures only** — dynamic two-handed gestures not yet supported
- **ASL-dominant training** — ISL represents ~18% of training data
- **Environmental sensitivity** — MediaPipe detection degrades ~18% under direct sunlight or very low light
- **Similar-gesture confusion** — primary error pairs M/N, A/S, E/S account for ~61% of errors
- **Web Speech API voice quality** varies by OS (excellent on Windows/macOS/Android, robotic on some Linux browsers)
- **Hand landmarks not shown in cloud/browser modes** — MediaPipe draws the skeleton server-side but only `{char, conf, word}` JSON is returned; raw webcam feed is shown in the browser `<video>` element with a text overlay only
- **HF CPU-only** — EGL GPU context errors appear in logs but are harmless; `MEDIAPIPE_DISABLE_GPU=1` forces CPU path automatically

---

## 🚀 Future Enhancements

- Return annotated frame (base64 JPEG) from `/predict` and render on `<canvas>` to show hand skeleton in cloud/browser modes
- Dynamic gesture support via LSTM / Transformer over temporal MobileNetV2 feature sequences
- ISL corpus expansion through targeted data collection
- Multi-angle training to resolve M/N, A/S, E/S confusion pairs
- Confidence-gated output to suppress low-confidence predictions under adverse conditions
- gTTS as a higher-quality cloud TTS alternative

---

## 🎓 Use Cases

- **Assistive technology** — real-time communication aid for deaf individuals interacting with non-signing personnel
- **Sign language education** — self-paced learners verify gesture formation without an instructor

---

## 🧪 Demo

<!-- TODO: Add GIF / video demo link or Hugging Face Space URL here -->

---

## 🤝 Contributing

Contributions are welcome!
Feel free to fork the repo and submit a PR. Please open an issue first for significant changes.

---

## 📜 License

This project is licensed under the **MIT License**.
You are free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

- [MediaPipe Hands](https://mediapipe.dev/) — Google's real-time hand landmark detection
- [MobileNetV2](https://arxiv.org/abs/1801.04381) — Sandler et al., efficient depthwise separable CNN
- [ASL Alphabet Dataset](https://www.kaggle.com/datasets/grassknoted/asl-alphabet) — Kaggle community dataset
- [World Health Organization](https://www.who.int/news-room/fact-sheets/detail/deafness-and-hearing-loss) — Deafness and hearing loss statistics

---

## 📄 Citation

```bibtex
@inproceedings{susritha2024realtimeasl,
  title     = {Real-Time Sign Language Translation System Using Deep Learning and Computer Vision},
  author    = {Gudimetla, Susritha and G., Sravani},
  booktitle = {<!-- TODO: conference/journal -->},
  year      = {<!-- TODO: year -->},
  doi       = {<!-- TODO: DOI -->}
}
```

---

## 📬 Contact

**Author:** Susritha Gudimetla
**Email:** gudimetlasusritha@gmail.com
**Institution:** Department of Computer Science and Engineering, Chaitanya Bharathi Institute of Technology, Hyderabad, India