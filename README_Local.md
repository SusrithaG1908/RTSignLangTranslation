# 🤟 Sign Language Translator (AI/ML Project)

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![OpenCV](https://img.shields.io/badge/OpenCV-enabled-green)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow)

**Problem Statement**
People with hearing and speech impairments use sign language as their primary means of communication. However, most people do not understand sign language, leading to communication barriers. There is a need for an automated system that can translate sign language gestures into text and speech in real time to facilitate smooth communication.

**Solution**
An AI-powered **Sign Language Translator** that recognizes hand gestures from images or webcam feed and converts them into **text** (and optional **speech**). Built using **Computer Vision** and **Deep Learning (CNN)** to help bridge communication gaps for hearing-impaired users.

---

## 📌 Objectives

- To design a real-time sign language recognition system
- To train a deep learning model to classify hand gestures
- To display recognized signs as text
- To optionally convert text output into speech
- To build a simple user interface for interaction

---

## 📌 Features

- ✅ Real-time sign recognition using webcam (OpenCV)
- 🖼️ Image upload support (Streamlit UI)
- 🧠 CNN-based gesture classification (TensorFlow/Keras)
- 🔊 Optional text-to-speech output
- 📊 Model evaluation with accuracy, confusion matrix, and classification report
- 🧪 Train / Validate / Test pipeline
- 🌐 Simple web UI for demo

---

## 🧱 System Architecture

```
Camera / Image
      │
      ▼
Hand Detection (OpenCV / MediaPipe)
      │
      ▼
Preprocessing (Resize, Normalize)
      │
      ▼
CNN Model (TensorFlow/Keras)
      │
      ▼
Text Output  ──► (Optional) Text-to-Speech
```

---

## 🗂️ Project Structure

```
sign_language_translator/
│
├── data/
│   ├── train/
│   ├── val/
│   └── test/
│
├── models/
│   ├── sign_model.h5
│   └── class_labels.json
│
├── scripts/
│   ├── collect_data.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── predict_webcam.py
│   └── evaluate_model.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

- **Language:** Python
- **Deep Learning:** TensorFlow, Keras
- **Computer Vision:** OpenCV, MediaPipe
- **UI:** Streamlit
- **Utilities:** NumPy, scikit-learn, Matplotlib, pyttsx3

---

## 📦 Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/<your-username>/RTSignLangTranslation.git
cd RTSignLangTranslation
```

### 2️⃣ Create Virtual Environment (Recommended)

```bash
python -m venv venv
```

Activate:

**Windows**

```bash
venv\Scripts\activate
```

**Linux / macOS**

```bash
source venv/bin/activate
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 📁 Dataset Setup

### Option A: Download Dataset (Recommended)

Use ASL Alphabet dataset (Kaggle). Organize as:

```
data/
 ├── train/A, B, C, ...
 ├── val/A, B, C, ...
 └── test/A, B, C, ...
```

### Dataset Setup (Download & Organize)

```bash
python scripts/download_and_prepare_dataset.py
```

### Option B: Collect Your Own Data

```bash
python scripts/collect_data.py
```

Capture ~100–300 images per sign.

---

## 🏋️ Model Training

```bash
python scripts/train_model.py
```

This will:

- Train CNN model
- Save model to `models/sign_model.h5`
- Save class labels to `models/class_labels.json`
- Plot accuracy and loss

---

## 🎥 Real-Time Webcam Prediction

```bash
python scripts/predict_webcam.py
```

Controls:

- Show your hand in the ROI box
- Press `q` to quit

---

## 🌐 Web UI (Streamlit)

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

Features:

- Upload image
- View predicted sign
- Click **Speak** for voice output

---

## 📊 Model Evaluation

```bash
python scripts/evaluate_model.py
```

Outputs:

- Classification Report
- Confusion Matrix
- Precision, Recall, F1-score

---

## 📈 Results (Sample)

- Test Accuracy: ~90%+ (depends on dataset quality)
- Performs well under good lighting and simple background
- Some confusion between visually similar signs

---

## ⚠️ Limitations

- Sensitive to lighting and background
- Supports only static signs (alphabets)
- Limited vocabulary
- Continuous sign language not supported

---

## 🚀 Future Enhancements

- Support continuous sign language (sentences)
- Add speech-to-sign animation
- Mobile deployment using TensorFlow Lite
- Support Indian Sign Language (ISL)
- Use Transformer-based video models

---

## 🎓 Use Cases

- Assistive technology for hearing-impaired users
- Human–computer interaction
- Educational tools for learning sign language
- Accessibility features in smart devices

---

## 🧪 Demo

> 📸 Add screenshots / GIFs here  
> 🎥 Add demo video link (YouTube / Drive)

---

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the repo and submit a PR.

---

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

---

## 🙌 Acknowledgements

- ASL Dataset – Kaggle
- TensorFlow & OpenCV community
- MediaPipe by Google

---

## 📬 Contact

**Author:** Susritha Gudimetla  
**Email:** gudimetlasusritha@gmail.com  
**LinkedIn:**
