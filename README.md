# 🤟 Sign Language Translator (AI/ML Project)

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
Check available Python versions on system:
py -0p

If single python version is available on system:

Create Virtual Environment
python -m venv .venv

Activate:
**Windows**
.\.venv\Scripts\activate

If multiple python versions are available on system:

Create Virtual Environment
py -3.11 -m venv .venv

Activate:
**Windows**
.\.venv\Scripts\activate

Check Version:
python --version

Create ipykernel associated with venv
pip install ipykernel

Register ipykernel to select from IDE
python -m ipykernel install --user --name venv311 --display-name "Python 3.11 (.venv)"

Select "Python 3.11 (.venv)" as the Python Interpreter in IDE
Press Ctrl + Shift + P
Select Python Interpreter
From the available list, select "Python 3.11 (.venv)"
```

## 📁 Dataset Setup

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

### Collect Your Own Dataset (Looks Great in Demo & Viva)

```bash
python scripts/collect_data.py
```

## 🏋️ Model Training

## 🌐 Web UI (Streamlit)

## 📈 Results (Sample)

## ⚠️ Limitations

## 🚀 Future Enhancements

## 🎓 Use Cases

## 🧪 Demo

## 🤝 Contributing

Contributions are welcome!  
Feel free to fork the repo and submit a PR.

## 📜 License

This project is licensed under the **MIT License**.  
You are free to use, modify, and distribute with attribution.

## 🙌 Acknowledgements

## 📬 Contact

**Author:** Susritha Gudimetla  
**Email:** gudimetlasusritha@gmail.com
