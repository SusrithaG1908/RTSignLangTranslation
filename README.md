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
│   ├── download_and_prepare_dataset.py
│   ├── collect_data.py
│   ├── crop_existing_dataset_with_mediapipe.py
│   ├── preprocess.py
│   ├── train_model.py
│   ├── evaluate_model.py
│   ├── predict_webcam.py
│   ├── crop_input_image_with_mediapipe.py
│   ├── predict_single_image.py
│   └── app.py
│
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

### Collect Your Own Dataset

```bash
python scripts/collect_data.py
```

### Preprocess

```bash
python scripts/crop_existing_dataset_with_mediapipe.py
python scripts/preprocess.py
```

## 🏋️ Model Training

```bash
python scripts/train_model.py
python scripts/evaluate_model.py
```

## Real Time Inference

```bash
python scripts/predict_webcam.py
python scripts/crop_input_image_with_mediapipe.py
python scripts/predict_single_image.py
```

## 🌐 Web UI (Streamlit)

```bash
python scripts/app.py
```

## 📈 Results

Accuracy Loss Plot:
![alt text](reports/Figure_2_epochs_accuary_loss_withMediaPipe.png)

Classification Report:

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

Confusion Matrix:
![alt text](reports/Figure_3_confusionmatrix_withMediaPipe.png)

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
