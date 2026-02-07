# 🤟 Sign Language Translator (AI/ML Project)

An AI-powered **Sign Language Translator** that recognizes hand gestures from images or webcam feed and converts them into **text** (and optional **speech**). Built using **Computer Vision** and **Deep Learning (CNN)** to help bridge communication gaps for hearing-impaired users.

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
git clone https://github.com/<your-username>/sign_language_translator.git
cd sign_language_translator
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
