import streamlit as st
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import pyttsx3
from PIL import Image

IMG_SIZE = (128, 128)

@st.cache_resource
def load_trained_model():
    model = load_model("models/sign_model.h5")
    with open("models/class_labels.json", "r") as f:
        class_indices = json.load(f)
    class_labels = {int(v): k for k, v in class_indices.items()}
    return model, class_labels

model, class_labels = load_trained_model()

st.title("🤟 Sign Language Translator (AI/ML)")
st.write("Upload a hand gesture image and get the predicted sign.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "png", "jpeg"])

def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    img = image.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    preds = model.predict(img)
    pred_class = np.argmax(preds)
    confidence = float(np.max(preds))

    label = class_labels[pred_class]

    st.success(f"Predicted Sign: {label}")
    st.write(f"Confidence: {confidence:.2f}")

    if st.button("🔊 Speak"):
        speak(label)
