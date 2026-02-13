import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ---- Resolve project root safely ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

TEST_DIR = PROJECT_ROOT / "data" / "test"
MODEL_PATH = PROJECT_ROOT / "models" / "sign_model.h5"
LABELS_PATH = PROJECT_ROOT / "models" / "class_labels.json"

test_datagen = ImageDataGenerator(rescale=1./255)

test_gen = test_datagen.flow_from_directory(
    str(TEST_DIR),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

model = load_model(str(MODEL_PATH))

with open(LABELS_PATH, "r") as f:
    class_indices = json.load(f)

class_labels = {int(v): k for k, v in class_indices.items()}
labels = [class_labels[i] for i in range(len(class_labels))]

preds = model.predict(test_gen, verbose=1)
y_pred = np.argmax(preds, axis=1)
y_true = test_gen.classes

print("\nClassification Report:\n")
print(classification_report(y_true, y_pred, target_names=labels))

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=False, fmt="d", cmap="Blues",
            xticklabels=labels, yticklabels=labels)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
