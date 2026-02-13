import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from pathlib import Path

IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# ---- Resolve project root safely ----
PROJECT_ROOT = Path(__file__).resolve().parents[1]

train_dir = PROJECT_ROOT / "data_mp" / "train"
val_dir   = PROJECT_ROOT / "data_mp" / "val"
test_dir  = PROJECT_ROOT / "data_mp" / "test"

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    zoom_range=0.2,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_gen = train_datagen.flow_from_directory(
    str(train_dir),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

val_gen = val_datagen.flow_from_directory(
    str(val_dir),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

test_gen = test_datagen.flow_from_directory(
    str(test_dir),
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)

print("Classes:", train_gen.class_indices)
