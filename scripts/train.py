import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import os

# ── Paths ──
BASE_PATH  = r"C:\Users\nihaa\OneDrive\banana ripness project"
TRAIN_DIR  = os.path.join(BASE_PATH, "dataset", "Banana Ripeness Classification Dataset", "train")
VAL_DIR    = os.path.join(BASE_PATH, "dataset", "Banana Ripeness Classification Dataset", "valid")
SAVE_PATH  = os.path.join(BASE_PATH, "model", "saved", "banana_model.keras")
CKPT_PATH  = os.path.join(BASE_PATH, "model", "checkpoints", "best_model.h5")

# ── Config ──
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS     = 15

# ── Data ──
train_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.8, 1.2]
)
val_gen = ImageDataGenerator(rescale=1./255)

train_data = train_gen.flow_from_directory(TRAIN_DIR, target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')
val_data   = val_gen.flow_from_directory(VAL_DIR,   target_size=IMAGE_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

print("Classes:", train_data.class_indices)

# ── Model ──
base = MobileNetV2(input_shape=(224,224,3), include_top=False, weights='imagenet')
base.trainable = False

model = models.Sequential([
    base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# ── Train ──
callbacks = [
    EarlyStopping(patience=3, restore_best_weights=True),
    ModelCheckpoint(CKPT_PATH, save_best_only=True)
]

history = model.fit(train_data, validation_data=val_data, epochs=EPOCHS, callbacks=callbacks)

# ── Save ──
model.save(SAVE_PATH)
print(f"✅ Model saved to {SAVE_PATH}")