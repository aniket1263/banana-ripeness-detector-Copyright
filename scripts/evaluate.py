import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix
import os

# ── Paths ──
BASE_PATH  = r"C:\Users\nihaa\OneDrive\banana ripness project"
TEST_DIR   = os.path.join(BASE_PATH, "dataset", "Banana Ripeness Classification Dataset", "test")
MODEL_PATH = os.path.join(BASE_PATH, "model", "saved", "banana_model.keras")

# ── Load ──
print("🍌 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

test_gen  = ImageDataGenerator(rescale=1./255)
test_data = test_gen.flow_from_directory(
    TEST_DIR, target_size=(224,224),
    batch_size=32, class_mode='categorical', shuffle=False
)

class_names = list(test_data.class_indices.keys())

# ── Evaluate ──
loss, acc = model.evaluate(test_data)
print(f"\n✅ Test Accuracy : {acc * 100:.2f}%")
print(f"✅ Test Loss     : {loss:.4f}")

# ── Confusion Matrix ──
y_pred = np.argmax(model.predict(test_data), axis=1)
y_true = test_data.classes

cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=class_names,
            yticklabels=class_names,
            cmap='YlOrBr')
plt.title("Confusion Matrix")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig(os.path.join(BASE_PATH, "model", "confusion_matrix.png"))
plt.show()

# ── Report ──
print("\n📊 Classification Report:")
print(classification_report(y_true, y_pred, target_names=class_names))