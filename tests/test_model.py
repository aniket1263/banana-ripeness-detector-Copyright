import pytest
import numpy as np
import tensorflow as tf
import os

MODEL_PATH  = r"C:\Users\nihaa\OneDrive\banana ripness project\model\saved\banana_model.keras"
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]

@pytest.fixture(scope="module")
def model():
    return tf.keras.models.load_model(MODEL_PATH)

def test_model_loads(model):
    assert model is not None
    print("✅ Model loads successfully")

def test_output_shape(model):
    dummy = np.zeros((1, 224, 224, 3), dtype=np.float32)
    preds = model.predict(dummy)
    assert preds.shape == (1, 4), f"Expected (1,4) got {preds.shape}"
    print("✅ Output shape correct:", preds.shape)

def test_output_sum(model):
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    preds = model.predict(dummy)
    total = np.sum(preds)
    assert abs(total - 1.0) < 1e-5, f"Probabilities should sum to 1, got {total}"
    print("✅ Probabilities sum to 1")

def test_prediction_is_valid_class(model):
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    preds = model.predict(dummy)
    idx   = np.argmax(preds)
    assert CLASS_NAMES[idx] in CLASS_NAMES
    print(f"✅ Valid class predicted: {CLASS_NAMES[idx]}")

def test_confidence_range(model):
    dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
    preds = model.predict(dummy)[0]
    assert all(0 <= p <= 1 for p in preds), "All confidence values should be 0-1"
    print("✅ All confidence values in valid range")