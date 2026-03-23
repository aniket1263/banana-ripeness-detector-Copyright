from flask import Blueprint, request, jsonify
from utils.preprocess import preprocess_image, validate_image
import tensorflow as tf
import numpy as np
import io
import sys
import os
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from config import MODEL_PATH, CLASS_NAMES

predict_bp = Blueprint('predict', __name__)

print("🍌 Loading model...")
print(f"📂 Model path: {MODEL_PATH}")
print(f"📂 Path exists: {os.path.exists(MODEL_PATH)}")

try:
    model = tf.saved_model.load(MODEL_PATH)
    infer = model.signatures["serving_default"]
    USE_SAVED_MODEL = True
    print("✅ Model loaded (SavedModel format)!")
    # Print input/output keys for debugging
    print(f"🔑 Input keys:  {list(infer.structured_input_signature[1].keys())}")
    print(f"🔑 Output keys: {list(infer.structured_outputs.keys())}")
except Exception as e:
    print(f"⚠️ SavedModel failed: {e}")
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        infer = None
        USE_SAVED_MODEL = False
        print("✅ Model loaded (Keras format)!")
    except Exception as e2:
        print(f"❌ Keras load also failed: {e2}")
        model = None
        infer = None
        USE_SAVED_MODEL = False

@predict_bp.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model failed to load"}), 500

    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']

    is_valid, error = validate_image(file)
    if not is_valid:
        return jsonify({"error": error}), 400

    try:
        img_array = preprocess_image(io.BytesIO(file.read()))
        print(f"📸 Image shape: {img_array.shape}")

        if USE_SAVED_MODEL:
            input_tensor = tf.constant(img_array)
            # Get correct input key
            input_key    = list(infer.structured_input_signature[1].keys())[0]
            result       = infer(**{input_key: input_tensor})
            output_key   = list(result.keys())[0]
            preds        = result[output_key].numpy()[0]
            print(f"✅ SavedModel preds: {preds}")
        else:
            preds = model.predict(img_array)[0]
            print(f"✅ Keras preds: {preds}")

        idx        = int(np.argmax(preds))
        label      = CLASS_NAMES[idx]
        confidence = round(float(preds[idx]) * 100, 2)
        scores     = {
            CLASS_NAMES[i]: round(float(preds[i]) * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "label":      label,
            "confidence": confidence,
            "scores":     scores
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500