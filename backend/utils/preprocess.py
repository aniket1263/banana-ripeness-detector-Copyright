from PIL import Image
import numpy as np
import io

IMAGE_SIZE = (224, 224)

def preprocess_image(file_bytes):
    """
    Takes raw file bytes, returns normalized numpy array
    ready for model.predict()
    """
    img = Image.open(file_bytes).resize(IMAGE_SIZE).convert('RGB')
    arr = np.array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


def validate_image(file):
    """
    Checks file extension is allowed.
    Returns (True, None) if valid, (False, error_msg) if not.
    """
    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    
    if file.filename == '':
        return False, "Empty filename"
    
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return False, f"File type .{ext} not allowed. Use jpg, png, or webp."
    
    return True, None