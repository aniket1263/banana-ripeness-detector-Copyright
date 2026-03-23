import os

# This works both locally AND inside Docker
BASE_PATH   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH  = os.path.join(BASE_PATH, "model", "saved", "banana_model_docker")
IMAGE_SIZE  = (224, 224)
CLASS_NAMES = ["overripe", "ripe", "rotten", "unripe"]
HOST        = "0.0.0.0"
PORT        = 5000
DEBUG       = False
ALLOWED_EXT = {'jpg', 'jpeg', 'png', 'webp'}