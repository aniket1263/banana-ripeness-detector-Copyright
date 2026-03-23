import os
import zipfile

DATASET_ID   = "shahriar26s/banana-ripeness-classification-dataset"
DOWNLOAD_DIR = r"C:\Users\nihaa\OneDrive\banana ripness project\dataset"

print("📦 Downloading dataset from Kaggle...")
os.system(f"kaggle datasets download -d {DATASET_ID} -p \"{DOWNLOAD_DIR}\"")

# Unzip
zip_path = os.path.join(DOWNLOAD_DIR, "banana-ripeness-classification-dataset.zip")
if os.path.exists(zip_path):
    print("📂 Extracting...")
    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(DOWNLOAD_DIR)
    os.remove(zip_path)
    print("✅ Dataset ready!")
else:
    print("❌ Zip not found. Check your Kaggle API key.")