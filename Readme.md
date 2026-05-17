This is copyright project of our team.
Aniket Arora(2210991263)
Alok Ranjan(22210991242)
Sudeep Rohaj(2210992412)

Copyright-Diary-No.=>

# 🍌 Banana Ripeness Detector

An end-to-end Machine Learning web app that classifies banana ripeness
into 4 stages using MobileNetV2 transfer learning + Flask REST API

- mobile-friendly frontend with camera support + Docker deployment.

---

## 🎯 Demo

Upload or snap a banana photo → get instant ripeness classification.

| Stage | Label    | Description                       |
| ----- | -------- | --------------------------------- |
| 🟢    | Unripe   | Firm & starchy. Best for cooking. |
| 🟡    | Ripe     | Sweet & soft. Perfect to eat now. |
| 🟠    | Overripe | Very sweet. Great for smoothies.  |
| 🔴    | Rotten   | Discard or compost.               |

---

## 🗂️ Project Structure

```
banana ripness project/
├── backend/
│   ├── app.py                  ← Flask entry point
│   ├── config.py               ← Paths, class names, settings
│   ├── routes/
│   │   └── predict.py          ← POST /predict endpoint
│   └── utils/
│       ├── preprocess.py       ← Image resize + normalize
│       └── gradcam.py          ← Grad-CAM heatmap generation
├── frontend/
│   └── templates/
│       └── index.html          ← BananaLens UI (camera + upload)
├── model/
│   ├── saved/
│   │   ├── banana_model.keras          ← Local model
│   │   └── banana_model_docker/        ← Docker model (SavedModel)
│   └── checkpoints/
│       └── best_model.h5
├── notebooks/
│   └── ripeness.ipynb          ← Training + Grad-CAM notebook
├── scripts/
│   ├── download_dataset.py     ← Kaggle dataset downloader
│   ├── train.py                ← Standalone training script
│   └── evaluate.py             ← Test set evaluation
├── tests/
│   ├── test_model.py           ← Model inference unit tests
│   └── test_api.py             ← Flask API endpoint tests
├── Dockerfile                  ← Docker image config
├── docker-compose.yml          ← Docker compose config
├── .dockerignore               ← Docker ignore rules
├── requirements.txt            ← Python dependencies
├── .gitignore                  ← Git ignore rules
└── README.md
```

---

## 🧠 ML Concepts Used

| Concept                  | Usage                                      |
| ------------------------ | ------------------------------------------ |
| Transfer Learning        | MobileNetV2 pretrained on ImageNet         |
| CNN                      | Feature extraction from images             |
| Data Augmentation        | Rotation, zoom, flip, brightness           |
| Dropout                  | Regularization to prevent overfitting      |
| Softmax                  | Multi-class probability output             |
| Grad-CAM                 | Explainable AI — visualize model attention |
| Early Stopping           | Stop training when val_loss plateaus       |
| Adam Optimizer           | Adaptive learning rate optimization        |
| Categorical Crossentropy | Multi-class loss function                  |
| GlobalAveragePooling     | Dimensionality reduction                   |

---

## 🚀 Quick Start

### Option A — Run with Docker (Recommended)

```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/banana-ripeness-detector.git
cd banana-ripeness-detector

# Start app
docker-compose up

# Open browser
http://localhost:5000
```

### Option B — Run Locally

```bash
# Install dependencies
pip install -r requirements.txt

# Run API
cd backend
python app.py

# Open browser
http://127.0.0.1:5000
```

---

## 📡 API Reference

### `GET /`

Serves the frontend UI.

### `GET /health`

Returns API status.

```json
{ "health": "ok" }
```

### `POST /predict`

Upload banana image → get ripeness classification.

**Request:**

```
Content-Type: multipart/form-data
Body: file = <image file>
```

**Response:**

```json
{
  "label": "ripe",
  "confidence": 91.5,
  "scores": {
    "overripe": 3.2,
    "ripe": 91.5,
    "rotten": 1.8,
    "unripe": 3.5
  }
}
```

**Test with curl:**

```bash
curl -X POST -F "file=@banana.jpg" http://localhost:5000/predict
```

---

## 🐳 Docker Commands

| Command                           | Description              |
| --------------------------------- | ------------------------ |
| `docker-compose up`               | Start app                |
| `docker-compose up -d`            | Start in background      |
| `docker-compose down`             | Stop app                 |
| `docker-compose up --build`       | Rebuild and start        |
| `docker logs banana_ripeness_app` | View logs                |
| `docker ps`                       | Check running containers |

---

## 🧪 Run Tests

```bash
# Run all tests
pytest tests/ -v

# Run specific tests
pytest tests/test_model.py -v
pytest tests/test_api.py -v
```

---

## 📊 Dataset

- **Source:** Kaggle — shahriar26s/banana-ripeness-classification-dataset
- **Classes:** overripe, ripe, rotten, unripe
- **Split:** train / valid / test (pre-split)
- **Download:**

```bash
python scripts/download_dataset.py
```

---

## 🔁 Retrain Model

```bash
# Download dataset
python scripts/download_dataset.py

# Train
python scripts/train.py

# Evaluate
python scripts/evaluate.py
```

---

## 🛠️ Tech Stack

| Layer          | Technology                     |
| -------------- | ------------------------------ |
| ML Model       | TensorFlow, Keras, MobileNetV2 |
| Backend        | Flask, Flask-CORS              |
| Frontend       | HTML, CSS, Vanilla JS          |
| Deployment     | Docker, Docker Compose         |
| Explainability | Grad-CAM                       |
| Testing        | Pytest                         |
| IDE            | VS Code                        |

---

## 📱 Features

- 📷 Camera capture on mobile and desktop
- 📁 Image upload from gallery or file system
- 📊 Confidence bars for all 4 classes
- 💡 Storage tips based on ripeness stage
- 🐳 Docker containerized deployment
- ✅ API health check indicator
- 📱 Fully responsive mobile-friendly UI

---

## 👨‍💻 Author

**Aniket**
B.Tech Computer Science
