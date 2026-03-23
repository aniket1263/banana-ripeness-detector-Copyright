FROM tensorflow/tensorflow:2.13.0

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir flask flask-cors pillow numpy matplotlib gunicorn

COPY . .

EXPOSE 5000

CMD ["python", "backend/app.py"]
