FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .

RUN pip install --upgrade pip
RUN pip install --no-cache-dir flask flask-cors pillow numpy matplotlib gunicorn tensorflow-cpu

COPY . .

EXPOSE 5000

CMD ["python", "backend/app.py"]
