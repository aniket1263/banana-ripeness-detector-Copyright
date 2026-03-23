import pytest
import sys
import os
import io

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from app import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_home_route(client):
    res = client.get('/')
    assert res.status_code == 200
    data = res.get_json()
    assert "status" in data
    print("✅ Home route works")

def test_predict_no_file(client):
    res = client.post('/predict')
    assert res.status_code == 400
    data = res.get_json()
    assert "error" in data
    print("✅ Returns 400 when no file uploaded")

def test_predict_wrong_type(client):
    data = {'file': (io.BytesIO(b"not an image"), 'test.txt')}
    res  = client.post('/predict', data=data, content_type='multipart/form-data')
    assert res.status_code == 400
    print("✅ Returns 400 for invalid file type")

def test_predict_with_real_image(client):
    img_path = r"C:\Users\nihaa\Desktop\test_banana.jpg"
    if not os.path.exists(img_path):
        pytest.skip("Test image not found on Desktop")
    with open(img_path, 'rb') as f:
        data = {'file': (f, 'test_banana.jpg')}
        res  = client.post('/predict', data=data, content_type='multipart/form-data')
    assert res.status_code == 200
    result = res.get_json()
    assert "label"      in result
    assert "confidence" in result
    assert "scores"     in result
    assert result["label"] in ["overripe", "ripe", "rotten", "unripe"]
    print(f"✅ Prediction: {result['label']} ({result['confidence']}%)")