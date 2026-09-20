from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)

def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

def test_predict_form():
    with open("test_sample_form.png", "rb") as f:
        response = client.post("/predict", files={"file": f})

    assert response.status_code == 200
    data = response.json()
    assert data["predicted_class"] == "form"
    assert data["confidence"] > 0.9
