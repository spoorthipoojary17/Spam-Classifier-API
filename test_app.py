
from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_home():
    response = client.get("/")
    assert response.status_code == 200

def test_spam_prediction():
    response = client.post(
        "/predict",
        json={"message": "You won a free iPhone! Call now!"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "spam"

def test_ham_prediction():
    response = client.post(
        "/predict",
        json={"message": "Hey, are we meeting for lunch?"}
    )
    assert response.status_code == 200
    assert response.json()["prediction"] == "ham"
