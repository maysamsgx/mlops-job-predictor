"""
Integration tests for the Inference Service.
Tests the API endpoints using TestClient.
"""
import os
import joblib
import pytest
import httpx
from fastapi.testclient import TestClient
from inference_service import app, MODEL_PATH

client = TestClient(app)

# Mock Model for testing without training
class MockModel:
    def predict_proba(self, X):
         # Returns [0.1, 0.9] confidence for 2 classes
        return [[0.1, 0.9]]

    @property
    def classes_(self):
        return ["Data_Scientist", "ML_Engineer"]

# Fixture to mock the model
@pytest.fixture
def mock_model(monkeypatch):
    """
    Mocks the joblib.load to return a MockModel.
    """
    def mock_load(path):
        return MockModel()

    monkeypatch.setattr(joblib, "load", mock_load)
    # We also need to patch the global 'model' variable in inference_service
    # But since app loads it on startup, we might need to rely on the module's reload or mock patch object.
    from inference_service import model
    # Doing a direct patch on the loaded model object if possible,
    # but inference_service loads it at top level (maybe).
    # Let's verify inference_service.py structure.
    # It loads: model = joblib.load(MODEL_PATH)

    monkeypatch.setattr("inference_service.model", MockModel())


def test_predict_endpoint_valid(mock_model):
    """
    Test /predict with valid input.
    """
    payload = {
        "skills": "Python, TensorFlow",
        "qualification": "M.Sc",
        "experience_level": "Senior"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_role"] == "ML_Engineer"
    assert data["confidence"] == 0.9
    assert data["status"] == "Success"

def test_predict_endpoint_fallback(monkeypatch):
    """
    Test algorithmic fallback when confidence is low.
    """
    # Mock model with low confidence
    class LowConfModel:
        def predict_proba(self, X):
            return [[0.4, 0.4, 0.2]] # Max 0.4 < 0.6
        @property
        def classes_(self):
            return ["A", "B", "C"]

    monkeypatch.setattr("inference_service.model", LowConfModel())

    payload = {
        "skills": "Basic Skills",
        "qualification": "B.Sc",
        "experience_level": "Junior"
    }

    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["predicted_role"] == "Generalist_Candidate_Review_Required"
    assert "Fallback_Triggered" in data["status"]
