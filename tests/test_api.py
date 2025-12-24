"""
Tests for Financial Risk Prediction API
"""
import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestHealthEndpoint:
    """Tests for /health endpoint."""
    
    def test_health_returns_200(self):
        """Health endpoint should return 200."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        
    def test_health_contains_status(self):
        """Health response should contain status field."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/health")
        data = response.json()
        assert "status" in data
        assert data["status"] == "ok"


class TestRootEndpoint:
    """Tests for root endpoint."""
    
    def test_root_returns_200(self):
        """Root endpoint should return 200."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/")
        assert response.status_code == 200
        
    def test_root_contains_message(self):
        """Root response should contain welcome message."""
        from src.api.main import app
        client = TestClient(app)
        response = client.get("/")
        data = response.json()
        assert "message" in data


class TestPredictEndpoint:
    """Tests for /predict endpoint."""
    
    @pytest.fixture
    def valid_payload(self):
        """Valid prediction request payload."""
        return {
            "age": 35,
            "gender": "Male",
            "education_level": "Bachelor's",
            "marital_status": "Married",
            "income": 75000.0,
            "credit_score": 720.0,
            "loan_amount": 25000.0,
            "loan_purpose": "Auto",
            "employment_status": "Employed",
            "years_at_current_job": 5.0,
            "payment_history": "Good",
            "debt_to_income_ratio": 0.25,
            "assets_value": 150000.0,
            "number_of_dependents": 2,
            "previous_defaults": 0.0,
            "marital_status_change": 0
        }
    
    def test_predict_validation_error_on_missing_field(self):
        """Should return 422 if required field is missing."""
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={"age": 35})
        assert response.status_code == 422
        
    def test_predict_validation_error_on_invalid_age(self):
        """Should return 422 if age is out of range."""
        from src.api.main import app
        client = TestClient(app)
        response = client.post("/predict", json={
            "age": 150,  # Invalid
            "gender": "Male",
            "education_level": "Bachelor's",
            "marital_status": "Single",
            "income": 50000,
            "credit_score": 700,
            "loan_amount": 10000,
            "loan_purpose": "Auto",
            "employment_status": "Employed",
            "years_at_current_job": 2,
            "payment_history": "Good",
            "debt_to_income_ratio": 0.3,
            "assets_value": 50000,
            "number_of_dependents": 0,
            "previous_defaults": 0,
            "marital_status_change": 0
        })
        assert response.status_code == 422
