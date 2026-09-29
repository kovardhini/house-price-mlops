"""
test_api.py

Tests for the FastAPI endpoints.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "api"))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_health_endpoint():
    """GET /health should return status ok."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_root_endpoint():
    """GET / should return a welcome message."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_predict_endpoint_valid_input():
    """POST /predict with valid data should return a predicted_price."""
    payload = {
        "MSSubClass": "60", "LotFrontage": 65, "LotArea": 8450,
        "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003,
        "YearRemodAdd": 2003, "TotalBsmtSF": 856, "1stFlrSF": 856,
        "2ndFlrSF": 854, "GrLivArea": 1710, "FullBath": 2,
        "BedroomAbvGr": 3, "TotRmsAbvGrd": 8, "MoSold": 2, "YrSold": 2008,
        "MSZoning": "RL", "LotShape": "Reg", "LandContour": "Lvl",
        "LotConfig": "Inside", "LandSlope": "Gtl", "Neighborhood": "CollgCr",
        "BldgType": "1Fam", "HouseStyle": "2Story", "RoofStyle": "Gable",
        "Exterior1st": "VinylSd", "Exterior2nd": "VinylSd",
        "ExterQual": "Gd", "ExterCond": "TA", "Foundation": "PConc",
        "HeatingQC": "Ex", "KitchenQual": "Gd",
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price" in response.json()
    assert response.json()["predicted_price"] > 0


def test_predict_endpoint_invalid_quality():
    """POST /predict with OverallQual out of range should return 422 (Pydantic validation error)."""
    payload = {"OverallQual": 15}  # missing most required fields too
    response = client.post("/predict", json=payload)
    assert response.status_code == 422