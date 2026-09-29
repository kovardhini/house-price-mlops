"""
main.py

FastAPI backend serving the house price prediction pipeline.
"""

import sys
import os

# Allow importing from src/ (one level up, then into src)
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional

from predict import predict_price

app = FastAPI(
    title="House Price Prediction API",
    description="Predicts residential property sale price from structural, quality, and location features.",
    version="1.0.0"
)


class HouseFeatures(BaseModel):
    """
    Raw property features required for a prediction.
    Pydantic automatically validates types and required fields.
    """
    MSSubClass: str
    LotFrontage: float
    LotArea: float
    OverallQual: int = Field(..., ge=1, le=10)
    OverallCond: int = Field(..., ge=1, le=10)
    YearBuilt: int
    YearRemodAdd: int
    MasVnrArea: float = 0
    BsmtFinSF1: float = 0
    BsmtFinSF2: float = 0
    BsmtUnfSF: float = 0
    TotalBsmtSF: float = 0
    FirstFlrSF: float = Field(..., alias="1stFlrSF")
    SecondFlrSF: float = Field(0, alias="2ndFlrSF")
    LowQualFinSF: float = 0
    GrLivArea: float = Field(..., gt=0)
    BsmtFullBath: int = 0
    BsmtHalfBath: int = 0
    FullBath: int
    HalfBath: int = 0
    BedroomAbvGr: int
    KitchenAbvGr: int = 1
    TotRmsAbvGrd: int
    Fireplaces: int = 0
    GarageYrBlt: Optional[float] = 0
    GarageCars: int = 0
    GarageArea: float = 0
    WoodDeckSF: float = 0
    OpenPorchSF: float = 0
    EnclosedPorch: float = 0
    ThreeSsnPorch: float = Field(0, alias="3SsnPorch")
    ScreenPorch: float = 0
    PoolArea: float = 0
    MiscVal: float = 0
    MoSold: int
    YrSold: int
    MSZoning: str
    Street: str = "Pave"
    Alley: str = "None"
    LotShape: str
    LandContour: str
    Utilities: str = "AllPub"
    LotConfig: str
    LandSlope: str
    Neighborhood: str
    Condition1: str = "Norm"
    Condition2: str = "Norm"
    BldgType: str
    HouseStyle: str
    RoofStyle: str
    RoofMatl: str = "CompShg"
    Exterior1st: str
    Exterior2nd: str
    MasVnrType: str = "None"
    ExterQual: str
    ExterCond: str
    Foundation: str
    BsmtQual: str = "None"
    BsmtCond: str = "None"
    BsmtExposure: str = "None"
    BsmtFinType1: str = "None"
    BsmtFinType2: str = "None"
    Heating: str = "GasA"
    HeatingQC: str
    CentralAir: str = "Y"
    Electrical: str = "SBrkr"
    KitchenQual: str
    Functional: str = "Typ"
    FireplaceQu: str = "None"
    GarageType: str = "None"
    GarageFinish: str = "None"
    GarageQual: str = "None"
    GarageCond: str = "None"
    PavedDrive: str = "Y"
    PoolQC: str = "None"
    Fence: str = "None"
    MiscFeature: str = "None"
    SaleType: str = "WD"
    SaleCondition: str = "Normal"

    class Config:
        populate_by_name = True


def engineer_features(raw: dict) -> dict:
    """Compute the 13 engineered features from raw input, same logic as Phase 8."""
    features = dict(raw)

    features["1stFlrSF"] = raw.get("1stFlrSF", raw.get("FirstFlrSF", 0))
    features["2ndFlrSF"] = raw.get("2ndFlrSF", raw.get("SecondFlrSF", 0))
    features["3SsnPorch"] = raw.get("3SsnPorch", raw.get("ThreeSsnPorch", 0))

    features["TotalSF"] = features["TotalBsmtSF"] + features["1stFlrSF"] + features["2ndFlrSF"]
    features["TotalBathrooms"] = (features["FullBath"] + 0.5 * features["HalfBath"] +
                                    features["BsmtFullBath"] + 0.5 * features["BsmtHalfBath"])
    features["HouseAge"] = features["YrSold"] - features["YearBuilt"]
    features["RemodAge"] = features["YrSold"] - features["YearRemodAdd"]
    garage_yr = features.get("GarageYrBlt", 0) or 0
    features["GarageAge"] = 0 if garage_yr == 0 else features["YrSold"] - garage_yr
    features["TotalPorchArea"] = (features["OpenPorchSF"] + features["EnclosedPorch"] +
                                    features["3SsnPorch"] + features["ScreenPorch"] +
                                    features.get("WoodDeckSF", 0))
    features["TotalQualityScore"] = features["OverallQual"] * features["OverallCond"]
    features["QualityLivingArea"] = features["OverallQual"] * features["GrLivArea"]
    features["HasPool"] = int(features.get("PoolArea", 0) > 0)
    features["HasGarage"] = int(features.get("GarageArea", 0) > 0)
    features["HasFireplace"] = int(features.get("Fireplaces", 0) > 0)
    features["Has2ndFloor"] = int(features["2ndFlrSF"] > 0)
    features["HasBsmt"] = int(features["TotalBsmtSF"] > 0)

    return features


@app.get("/health")
def health_check():
    """Simple health check endpoint."""
    return {"status": "ok", "message": "House Price Prediction API is running."}


@app.post("/predict")
def predict(house: HouseFeatures):
    """Predict the sale price of a house from its features."""
    try:
        raw_dict = house.dict(by_alias=True)
        full_features = engineer_features(raw_dict)
        price = predict_price(full_features)
        return {"predicted_price": price}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/")
def root():
    return {"message": "House Price Prediction API. Visit /docs to try it out."}