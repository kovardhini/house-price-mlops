"""
predict.py

Loads the saved house price prediction pipeline and provides a function
to predict the sale price of a house from raw property features.
"""

import os
import numpy as np
import pandas as pd
import joblib

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PIPELINE_PATH = os.path.join(PROJECT_ROOT, "models", "house_price_pipeline.pkl")

# Load once, at import time - not on every prediction call
_pipeline = joblib.load(PIPELINE_PATH)


def validate_input(features: dict) -> None:
    """
    Basic sanity checks on raw input before prediction.
    Raises ValueError with a clear message if something looks wrong.
    """
    required_numeric_positive = ["GrLivArea", "LotArea", "OverallQual"]

    for field in required_numeric_positive:
        if field not in features:
            raise ValueError(f"Missing required field: '{field}'")
        value = features[field]
        if not isinstance(value, (int, float)):
            raise ValueError(f"'{field}' must be numeric, got {type(value).__name__}")
        if value <= 0:
            raise ValueError(f"'{field}' must be positive, got {value}")

    if "OverallQual" in features and not (1 <= features["OverallQual"] <= 10):
        raise ValueError(f"'OverallQual' must be between 1 and 10, got {features['OverallQual']}")

    if "YearBuilt" in features:
        if features["YearBuilt"] < 1800 or features["YearBuilt"] > 2026:
            raise ValueError(f"'YearBuilt' looks invalid: {features['YearBuilt']}")


def predict_price(features: dict) -> float:
    """
    Predict the sale price of a house given its raw features.

    Parameters
    ----------
    features : dict
        Raw property features, e.g.
        {"GrLivArea": 1500, "OverallQual": 7, "YearBuilt": 2005, ...}
        Must include every column the pipeline's preprocessor expects.

    Returns
    -------
    float
        Predicted sale price in dollars.
    """
    validate_input(features)

    # Pipeline expects a DataFrame with the same columns as training data
    input_df = pd.DataFrame([features])

    prediction_log = _pipeline.predict(input_df)[0]
    prediction_dollar = np.expm1(prediction_log)

    return round(float(prediction_dollar), 2)


if __name__ == "__main__":
    # Quick manual test
    sample_house = {
        "MSSubClass": "60",
        "LotFrontage": 65,
        "LotArea": 8450,
        "OverallQual": 7,
        "OverallCond": 5,
        "YearBuilt": 2003,
        "YearRemodAdd": 2003,
        "MasVnrArea": 196,
        "BsmtFinSF1": 706,
        "BsmtFinSF2": 0,
        "BsmtUnfSF": 150,
        "TotalBsmtSF": 856,
        "1stFlrSF": 856,
        "2ndFlrSF": 854,
        "LowQualFinSF": 0,
        "GrLivArea": 1710,
        "BsmtFullBath": 1,
        "BsmtHalfBath": 0,
        "FullBath": 2,
        "HalfBath": 1,
        "BedroomAbvGr": 3,
        "KitchenAbvGr": 1,
        "TotRmsAbvGrd": 8,
        "Fireplaces": 0,
        "GarageYrBlt": 2003,
        "GarageCars": 2,
        "GarageArea": 548,
        "WoodDeckSF": 0,
        "OpenPorchSF": 61,
        "EnclosedPorch": 0,
        "3SsnPorch": 0,
        "ScreenPorch": 0,
        "PoolArea": 0,
        "MiscVal": 0,
        "MoSold": 2,
        "YrSold": 2008,
        # engineered features (must be provided since pipeline doesn't compute them)
        "TotalSF": 856 + 856 + 854,
        "TotalBathrooms": 2 + 0.5 * 1 + 1 + 0,
        "HouseAge": 2008 - 2003,
        "RemodAge": 2008 - 2003,
        "GarageAge": 2008 - 2003,
        "TotalPorchArea": 61,
        "TotalQualityScore": 7 * 5,
        "QualityLivingArea": 7 * 1710,
        "HasPool": 0,
        "HasGarage": 1,
        "HasFireplace": 0,
        "Has2ndFloor": 1,
        "HasBsmt": 1,
        # categorical features
        "MSZoning": "RL", "Street": "Pave", "Alley": "None", "LotShape": "Reg",
        "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside",
        "LandSlope": "Gtl", "Neighborhood": "CollgCr", "Condition1": "Norm",
        "Condition2": "Norm", "BldgType": "1Fam", "HouseStyle": "2Story",
        "RoofStyle": "Gable", "RoofMatl": "CompShg", "Exterior1st": "VinylSd",
        "Exterior2nd": "VinylSd", "MasVnrType": "BrkFace", "ExterQual": "Gd",
        "ExterCond": "TA", "Foundation": "PConc", "BsmtQual": "Gd",
        "BsmtCond": "TA", "BsmtExposure": "No", "BsmtFinType1": "GLQ",
        "BsmtFinType2": "Unf", "Heating": "GasA", "HeatingQC": "Ex",
        "CentralAir": "Y", "Electrical": "SBrkr", "KitchenQual": "Gd",
        "Functional": "Typ", "FireplaceQu": "None", "GarageType": "Attchd",
        "GarageFinish": "RFn", "GarageQual": "TA", "GarageCond": "TA",
        "PavedDrive": "Y", "PoolQC": "None", "Fence": "None",
        "MiscFeature": "None", "SaleType": "WD", "SaleCondition": "Normal",
    }

    price = predict_price(sample_house)
    print(f"Predicted sale price: ${price:,.2f}")