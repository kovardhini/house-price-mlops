"""
test_predict.py

Tests for the prediction script - feature engineering, validation, and output.
"""

import sys
import os
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from predict import predict_price, validate_input


VALID_HOUSE = {
    "MSSubClass": "60", "LotFrontage": 65, "LotArea": 8450,
    "OverallQual": 7, "OverallCond": 5, "YearBuilt": 2003,
    "YearRemodAdd": 2003, "MasVnrArea": 196, "BsmtFinSF1": 706,
    "BsmtFinSF2": 0, "BsmtUnfSF": 150, "TotalBsmtSF": 856,
    "1stFlrSF": 856, "2ndFlrSF": 854, "LowQualFinSF": 0,
    "GrLivArea": 1710, "BsmtFullBath": 1, "BsmtHalfBath": 0,
    "FullBath": 2, "HalfBath": 1, "BedroomAbvGr": 3, "KitchenAbvGr": 1,
    "TotRmsAbvGrd": 8, "Fireplaces": 0, "GarageYrBlt": 2003,
    "GarageCars": 2, "GarageArea": 548, "WoodDeckSF": 0,
    "OpenPorchSF": 61, "EnclosedPorch": 0, "3SsnPorch": 0,
    "ScreenPorch": 0, "PoolArea": 0, "MiscVal": 0, "MoSold": 2,
    "YrSold": 2008, "TotalSF": 856 + 856 + 854,
    "TotalBathrooms": 2 + 0.5 * 1 + 1 + 0, "HouseAge": 5,
    "RemodAge": 5, "GarageAge": 5, "TotalPorchArea": 61,
    "TotalQualityScore": 35, "QualityLivingArea": 7 * 1710,
    "HasPool": 0, "HasGarage": 1, "HasFireplace": 0,
    "Has2ndFloor": 1, "HasBsmt": 1,
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


def test_predict_price_returns_number():
    """A valid house input should return a numeric prediction."""
    price = predict_price(VALID_HOUSE)
    assert isinstance(price, float)


def test_predict_price_is_positive():
    """A house price prediction should never be negative or zero."""
    price = predict_price(VALID_HOUSE)
    assert price > 0


def test_predict_price_is_reasonable_range():
    """Prediction should fall within a plausible house price range."""
    price = predict_price(VALID_HOUSE)
    assert 10_000 < price < 5_000_000


def test_validate_input_rejects_missing_field():
    """Missing a required field should raise ValueError."""
    bad_input = dict(VALID_HOUSE)
    del bad_input["GrLivArea"]
    with pytest.raises(ValueError):
        validate_input(bad_input)


def test_validate_input_rejects_negative_area():
    """Negative living area should raise ValueError."""
    bad_input = dict(VALID_HOUSE)
    bad_input["GrLivArea"] = -100
    with pytest.raises(ValueError):
        validate_input(bad_input)


def test_validate_input_rejects_invalid_quality():
    """OverallQual outside 1-10 should raise ValueError."""
    bad_input = dict(VALID_HOUSE)
    bad_input["OverallQual"] = 15
    with pytest.raises(ValueError):
        validate_input(bad_input)


def test_validate_input_accepts_valid_house():
    """A correctly formed input should not raise any error."""
    validate_input(VALID_HOUSE)  # should not 