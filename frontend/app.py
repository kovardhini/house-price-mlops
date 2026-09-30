"""
frontend/app.py

Gradio UI for the House Price Prediction model.
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

import gradio as gr
from predict import predict_price


DEFAULT_FEATURES = {
    "MSSubClass": "60", "LotFrontage": 65, "MasVnrArea": 0,
    "BsmtFinSF1": 0, "BsmtFinSF2": 0, "BsmtUnfSF": 0,
    "LowQualFinSF": 0, "BsmtFullBath": 0, "BsmtHalfBath": 0,
    "HalfBath": 0, "KitchenAbvGr": 1, "WoodDeckSF": 0,
    "OpenPorchSF": 0, "EnclosedPorch": 0, "3SsnPorch": 0,
    "ScreenPorch": 0, "PoolArea": 0, "MiscVal": 0, "MoSold": 6,
    "MSZoning": "RL", "Street": "Pave", "Alley": "None", "LotShape": "Reg",
    "LandContour": "Lvl", "Utilities": "AllPub", "LotConfig": "Inside",
    "LandSlope": "Gtl", "Condition1": "Norm", "Condition2": "Norm",
    "RoofStyle": "Gable", "RoofMatl": "CompShg", "MasVnrType": "None",
    "ExterCond": "TA", "BsmtQual": "TA", "BsmtCond": "TA",
    "BsmtExposure": "No", "BsmtFinType1": "Unf", "BsmtFinType2": "Unf",
    "Heating": "GasA", "Electrical": "SBrkr", "Functional": "Typ",
    "FireplaceQu": "None", "GarageType": "Attchd", "GarageFinish": "Unf",
    "GarageQual": "TA", "GarageCond": "TA", "PavedDrive": "Y",
    "PoolQC": "None", "Fence": "None", "MiscFeature": "None",
    "SaleType": "WD", "SaleCondition": "Normal",
}


def engineer_features(raw: dict) -> dict:
    """Compute the 13 engineered features from raw input, same logic as training."""
    features = dict(raw)
    features["TotalSF"] = features["TotalBsmtSF"] + features["1stFlrSF"] + features["2ndFlrSF"]
    features["TotalBathrooms"] = (features["FullBath"] + 0.5 * features["HalfBath"] +
                                    features["BsmtFullBath"] + 0.5 * features["BsmtHalfBath"])
    features["HouseAge"] = features["YrSold"] - features["YearBuilt"]
    features["RemodAge"] = features["YrSold"] - features["YearRemodAdd"]
    garage_yr = features.get("GarageYrBlt", 0) or 0
    features["GarageAge"] = 0 if garage_yr == 0 else features["YrSold"] - garage_yr
    features["TotalPorchArea"] = (features["OpenPorchSF"] + features["EnclosedPorch"] +
                                    features["3SsnPorch"] + features["ScreenPorch"] +
                                    features["WoodDeckSF"])
    features["TotalQualityScore"] = features["OverallQual"] * features["OverallCond"]
    features["QualityLivingArea"] = features["OverallQual"] * features["GrLivArea"]
    features["HasPool"] = int(features["PoolArea"] > 0)
    features["HasGarage"] = int(features["GarageArea"] > 0)
    features["HasFireplace"] = int(features["Fireplaces"] > 0)
    features["Has2ndFloor"] = int(features["2ndFlrSF"] > 0)
    features["HasBsmt"] = int(features["TotalBsmtSF"] > 0)
    return features


def predict_from_ui(
    living_area, overall_qual, year_built, garage_cars, garage_area,
    total_bsmt_sf, bedrooms, full_bath, neighborhood, house_style,
    bldg_type, kitchen_qual, exter_qual, lot_area, fireplaces,
):
    try:
        raw = dict(DEFAULT_FEATURES)
        raw.update({
            "GrLivArea": living_area,
            "OverallQual": overall_qual,
            "OverallCond": 5,
            "YearBuilt": year_built,
            "YearRemodAdd": year_built,
            "GarageYrBlt": year_built,
            "GarageCars": garage_cars,
            "GarageArea": garage_area,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": living_area if living_area <= 1500 else living_area * 0.6,
            "2ndFlrSF": 0 if living_area <= 1500 else living_area * 0.4,
            "BedroomAbvGr": bedrooms,
            "FullBath": full_bath,
            "TotRmsAbvGrd": bedrooms + 3,
            "Neighborhood": neighborhood,
            "HouseStyle": house_style,
            "BldgType": bldg_type,
            "KitchenQual": kitchen_qual,
            "ExterQual": exter_qual,
            "HeatingQC": exter_qual,
            "Foundation": "PConc",
            "Exterior1st": "VinylSd",
            "Exterior2nd": "VinylSd",
            "LotArea": lot_area,
            "Fireplaces": fireplaces,
            "CentralAir": "Y",
            "YrSold": 2010,
        })

        full_features = engineer_features(raw)
        price = predict_price(full_features)

        return f"## 🏡 ${price:,.0f}", gr.update(visible=True)

    except Exception as e:
        return f"⚠️ Error: {str(e)}", gr.update(visible=True)


# ---- Custom theme ----
# ---- Redesigned Gradio interface ----
from house_ui import build_app, launch_app

# Preserve the exact dropdown choices from your original app.
UI_CHOICES = {
    "neighborhood": [
        "CollgCr", "NAmes", "OldTown", "Edwards", "Somerst",
        "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW",
    ],
    "house_style": [
        "1Story", "2Story", "1.5Fin", "SLvl", "SFoyer",
    ],
    "bldg_type": [
        "1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon",
    ],
    "kitchen_qual": ["Ex", "Gd", "TA", "Fa"],
    "exter_qual": ["Ex", "Gd", "TA", "Fa"],
}

demo = build_app(
    predict_from_ui=predict_from_ui,
    choices=UI_CHOICES,
)

if __name__ == "__main__":
    import house_ui

    print("Running app:", os.path.abspath(__file__))
    print("Using UI:", house_ui.__file__)

    os.environ["PORT"] = "10000"  
    launch_app(demo)