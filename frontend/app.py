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
theme = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="slate",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Poppins"), "sans-serif"],
).set(
    button_primary_background_fill="*primary_600",
    button_primary_background_fill_hover="*primary_700",
    block_title_text_weight="600",
)

CUSTOM_CSS = """
#header {text-align: center; padding: 10px 0 0 0;}
#header h1 {font-size: 2.2rem; margin-bottom: 0;}
#header p {color: var(--body-text-color-subdued); font-size: 1rem;}
#result-box {
    text-align: center;
    padding: 20px;
    border-radius: 16px;
    background: linear-gradient(135deg, #10b98122, #06b6d422);
    border: 1px solid #10b98155;
}
#result-box h2 {font-size: 2.4rem; margin: 0;}
.gradio-container {max-width: 1100px !important; margin: auto;}
"""

with gr.Blocks(title="House Price Predictor") as demo:

    with gr.Column(elem_id="header"):
        gr.Markdown("# 🏠 House Price Predictor")
        gr.Markdown("Estimate a home's market value from its key features — powered by a tuned Ridge regression model.")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### 📐 Size & Structure")
                living_area = gr.Slider(400, 5000, value=1500, step=50, label="Living Area (sq ft)")
                lot_area = gr.Slider(1000, 20000, value=8000, step=100, label="Lot Area (sq ft)")
                total_bsmt_sf = gr.Slider(0, 3000, value=800, step=50, label="Basement Area (sq ft)")
                year_built = gr.Slider(1900, 2010, value=2000, step=1, label="Year Built")

            with gr.Group():
                gr.Markdown("### 🛏️ Rooms")
                bedrooms = gr.Slider(0, 8, value=3, step=1, label="Bedrooms Above Ground")
                full_bath = gr.Slider(0, 4, value=2, step=1, label="Full Bathrooms")
                fireplaces = gr.Slider(0, 3, value=0, step=1, label="Fireplaces")

        with gr.Column(scale=1):
            with gr.Group():
                gr.Markdown("### ⭐ Quality")
                overall_qual = gr.Slider(1, 10, value=6, step=1, label="Overall Quality")
                kitchen_qual = gr.Dropdown(["Ex", "Gd", "TA", "Fa"], value="TA", label="Kitchen Quality")
                exter_qual = gr.Dropdown(["Ex", "Gd", "TA", "Fa"], value="TA", label="Exterior Quality")

            with gr.Group():
                gr.Markdown("### 🚗 Garage")
                garage_cars = gr.Slider(0, 4, value=2, step=1, label="Garage Capacity (cars)")
                garage_area = gr.Slider(0, 1200, value=400, step=50, label="Garage Area (sq ft)")

            with gr.Group():
                gr.Markdown("### 📍 Location & Style")
                neighborhood = gr.Dropdown(
                    ["CollgCr", "NAmes", "OldTown", "Edwards", "Somerst",
                     "Gilbert", "NridgHt", "Sawyer", "NWAmes", "SawyerW"],
                    value="CollgCr", label="Neighborhood"
                )
                house_style = gr.Dropdown(
                    ["1Story", "2Story", "1.5Fin", "SLvl", "SFoyer"],
                    value="1Story", label="House Style"
                )
                bldg_type = gr.Dropdown(
                    ["1Fam", "TwnhsE", "Twnhs", "Duplex", "2fmCon"],
                    value="1Fam", label="Building Type"
                )

    predict_btn = gr.Button("🔮 Predict Price", variant="primary", size="lg")

    with gr.Group(elem_id="result-box", visible=False) as result_box:
        result_output = gr.Markdown("## $0")

    predict_btn.click(
        fn=predict_from_ui,
        inputs=[living_area, overall_qual, year_built, garage_cars, garage_area,
                total_bsmt_sf, bedrooms, full_bath, neighborhood, house_style,
                bldg_type, kitchen_qual, exter_qual, lot_area, fireplaces],
        outputs=[result_output, result_box]
    )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(theme=theme, css=CUSTOM_CSS, server_name="0.0.0.0", server_port=port)