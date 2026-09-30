"""
predict.py

Loads the final house-price pipeline and predicts a sale price
from raw property features.
"""

from pathlib import Path
from numbers import Real

import joblib
import numpy as np
import pandas as pd


# Project root:
# house-price-mlops/
# ├── src/predict.py
# └── models/house_price_pipeline.pkl
PROJECT_ROOT = Path(__file__).resolve().parents[1]

PIPELINE_PATH = (
    PROJECT_ROOT
    / "models"
    / "house_price_pipeline.pkl"
)

if not PIPELINE_PATH.exists():
    raise FileNotFoundError(
        f"Final model not found at:\n{PIPELINE_PATH}\n"
        "Run 16_model_saving.ipynb first."
    )


# Load the complete preprocessing + Ridge pipeline once.
_pipeline = joblib.load(PIPELINE_PATH)


def _get_expected_columns():
    """
    Get the raw feature columns expected by the trained preprocessor.
    """
    try:
        preprocessor = _pipeline.named_steps["preprocessor"]
        return list(preprocessor.feature_names_in_)
    except Exception:
        return None


def validate_input(features: dict) -> None:
    """
    Validate raw input before prediction.
    """

    if not isinstance(features, dict):
        raise TypeError(
            "features must be provided as a dictionary."
        )

    required_fields = [
        "GrLivArea",
        "LotArea",
        "OverallQual",
        "YearBuilt",
    ]

    missing_required = [
        field
        for field in required_fields
        if field not in features
    ]

    if missing_required:
        raise ValueError(
            f"Missing required fields: {missing_required}"
        )

    numeric_positive_fields = [
        "GrLivArea",
        "LotArea",
        "OverallQual",
    ]

    for field in numeric_positive_fields:
        value = features[field]

        if not isinstance(value, Real):
            raise ValueError(
                f"'{field}' must be numeric."
            )

        if not np.isfinite(float(value)):
            raise ValueError(
                f"'{field}' must be finite."
            )

        if value <= 0:
            raise ValueError(
                f"'{field}' must be positive."
            )

    overall_quality = features["OverallQual"]

    if not 1 <= overall_quality <= 10:
        raise ValueError(
            "'OverallQual' must be between 1 and 10."
        )

    year_built = features["YearBuilt"]

    if not 1800 <= year_built <= 2026:
        raise ValueError(
            f"'YearBuilt' looks invalid: {year_built}"
        )

    if "YrSold" in features:
        year_sold = features["YrSold"]

        if not 1900 <= year_sold <= 2026:
            raise ValueError(
                f"'YrSold' looks invalid: {year_sold}"
            )

    # Confirm that all columns expected by the trained pipeline exist.
    expected_columns = _get_expected_columns()

    if expected_columns is not None:
        missing_columns = [
            column
            for column in expected_columns
            if column not in features
        ]

        if missing_columns:
            raise ValueError(
                "Prediction input is missing these trained features:\n"
                + ", ".join(missing_columns)
            )


def predict_price(features: dict) -> float:
    """
    Predict the house sale price in dollars.

    Parameters
    ----------
    features : dict
        Raw property features including the engineered features.

    Returns
    -------
    float
        Predicted sale price in dollars.
    """

    validate_input(features)

    # The trained pipeline expects one row in DataFrame format.
    input_df = pd.DataFrame([features])

    prediction_log = float(
        _pipeline.predict(input_df)[0]
    )

    if not np.isfinite(prediction_log):
        raise ValueError(
            "The model returned an invalid prediction."
        )

    # The model was trained on log1p(SalePrice).
    prediction_dollar = np.expm1(prediction_log)

    return round(float(prediction_dollar), 2)


if __name__ == "__main__":
    print("House-price prediction pipeline loaded successfully.")
    print(f"Pipeline path: {PIPELINE_PATH}")
    print("Expected raw columns:")

    expected_columns = _get_expected_columns()

    if expected_columns is not None:
        print(f"{len(expected_columns)} columns expected.")
    else:
        print("Could not read expected column names.")