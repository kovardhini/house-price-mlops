"""
train_with_mlflow.py

Retrains the final Ridge model on the full dataset, logging parameters,
metrics, and the model artifact to MLflow for experiment tracking.
"""

import os
import time
import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Point MLflow at a local tracking directory (no cloud service, fully free/local)
mlflow.set_tracking_uri(f"sqlite:///{PROJECT_ROOT}/mlflow.db")
mlflow.set_experiment("house-price-prediction")


def load_data():
    X_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/X_test.csv"))
    y_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/y_train.csv")).squeeze()
    y_test = pd.read_csv(os.path.join(PROJECT_ROOT, "data/processed/y_test.csv")).squeeze()
    return X_train, X_test, y_train, y_test


def train_and_log(alpha: float = 10.0):
    X_train, X_test, y_train, y_test = load_data()
    preprocessor = joblib.load(os.path.join(PROJECT_ROOT, "models/preprocessor.pkl"))

    X_train_processed = preprocessor.transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    with mlflow.start_run(run_name=f"ridge_alpha_{alpha}"):
        # --- Log hyperparameters ---
        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("n_features", X_train_processed.shape[1])
        mlflow.log_param("n_train_samples", X_train.shape[0])

        # --- Train, timing it ---
        start = time.time()
        model = Ridge(alpha=alpha, random_state=42)
        model.fit(X_train_processed, y_train)
        training_duration = time.time() - start

        # --- Predict and evaluate on the test set ---
        y_pred_log = model.predict(X_test_processed)
        y_pred_dollar = np.expm1(y_pred_log)
        y_test_dollar = np.expm1(y_test)

        mae = mean_absolute_error(y_test_dollar, y_pred_dollar)
        rmse = np.sqrt(mean_squared_error(y_test_dollar, y_pred_dollar))
        r2 = r2_score(y_test_dollar, y_pred_dollar)
        rmsle = np.sqrt(mean_squared_error(y_test, y_pred_log))

        # --- Log metrics ---
        mlflow.log_metric("MAE", mae)
        mlflow.log_metric("RMSE", rmse)
        mlflow.log_metric("R2", r2)
        mlflow.log_metric("RMSLE", rmsle)
        mlflow.log_metric("training_duration_sec", training_duration)

        # --- Log the model artifact itself ---
        mlflow.sklearn.log_model(model, "model")

        print(f"Run logged - alpha={alpha}: MAE=${mae:,.0f}, RMSE=${rmse:,.0f}, R2={r2:.4f}")

        return mae, rmse, r2, rmsle


if __name__ == "__main__":
    # Log a few different alpha values so there's something meaningful to compare in the UI
    for alpha in [1.0, 5.0, 10.0, 20.0]:
        train_and_log(alpha=alpha)