# 🏠 House Price Prediction — MLOps Project

An end-to-end machine learning system that predicts residential property sale prices, built with a full MLOps pipeline: data versioning, experiment tracking, model serving, containerization, and a web interface.

## Overview

This project predicts house sale prices using the Ames Housing dataset (via Kaggle's "House Prices - Advanced Regression Techniques" competition). It walks through the complete ML lifecycle — from raw data to a deployed, containerized prediction API — with an emphasis on reproducibility and sound experimental practice.

**Final model performance** (on a held-out test set, never used during development):
- MAE: $13,908
- RMSE: $19,155
- R²: 0.933

## Tech Stack

- **Modeling**: scikit-learn (Ridge Regression), XGBoost
- **Experiment Tracking**: MLflow
- **Data Versioning**: DVC
- **API**: FastAPI
- **Frontend**: Gradio
- **Containerization**: Docker
- **Version Control**: Git + DVC

## Project Structure
house-price-mlops/
├── data/ # Raw and processed data (DVC-tracked)
├── notebooks/ # Step-by-step development notebooks (01-15)
├── src/ # Production prediction and training scripts
├── api/ # FastAPI backend
├── frontend/ # Gradio UI
├── models/ # Saved pipeline and model artifacts (DVC-tracked)
├── Dockerfile # Container definition for the API
└── PROJECT_LOG.md # Running log of experiments, decisions, and results

## How to Run

### Setup
```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Get the data (DVC)
```bash
dvc pull
```

### Run the API locally
```bash
uvicorn api.main:app --reload
```
Visit `http://localhost:8000/docs` for interactive API documentation.

### Run with Docker
```bash
docker build -t house-price-api .
docker run -p 8000:8000 house-price-api
```

### Run the frontend
```bash
python3 frontend/app.py
```

### View experiment tracking
```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```
Visit `http://localhost:5000`.

## Methodology

The full development process — data cleaning, EDA, missing value strategy, outlier analysis, feature engineering, model comparison, hyperparameter tuning, feature selection, and error analysis — is documented step by step in the `notebooks/` folder (01 through 15) and summarized in `PROJECT_LOG.md`.

Key decisions:
- **Ridge Regression** was selected over Random Forest, XGBoost, CatBoost, and HistGradientBoosting after cross-validated comparison, due to competitive accuracy combined with speed, interpretability, and low resource requirements.
- **Log-transformation** of the target variable was used to address right-skew in sale prices.
- **13 engineered features** (e.g., TotalSF, QualityLivingArea, HouseAge) were validated empirically rather than assumed to help.
- The test set was held out from all development decisions and used exactly once for final evaluation.

## Dataset

Ames Housing Dataset, via Kaggle: [House Prices - Advanced Regression Techniques](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)