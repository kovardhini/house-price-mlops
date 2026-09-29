"""
test_preprocessing.py

Tests for the saved preprocessing pipeline.
"""

import sys
import os
import pandas as pd
import joblib
import pytest

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


@pytest.fixture
def preprocessor():
    return joblib.load(os.path.join(PROJECT_ROOT, "models", "preprocessor.pkl"))


@pytest.fixture
def sample_train_row():
    X_train = pd.read_csv(os.path.join(PROJECT_ROOT, "data", "processed", "X_train.csv"))
    return X_train.iloc[[0]]


def test_preprocessor_loads(preprocessor):
    """The saved preprocessor should load without error."""
    assert preprocessor is not None


def test_preprocessor_transform_shape(preprocessor, sample_train_row):
    """Transforming a single row should produce a 2D array with 1 row."""
    result = preprocessor.transform(sample_train_row)
    assert result.shape[0] == 1


def test_preprocessor_output_is_numeric(preprocessor, sample_train_row):
    """After preprocessing, output should be entirely numeric (no strings/NaN)."""
    result = preprocessor.transform(sample_train_row)
    assert not pd.isnull(result).any()