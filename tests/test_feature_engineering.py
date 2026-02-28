"""
Unit tests for Feature Engineering module
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.features.feature_engineering import FeatureEngineer


@pytest.fixture
def feature_engineer():
    """Fixture for FeatureEngineer instance."""
    return FeatureEngineer()


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    return pd.DataFrame({
        'age': [25, 35, 45, 55, 30, 40],
        'income': [30000, 50000, 70000, 90000, 40000, 60000],
        'savings': [5000, 15000, 25000, 35000, 8000, 20000],
        'approved': [0, 1, 1, 1, 0, 1]
    })


def test_feature_engineer_initialization(feature_engineer):
    """Test FeatureEngineer initialization."""
    assert feature_engineer is not None
    assert feature_engineer.logger is not None
    assert feature_engineer.scaler is not None


def test_handle_missing_values(feature_engineer):
    """Test handling missing values."""
    data_with_missing = pd.DataFrame({
        'age': [25, None, 45],
        'income': [30000, 50000, None],
        'savings': [5000, 15000, 25000],
        'approved': [0, 1, 1]
    })
    
    cleaned_data = feature_engineer.handle_missing_values(data_with_missing)
    assert cleaned_data.isnull().sum().sum() == 0


def test_create_features(feature_engineer, sample_data):
    """Test feature creation."""
    df_with_features = feature_engineer.create_features(sample_data)
    
    assert 'savings_to_income_ratio' in df_with_features.columns
    assert 'age_group_encoded' in df_with_features.columns
    assert 'income_category_encoded' in df_with_features.columns


def test_scale_features(feature_engineer, sample_data):
    """Test feature scaling."""
    X = sample_data[['age', 'income', 'savings']]
    X_train = X.iloc[:4]
    X_test = X.iloc[4:]
    
    X_train_scaled, X_test_scaled = feature_engineer.scale_features(X_train, X_test, fit=True)
    
    assert X_train_scaled.shape == X_train.shape
    assert X_test_scaled.shape == X_test.shape
    assert np.abs(X_train_scaled.mean().mean()) < 0.1


def test_split_data(feature_engineer, sample_data):
    """Test data splitting."""
    df_with_features = feature_engineer.create_features(sample_data)
    X_train, X_test, y_train, y_test = feature_engineer.split_data(df_with_features)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
    assert len(X_train) + len(X_test) == len(sample_data)
    assert len(y_train) == len(X_train)
    assert len(y_test) == len(X_test)
