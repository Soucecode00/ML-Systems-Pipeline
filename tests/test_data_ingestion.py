"""
Unit tests for Data Ingestion module
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))
from src.data.data_ingestion import DataIngestion


@pytest.fixture
def data_ingestion():
    """Fixture for DataIngestion instance."""
    return DataIngestion()


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    return pd.DataFrame({
        'age': [25, 35, 45, 55],
        'income': [30000, 50000, 70000, 90000],
        'savings': [5000, 15000, 25000, 35000],
        'approved': [0, 1, 1, 1]
    })


def test_data_ingestion_initialization(data_ingestion):
    """Test DataIngestion initialization."""
    assert data_ingestion is not None
    assert data_ingestion.logger is not None
    assert data_ingestion.config is not None


def test_validate_data_valid(data_ingestion, sample_data):
    """Test data validation with valid data."""
    is_valid, errors = data_ingestion.validate_data(sample_data)
    assert is_valid or len(errors) == 0


def test_validate_data_missing_columns(data_ingestion):
    """Test data validation with missing columns."""
    invalid_data = pd.DataFrame({'age': [25, 35]})
    is_valid, errors = data_ingestion.validate_data(invalid_data)
    assert not is_valid
    assert len(errors) > 0


def test_get_data_info(data_ingestion, sample_data):
    """Test getting data information."""
    info = data_ingestion.get_data_info(sample_data)
    assert info['shape'] == (4, 4)
    assert len(info['columns']) == 4
    assert 'age' in info['columns']
    assert 'approved' in info['columns']
