"""
Unit tests for Model Predictor
"""

import pytest
import pandas as pd
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent))


def test_predictor_initialization():
    """Test predictor initialization (requires trained model)."""
    try:
        from src.models.predict import LoanPredictor
        predictor = LoanPredictor()
        assert predictor is not None
        assert predictor.model is not None
    except FileNotFoundError:
        pytest.skip("Model not trained yet")


def test_validate_input():
    """Test input validation."""
    try:
        from src.models.predict import LoanPredictor
        predictor = LoanPredictor()
        
        valid_input = {'age': 35, 'income': 50000, 'savings': 15000}
        assert predictor.validate_input(valid_input) == True
        
        with pytest.raises(ValueError):
            predictor.validate_input({'age': 35, 'income': 50000})
        
        with pytest.raises(ValueError):
            predictor.validate_input({'age': 15, 'income': 50000, 'savings': 15000})
        
        with pytest.raises(ValueError):
            predictor.validate_input({'age': 35, 'income': -1000, 'savings': 15000})
    
    except FileNotFoundError:
        pytest.skip("Model not trained yet")


def test_predict():
    """Test prediction functionality."""
    try:
        from src.models.predict import LoanPredictor
        predictor = LoanPredictor()
        
        result = predictor.predict(age=35, income=50000, savings=15000)
        
        assert 'approved' in result
        assert 'approval_status' in result
        assert 'probability' in result
        assert 'confidence' in result
        assert result['approved'] in [0, 1]
        assert 0 <= result['probability'] <= 1
        assert 0 <= result['confidence'] <= 1
    
    except FileNotFoundError:
        pytest.skip("Model not trained yet")


def test_predict_batch():
    """Test batch prediction functionality."""
    try:
        from src.models.predict import LoanPredictor
        predictor = LoanPredictor()
        
        batch_data = [
            {'age': 35, 'income': 50000, 'savings': 15000},
            {'age': 25, 'income': 30000, 'savings': 5000}
        ]
        
        results = predictor.predict_batch(batch_data)
        
        assert len(results) == 2
        assert all('approved' in r for r in results)
        assert all('probability' in r for r in results)
    
    except FileNotFoundError:
        pytest.skip("Model not trained yet")
