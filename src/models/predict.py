"""
Prediction/Inference Module

This module handles making predictions on new data using trained models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Union, List
import joblib
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.features.feature_engineering import FeatureEngineer


class LoanPredictor:
    """
    Handles predictions for loan applications.
    
    This class is responsible for:
    - Loading trained model and preprocessing artifacts
    - Preprocessing input data
    - Making predictions
    - Providing prediction probabilities
    """
    
    def __init__(self):
        """Initialize LoanPredictor with logger and configuration."""
        self.logger = setup_logger(__name__, "prediction")
        self.config = load_config("model_config")
        self.model = None
        self.feature_engineer = None
        self.load_artifacts()
        self.logger.info("LoanPredictor initialized")
    
    def load_artifacts(self):
        """Load trained model and preprocessing artifacts."""
        self.logger.info("Loading model and preprocessing artifacts...")
        
        model_path = self.config['model_path']
        if not Path(model_path).exists():
            self.logger.error(f"Model not found at: {model_path}")
            raise FileNotFoundError(
                f"Model not found at: {model_path}. Please train the model first."
            )
        
        self.model = joblib.load(model_path)
        self.logger.info(f"Model loaded from: {model_path}")
        
        self.feature_engineer = FeatureEngineer()
        try:
            self.feature_engineer.load_scaler()
        except FileNotFoundError:
            self.logger.warning("Scaler not found. Will use unscaled features.")
        
        self.logger.info("All artifacts loaded successfully")
    
    def validate_input(self, data: Dict[str, Any]) -> bool:
        """
        Validate input data.
        
        Args:
            data: Dictionary with input features
            
        Returns:
            True if valid, raises ValueError if invalid
        """
        required_features = ['age', 'income', 'savings']
        
        for feature in required_features:
            if feature not in data:
                raise ValueError(f"Missing required feature: {feature}")
        
        age_range = [18, 100]
        if not (age_range[0] <= data['age'] <= age_range[1]):
            raise ValueError(f"Age must be between {age_range[0]} and {age_range[1]}")
        
        if data['income'] < 0:
            raise ValueError("Income cannot be negative")
        
        if data['savings'] < 0:
            raise ValueError("Savings cannot be negative")
        
        return True
    
    def preprocess_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """
        Preprocess input data for prediction.
        
        Args:
            data: Input data as dictionary or DataFrame
            
        Returns:
            Preprocessed DataFrame ready for prediction
        """
        if isinstance(data, dict):
            df = pd.DataFrame([data])
        else:
            df = data.copy()
        
        self.logger.info(f"Preprocessing input data with shape: {df.shape}")
        
        df = self.feature_engineer.create_features(df)
        
        feature_cols = self.config['features']['input_features']
        if 'savings_to_income_ratio' in df.columns:
            feature_cols = feature_cols + ['savings_to_income_ratio']
        if 'age_group_encoded' in df.columns:
            feature_cols = feature_cols + ['age_group_encoded']
        if 'income_category_encoded' in df.columns:
            feature_cols = feature_cols + ['income_category_encoded']
        
        X = df[feature_cols]
        
        try:
            X_scaled, _ = self.feature_engineer.scale_features(X, fit=False)
            return X_scaled
        except Exception as e:
            self.logger.warning(f"Scaling failed: {e}. Using unscaled features.")
            return X
    
    def predict(self, age: int, income: float, savings: float) -> Dict[str, Any]:
        """
        Make a prediction for a single loan application.
        
        Args:
            age: Applicant's age
            income: Applicant's annual income
            savings: Applicant's savings amount
            
        Returns:
            Dictionary with prediction results
        """
        self.logger.info(f"Making prediction for: age={age}, income={income}, savings={savings}")
        
        input_data = {
            'age': age,
            'income': income,
            'savings': savings
        }
        
        self.validate_input(input_data)
        
        X = self.preprocess_input(input_data)
        
        prediction = self.model.predict(X)[0]
        probability = self.model.predict_proba(X)[0]
        
        result = {
            'approved': int(prediction),
            'approval_status': 'Approved' if prediction == 1 else 'Rejected',
            'probability': float(probability[1]),
            'confidence': float(max(probability)),
            'input_data': input_data,
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info(f"Prediction: {result['approval_status']} with probability {result['probability']:.4f}")
        
        return result
    
    def predict_batch(self, data: Union[List[Dict], pd.DataFrame]) -> List[Dict[str, Any]]:
        """
        Make predictions for multiple loan applications.
        
        Args:
            data: List of dictionaries or DataFrame with input features
            
        Returns:
            List of prediction results
        """
        if isinstance(data, list):
            df = pd.DataFrame(data)
        else:
            df = data.copy()
        
        self.logger.info(f"Making batch predictions for {len(df)} applications")
        
        X = self.preprocess_input(df)
        
        predictions = self.model.predict(X)
        probabilities = self.model.predict_proba(X)
        
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            result = {
                'application_id': i,
                'approved': int(pred),
                'approval_status': 'Approved' if pred == 1 else 'Rejected',
                'probability': float(prob[1]),
                'confidence': float(max(prob)),
                'timestamp': datetime.now().isoformat()
            }
            results.append(result)
        
        self.logger.info(f"Batch predictions completed: {sum(predictions)} approved, {len(predictions) - sum(predictions)} rejected")
        
        return results
    
    def explain_prediction(self, age: int, income: float, savings: float) -> Dict[str, Any]:
        """
        Provide an explanation for the prediction.
        
        Args:
            age: Applicant's age
            income: Applicant's annual income
            savings: Applicant's savings amount
            
        Returns:
            Dictionary with prediction and explanation
        """
        result = self.predict(age, income, savings)
        
        explanation = {
            'prediction': result,
            'factors': []
        }
        
        if income < 30000:
            explanation['factors'].append("Low income may reduce approval chances")
        elif income > 60000:
            explanation['factors'].append("Good income increases approval chances")
        
        if savings < 5000:
            explanation['factors'].append("Low savings may be a concern")
        elif savings > 15000:
            explanation['factors'].append("Strong savings profile")
        
        savings_ratio = savings / (income + 1)
        if savings_ratio > 0.3:
            explanation['factors'].append("Excellent savings-to-income ratio")
        elif savings_ratio < 0.1:
            explanation['factors'].append("Low savings-to-income ratio")
        
        if age < 25:
            explanation['factors'].append("Young applicant - limited credit history")
        elif 25 <= age <= 50:
            explanation['factors'].append("Prime age range for loan approval")
        
        return explanation


if __name__ == "__main__":
    predictor = LoanPredictor()
    
    print("\n" + "="*60)
    print("LOAN PREDICTION SYSTEM - EXAMPLES")
    print("="*60)
    
    print("\n1. Single Prediction:")
    result1 = predictor.predict(age=35, income=50000, savings=15000)
    print(f"   Applicant: Age=35, Income=$50,000, Savings=$15,000")
    print(f"   Result: {result1['approval_status']}")
    print(f"   Probability: {result1['probability']:.2%}")
    
    print("\n2. Another Prediction:")
    result2 = predictor.predict(age=22, income=25000, savings=2000)
    print(f"   Applicant: Age=22, Income=$25,000, Savings=$2,000")
    print(f"   Result: {result2['approval_status']}")
    print(f"   Probability: {result2['probability']:.2%}")
    
    print("\n3. Batch Predictions:")
    batch_data = [
        {'age': 40, 'income': 70000, 'savings': 25000},
        {'age': 28, 'income': 45000, 'savings': 8000},
        {'age': 55, 'income': 80000, 'savings': 30000}
    ]
    batch_results = predictor.predict_batch(batch_data)
    for i, result in enumerate(batch_results, 1):
        print(f"   Application {i}: {result['approval_status']} (Prob: {result['probability']:.2%})")
    
    print("\n4. Prediction with Explanation:")
    explanation = predictor.explain_prediction(age=30, income=35000, savings=5000)
    print(f"   Result: {explanation['prediction']['approval_status']}")
    print(f"   Factors:")
    for factor in explanation['factors']:
        print(f"   - {factor}")
