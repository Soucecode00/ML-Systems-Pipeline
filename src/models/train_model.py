"""
Model Training Module

This module handles model training, hyperparameter tuning, and experiment tracking
using MLflow.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config
from src.data.data_ingestion import DataIngestion
from src.features.feature_engineering import FeatureEngineer


class ModelTrainer:
    """
    Handles model training and hyperparameter tuning.
    
    This class is responsible for:
    - Training machine learning models
    - Performing cross-validation
    - Hyperparameter tuning
    - Model serialization
    - Experiment tracking
    """
    
    def __init__(self):
        """Initialize ModelTrainer with logger and configuration."""
        self.logger = setup_logger(__name__, "model_training")
        self.config = load_config("model_config")
        self.model = None
        self.best_params = None
        self.logger.info("ModelTrainer initialized")
    
    def create_model(self, model_name: str = None) -> Any:
        """
        Create a machine learning model.
        
        Args:
            model_name: Name of the model to create
            
        Returns:
            Model instance
        """
        if model_name is None:
            model_name = self.config['model']['name']
        
        self.logger.info(f"Creating model: {model_name}")
        
        if model_name == "LogisticRegression":
            model = LogisticRegression(
                random_state=self.config['model']['random_state'],
                max_iter=self.config['model']['max_iter'],
                solver=self.config['model']['solver'],
                class_weight=self.config['model'].get('class_weight', None)
            )
        elif model_name == "RandomForest":
            model = RandomForestClassifier(
                random_state=self.config['model']['random_state'],
                n_estimators=100,
                max_depth=10
            )
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        return model
    
    def train_model(self, X_train: pd.DataFrame, y_train: pd.Series) -> Any:
        """
        Train the model on training data.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Trained model
        """
        self.logger.info("Training model...")
        
        self.model = self.create_model()
        
        self.model.fit(X_train, y_train)
        
        self.logger.info("Model training completed")
        
        return self.model
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
        """
        Perform cross-validation.
        
        Args:
            X: Features
            y: Target
            cv: Number of cross-validation folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        self.logger.info(f"Performing {cv}-fold cross-validation...")
        
        if self.model is None:
            self.model = self.create_model()
        
        cv_scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        
        cv_results = {
            'cv_scores': cv_scores.tolist(),
            'mean_cv_score': float(cv_scores.mean()),
            'std_cv_score': float(cv_scores.std())
        }
        
        self.logger.info(f"Cross-validation results: {cv_results['mean_cv_score']:.4f} (+/- {cv_results['std_cv_score']:.4f})")
        
        return cv_results
    
    def hyperparameter_tuning(self, X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[Any, Dict]:
        """
        Perform hyperparameter tuning using GridSearchCV.
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Tuple of (best_model, best_params)
        """
        self.logger.info("Starting hyperparameter tuning...")
        
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],
            'penalty': ['l2'],
            'solver': ['lbfgs', 'liblinear']
        }
        
        base_model = LogisticRegression(
            random_state=self.config['model']['random_state'],
            max_iter=self.config['model']['max_iter']
        )
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        self.logger.info(f"Best parameters: {self.best_params}")
        self.logger.info(f"Best cross-validation score: {grid_search.best_score_:.4f}")
        
        return self.model, self.best_params
    
    def save_model(self, path: str = None):
        """
        Save the trained model to disk.
        
        Args:
            path: Path to save model. If None, uses path from config.
        """
        if self.model is None:
            self.logger.error("No model to save. Train a model first.")
            raise ValueError("No model to save. Train a model first.")
        
        if path is None:
            path = self.config['model_path']
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to: {path}")
    
    def load_model(self, path: str = None):
        """
        Load a saved model from disk.
        
        Args:
            path: Path to load model from. If None, uses path from config.
        """
        if path is None:
            path = self.config['model_path']
        
        if not Path(path).exists():
            self.logger.error(f"Model not found at: {path}")
            raise FileNotFoundError(f"Model not found at: {path}")
        
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from: {path}")
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Get feature importance from the trained model.
        
        Args:
            feature_names: List of feature names
            
        Returns:
            DataFrame with feature importance
        """
        if self.model is None:
            self.logger.error("No trained model available")
            raise ValueError("No trained model available")
        
        if hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_[0])
        elif hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        else:
            self.logger.warning("Model does not support feature importance")
            return None
        
        feature_importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        self.logger.info("Feature importance calculated")
        
        return feature_importance_df
    
    def train_pipeline(self, perform_tuning: bool = False) -> Dict[str, Any]:
        """
        Complete training pipeline.
        
        Args:
            perform_tuning: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary with training results
        """
        self.logger.info("="*60)
        self.logger.info("STARTING COMPLETE TRAINING PIPELINE")
        self.logger.info("="*60)
        
        self.logger.info("Step 1: Data Ingestion")
        ingestion = DataIngestion()
        df, data_info = ingestion.ingest()
        
        self.logger.info("Step 2: Feature Engineering")
        engineer = FeatureEngineer()
        X_train, X_test, y_train, y_test = engineer.preprocess_pipeline(df, is_training=True)
        
        self.logger.info("Step 3: Model Training")
        if perform_tuning:
            self.model, best_params = self.hyperparameter_tuning(X_train, y_train)
        else:
            self.model = self.train_model(X_train, y_train)
            best_params = None
        
        self.logger.info("Step 4: Cross-Validation")
        cv_results = self.cross_validate(X_train, y_train)
        
        self.logger.info("Step 5: Feature Importance")
        feature_importance = self.get_feature_importance(X_train.columns.tolist())
        
        self.logger.info("Step 6: Saving Model")
        self.save_model()
        
        results = {
            'data_shape': data_info['shape'],
            'training_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
            'features': X_train.columns.tolist(),
            'cv_results': cv_results,
            'best_params': best_params,
            'feature_importance': feature_importance.to_dict() if feature_importance is not None else None,
            'model_path': self.config['model_path'],
            'timestamp': datetime.now().isoformat()
        }
        
        self.logger.info("="*60)
        self.logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        self.logger.info("="*60)
        
        return results, X_train, X_test, y_train, y_test


if __name__ == "__main__":
    trainer = ModelTrainer()
    
    results, X_train, X_test, y_train, y_test = trainer.train_pipeline(perform_tuning=False)
    
    print("\n" + "="*60)
    print("MODEL TRAINING SUMMARY")
    print("="*60)
    print(f"Dataset Shape: {results['data_shape']}")
    print(f"Training Samples: {results['training_samples']}")
    print(f"Test Samples: {results['test_samples']}")
    print(f"Features: {results['features']}")
    print(f"\nCross-Validation Score: {results['cv_results']['mean_cv_score']:.4f} (+/- {results['cv_results']['std_cv_score']:.4f})")
    
    if results['best_params']:
        print(f"\nBest Parameters: {results['best_params']}")
    
    if results['feature_importance']:
        print("\nTop 5 Important Features:")
        fi_df = pd.DataFrame(results['feature_importance'])
        print(fi_df.head())
    
    print(f"\nModel saved at: {results['model_path']}")
