"""
Feature Engineering and Preprocessing Module

This module handles data preprocessing, feature engineering, and transformation
to prepare data for machine learning models.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class FeatureEngineer:
    """
    Handles feature engineering and preprocessing for loan prediction.
    
    This class is responsible for:
    - Handling missing values
    - Feature scaling and normalization
    - Creating new features
    - Train-test splitting
    - Saving/loading preprocessing artifacts
    """
    
    def __init__(self):
        """Initialize FeatureEngineer with logger and configuration."""
        self.logger = setup_logger(__name__, "feature_engineering")
        self.data_config = load_config("data_config")
        self.model_config = load_config("model_config")
        self.scaler = StandardScaler()
        self.logger.info("FeatureEngineer initialized")
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        self.logger.info("Handling missing values...")
        
        initial_shape = df.shape
        missing_count = df.isnull().sum().sum()
        
        if missing_count > 0:
            self.logger.warning(f"Found {missing_count} missing values")
            
            method = self.data_config['preprocessing']['handle_missing']
            
            if method == 'drop':
                df = df.dropna()
                self.logger.info(f"Dropped rows with missing values. Shape: {initial_shape} -> {df.shape}")
            elif method == 'mean':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].mean())
                self.logger.info("Filled missing values with mean")
            elif method == 'median':
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
                self.logger.info("Filled missing values with median")
        else:
            self.logger.info("No missing values found")
        
        return df
    
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create new engineered features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new features
        """
        self.logger.info("Creating engineered features...")
        
        df = df.copy()
        
        if 'income' in df.columns and 'savings' in df.columns:
            df['savings_to_income_ratio'] = df['savings'] / (df['income'] + 1)
            self.logger.info("Created feature: savings_to_income_ratio")
        
        if 'age' in df.columns:
            df['age_group'] = pd.cut(
                df['age'], 
                bins=[0, 25, 35, 50, 100], 
                labels=['young', 'mid', 'senior', 'elderly']
            )
            df['age_group_encoded'] = df['age_group'].cat.codes
            df = df.drop('age_group', axis=1)
            self.logger.info("Created feature: age_group_encoded")
        
        if 'income' in df.columns:
            df['income_category'] = pd.cut(
                df['income'],
                bins=[0, 30000, 60000, 100000, float('inf')],
                labels=['low', 'medium', 'high', 'very_high']
            )
            df['income_category_encoded'] = df['income_category'].cat.codes
            df = df.drop('income_category', axis=1)
            self.logger.info("Created feature: income_category_encoded")
        
        self.logger.info(f"Feature engineering completed. New shape: {df.shape}")
        
        return df
    
    def scale_features(self, X_train: pd.DataFrame, X_test: pd.DataFrame = None, 
                      fit: bool = True) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Scale features using StandardScaler.
        
        Args:
            X_train: Training features
            X_test: Test features (optional)
            fit: Whether to fit the scaler on training data
            
        Returns:
            Tuple of (scaled_X_train, scaled_X_test)
        """
        self.logger.info("Scaling features...")
        
        feature_cols = X_train.columns
        
        if fit:
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.logger.info("Scaler fitted on training data")
        else:
            X_train_scaled = self.scaler.transform(X_train)
        
        X_train_scaled = pd.DataFrame(X_train_scaled, columns=feature_cols, index=X_train.index)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            X_test_scaled = pd.DataFrame(X_test_scaled, columns=feature_cols, index=X_test.index)
            self.logger.info("Features scaled successfully")
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled, None
    
    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                      pd.Series, pd.Series]:
        """
        Split data into train and test sets.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.logger.info("Splitting data into train and test sets...")
        
        feature_cols = self.model_config['features']['input_features']
        target_col = self.model_config['features']['target_feature']
        
        if 'savings_to_income_ratio' in df.columns:
            feature_cols = feature_cols + ['savings_to_income_ratio']
        if 'age_group_encoded' in df.columns:
            feature_cols = feature_cols + ['age_group_encoded']
        if 'income_category_encoded' in df.columns:
            feature_cols = feature_cols + ['income_category_encoded']
        
        X = df[feature_cols]
        y = df[target_col]
        
        test_size = self.model_config['data_split']['test_size']
        random_state = self.model_config['data_split']['random_state']
        stratify = y if self.model_config['data_split']['stratify'] else None
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            random_state=random_state,
            stratify=stratify
        )
        
        self.logger.info(f"Train set: {X_train.shape}, Test set: {X_test.shape}")
        self.logger.info(f"Train target distribution:\n{y_train.value_counts()}")
        self.logger.info(f"Test target distribution:\n{y_test.value_counts()}")
        
        return X_train, X_test, y_train, y_test
    
    def save_scaler(self, path: Optional[str] = None):
        """
        Save the fitted scaler to disk.
        
        Args:
            path: Path to save scaler. If None, uses path from config.
        """
        if path is None:
            path = self.model_config['scaler_path']
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, path)
        self.logger.info(f"Scaler saved to: {path}")
    
    def load_scaler(self, path: Optional[str] = None):
        """
        Load a previously saved scaler.
        
        Args:
            path: Path to load scaler from. If None, uses path from config.
        """
        if path is None:
            path = self.model_config['scaler_path']
        
        if not Path(path).exists():
            self.logger.error(f"Scaler not found at: {path}")
            raise FileNotFoundError(f"Scaler not found at: {path}")
        
        self.scaler = joblib.load(path)
        self.logger.info(f"Scaler loaded from: {path}")
    
    def preprocess_pipeline(self, df: pd.DataFrame, 
                           is_training: bool = True) -> Tuple:
        """
        Complete preprocessing pipeline.
        
        Args:
            df: Input DataFrame
            is_training: Whether this is training data (splits and fits scaler)
            
        Returns:
            If training: (X_train, X_test, y_train, y_test)
            If inference: (X_processed,)
        """
        self.logger.info("Starting preprocessing pipeline...")
        
        df = self.handle_missing_values(df)
        
        df = self.create_features(df)
        
        if is_training:
            X_train, X_test, y_train, y_test = self.split_data(df)
            
            X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test, fit=True)
            
            self.save_scaler()
            
            self.logger.info("Preprocessing pipeline completed for training data")
            return X_train_scaled, X_test_scaled, y_train, y_test
        else:
            feature_cols = self.model_config['features']['input_features']
            if 'savings_to_income_ratio' in df.columns:
                feature_cols = feature_cols + ['savings_to_income_ratio']
            if 'age_group_encoded' in df.columns:
                feature_cols = feature_cols + ['age_group_encoded']
            if 'income_category_encoded' in df.columns:
                feature_cols = feature_cols + ['income_category_encoded']
            
            X = df[feature_cols]
            
            self.load_scaler()
            X_scaled, _ = self.scale_features(X, fit=False)
            
            self.logger.info("Preprocessing pipeline completed for inference data")
            return X_scaled


if __name__ == "__main__":
    from src.data.data_ingestion import DataIngestion
    
    ingestion = DataIngestion()
    df, _ = ingestion.ingest()
    
    engineer = FeatureEngineer()
    X_train, X_test, y_train, y_test = engineer.preprocess_pipeline(df, is_training=True)
    
    print("\n" + "="*50)
    print("FEATURE ENGINEERING SUMMARY")
    print("="*50)
    print(f"Training features shape: {X_train.shape}")
    print(f"Test features shape: {X_test.shape}")
    print(f"Training target shape: {y_train.shape}")
    print(f"Test target shape: {y_test.shape}")
    print(f"\nFeature columns: {list(X_train.columns)}")
    print(f"\nTraining data sample:")
    print(X_train.head())
