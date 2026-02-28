"""
Data Ingestion Module

This module handles loading data from various sources, validating data quality,
and preparing it for preprocessing.
"""

import pandas as pd
from pathlib import Path
from typing import Tuple, Optional
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class DataIngestion:
    """
    Handles data ingestion from CSV files with validation.
    
    This class is responsible for:
    - Loading raw data from CSV files
    - Validating data schema and quality
    - Checking for missing values
    - Ensuring data types are correct
    """
    
    def __init__(self):
        """Initialize DataIngestion with logger and configuration."""
        self.logger = setup_logger(__name__, "data_ingestion")
        self.config = load_config("data_config")
        self.logger.info("DataIngestion initialized")
    
    def load_data(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Args:
            file_path: Path to CSV file. If None, uses path from config.
            
        Returns:
            pandas DataFrame with loaded data
            
        Raises:
            FileNotFoundError: If the data file doesn't exist
        """
        if file_path is None:
            file_path = self.config['data_source']['raw_data_path']
        
        file_path = Path(file_path)
        
        if not file_path.exists():
            self.logger.error(f"Data file not found: {file_path}")
            raise FileNotFoundError(f"Data file not found: {file_path}")
        
        self.logger.info(f"Loading data from: {file_path}")
        
        try:
            df = pd.read_csv(file_path)
            self.logger.info(f"Data loaded successfully. Shape: {df.shape}")
            return df
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def validate_data(self, df: pd.DataFrame) -> Tuple[bool, list]:
        """
        Validate data schema and quality.
        
        Args:
            df: DataFrame to validate
            
        Returns:
            Tuple of (is_valid, list of validation errors)
        """
        self.logger.info("Validating data...")
        errors = []
        
        required_columns = self.config['validation']['required_columns']
        missing_columns = set(required_columns) - set(df.columns)
        
        if missing_columns:
            error_msg = f"Missing required columns: {missing_columns}"
            errors.append(error_msg)
            self.logger.error(error_msg)
        
        if 'age' in df.columns:
            age_range = self.config['validation']['age_range']
            invalid_ages = df[
                (df['age'] < age_range[0]) | (df['age'] > age_range[1])
            ]
            if len(invalid_ages) > 0:
                error_msg = f"Found {len(invalid_ages)} records with invalid age"
                errors.append(error_msg)
                self.logger.warning(error_msg)
        
        if 'income' in df.columns:
            income_range = self.config['validation']['income_range']
            invalid_income = df[
                (df['income'] < income_range[0]) | (df['income'] > income_range[1])
            ]
            if len(invalid_income) > 0:
                error_msg = f"Found {len(invalid_income)} records with invalid income"
                errors.append(error_msg)
                self.logger.warning(error_msg)
        
        missing_values = df.isnull().sum()
        if missing_values.any():
            error_msg = f"Found missing values:\n{missing_values[missing_values > 0]}"
            errors.append(error_msg)
            self.logger.warning(error_msg)
        
        is_valid = len(errors) == 0
        
        if is_valid:
            self.logger.info("Data validation passed!")
        else:
            self.logger.warning(f"Data validation completed with {len(errors)} warnings/errors")
        
        return is_valid, errors
    
    def get_data_info(self, df: pd.DataFrame) -> dict:
        """
        Get comprehensive information about the dataset.
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'shape': df.shape,
            'columns': list(df.columns),
            'dtypes': df.dtypes.to_dict(),
            'missing_values': df.isnull().sum().to_dict(),
            'duplicates': df.duplicated().sum(),
            'memory_usage': df.memory_usage(deep=True).sum() / 1024**2
        }
        
        if 'approved' in df.columns:
            info['target_distribution'] = df['approved'].value_counts().to_dict()
        
        self.logger.info(f"Dataset info: {info['shape'][0]} rows, {info['shape'][1]} columns")
        
        return info
    
    def ingest(self, file_path: Optional[str] = None) -> Tuple[pd.DataFrame, dict]:
        """
        Complete data ingestion pipeline.
        
        Args:
            file_path: Path to data file
            
        Returns:
            Tuple of (DataFrame, data_info_dict)
        """
        self.logger.info("Starting data ingestion pipeline...")
        
        df = self.load_data(file_path)
        
        is_valid, errors = self.validate_data(df)
        
        data_info = self.get_data_info(df)
        
        self.logger.info("Data ingestion completed successfully!")
        
        return df, data_info


if __name__ == "__main__":
    ingestion = DataIngestion()
    df, info = ingestion.ingest()
    
    print("\n" + "="*50)
    print("DATA INGESTION SUMMARY")
    print("="*50)
    print(f"Dataset Shape: {info['shape']}")
    print(f"Columns: {info['columns']}")
    print(f"Missing Values: {info['missing_values']}")
    print(f"Duplicates: {info['duplicates']}")
    print(f"Memory Usage: {info['memory_usage']:.2f} MB")
    
    if 'target_distribution' in info:
        print(f"Target Distribution: {info['target_distribution']}")
    
    print("\nFirst 5 rows:")
    print(df.head())
