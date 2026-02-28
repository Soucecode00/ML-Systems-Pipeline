"""
Model Evaluation Module

This module handles comprehensive model evaluation including metrics calculation,
visualization, and reporting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
import json
from datetime import datetime
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))
from src.utils.logger import setup_logger
from src.utils.config_loader import load_config


class ModelEvaluator:
    """
    Handles model evaluation and performance analysis.
    
    This class is responsible for:
    - Calculating performance metrics
    - Generating confusion matrix
    - Creating ROC curves
    - Producing evaluation reports
    - Visualizing results
    """
    
    def __init__(self):
        """Initialize ModelEvaluator with logger and configuration."""
        self.logger = setup_logger(__name__, "model_evaluation")
        self.config = load_config("model_config")
        self.metrics = {}
        self.logger.info("ModelEvaluator initialized")
    
    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None) -> Dict[str, float]:
        """
        Calculate comprehensive evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            y_proba: Prediction probabilities (optional)
            
        Returns:
            Dictionary with all metrics
        """
        self.logger.info("Calculating evaluation metrics...")
        
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1_score': f1_score(y_true, y_pred, average='binary', zero_division=0)
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        
        self.metrics = metrics
        
        self.logger.info("Metrics calculated:")
        for metric_name, value in metrics.items():
            self.logger.info(f"  {metric_name}: {value:.4f}")
        
        return metrics
    
    def get_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray) -> np.ndarray:
        """
        Calculate confusion matrix.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Confusion matrix as numpy array
        """
        cm = confusion_matrix(y_true, y_pred)
        self.logger.info(f"Confusion Matrix:\n{cm}")
        return cm
    
    def get_classification_report(self, y_true: pd.Series, y_pred: np.ndarray) -> str:
        """
        Generate detailed classification report.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            Classification report as string
        """
        report = classification_report(y_true, y_pred, zero_division=0)
        self.logger.info(f"Classification Report:\n{report}")
        return report
    
    def plot_confusion_matrix(self, y_true: pd.Series, y_pred: np.ndarray, 
                             save_path: str = None):
        """
        Create and save confusion matrix visualization.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            save_path: Path to save the plot
        """
        self.logger.info("Generating confusion matrix plot...")
        
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Rejected', 'Approved'],
                   yticklabels=['Rejected', 'Approved'])
        plt.title('Confusion Matrix - Loan Prediction', fontsize=14, fontweight='bold')
        plt.ylabel('Actual', fontsize=12)
        plt.xlabel('Predicted', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix plot saved to: {save_path}")
        else:
            plt.savefig('models/confusion_matrix.png', dpi=300, bbox_inches='tight')
            self.logger.info("Confusion matrix plot saved to: models/confusion_matrix.png")
        
        plt.close()
    
    def plot_roc_curve(self, y_true: pd.Series, y_proba: np.ndarray, 
                      save_path: str = None):
        """
        Create and save ROC curve visualization.
        
        Args:
            y_true: True labels
            y_proba: Prediction probabilities
            save_path: Path to save the plot
        """
        self.logger.info("Generating ROC curve plot...")
        
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, 
                label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
                label='Random Classifier')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate', fontsize=12)
        plt.ylabel('True Positive Rate', fontsize=12)
        plt.title('ROC Curve - Loan Prediction', fontsize=14, fontweight='bold')
        plt.legend(loc="lower right")
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"ROC curve plot saved to: {save_path}")
        else:
            plt.savefig('models/roc_curve.png', dpi=300, bbox_inches='tight')
            self.logger.info("ROC curve plot saved to: models/roc_curve.png")
        
        plt.close()
    
    def plot_feature_importance(self, feature_importance_df: pd.DataFrame, 
                               save_path: str = None):
        """
        Create and save feature importance visualization.
        
        Args:
            feature_importance_df: DataFrame with feature importance
            save_path: Path to save the plot
        """
        self.logger.info("Generating feature importance plot...")
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=feature_importance_df.head(10), 
                   x='importance', y='feature', palette='viridis')
        plt.title('Top 10 Feature Importance - Loan Prediction', 
                 fontsize=14, fontweight='bold')
        plt.xlabel('Importance Score', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Feature importance plot saved to: {save_path}")
        else:
            plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
            self.logger.info("Feature importance plot saved to: models/feature_importance.png")
        
        plt.close()
    
    def save_metrics(self, metrics: Dict[str, Any], path: str = None):
        """
        Save evaluation metrics to JSON file.
        
        Args:
            metrics: Dictionary with metrics
            path: Path to save metrics
        """
        if path is None:
            path = self.config['metrics_path']
        
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        
        metrics['timestamp'] = datetime.now().isoformat()
        
        with open(path, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics saved to: {path}")
    
    def generate_evaluation_report(self, model, X_test: pd.DataFrame, 
                                   y_test: pd.Series) -> Dict[str, Any]:
        """
        Generate comprehensive evaluation report.
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary with complete evaluation results
        """
        self.logger.info("="*60)
        self.logger.info("GENERATING COMPREHENSIVE EVALUATION REPORT")
        self.logger.info("="*60)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        metrics = self.calculate_metrics(y_test, y_pred, y_proba)
        
        cm = self.get_confusion_matrix(y_test, y_pred)
        
        report = self.get_classification_report(y_test, y_pred)
        
        self.plot_confusion_matrix(y_test, y_pred)
        
        self.plot_roc_curve(y_test, y_proba)
        
        evaluation_results = {
            'metrics': metrics,
            'confusion_matrix': cm.tolist(),
            'classification_report': report,
            'test_samples': len(y_test),
            'predictions': {
                'correct': int((y_pred == y_test).sum()),
                'incorrect': int((y_pred != y_test).sum())
            }
        }
        
        self.save_metrics(evaluation_results)
        
        self.logger.info("="*60)
        self.logger.info("EVALUATION REPORT COMPLETED")
        self.logger.info("="*60)
        
        return evaluation_results


if __name__ == "__main__":
    from src.models.train_model import ModelTrainer
    
    trainer = ModelTrainer()
    results, X_train, X_test, y_train, y_test = trainer.train_pipeline()
    
    evaluator = ModelEvaluator()
    evaluation_results = evaluator.generate_evaluation_report(
        trainer.model, X_test, y_test
    )
    
    print("\n" + "="*60)
    print("MODEL EVALUATION SUMMARY")
    print("="*60)
    print(f"Test Samples: {evaluation_results['test_samples']}")
    print(f"Correct Predictions: {evaluation_results['predictions']['correct']}")
    print(f"Incorrect Predictions: {evaluation_results['predictions']['incorrect']}")
    print("\nPerformance Metrics:")
    for metric, value in evaluation_results['metrics'].items():
        print(f"  {metric.upper()}: {value:.4f}")
    print("\nClassification Report:")
    print(evaluation_results['classification_report'])
