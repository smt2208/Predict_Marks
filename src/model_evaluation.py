"""
Model evaluation utilities for the Student Performance Predictor
"""
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from src.exception import CustomException
from src.logger import logging
import sys

class ModelEvaluator:
    """
    Class for evaluating model performance with various metrics
    """
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Calculate various regression metrics
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            
        Returns:
            Dictionary containing calculated metrics
        """
        try:
            metrics = {
                'r2_score': r2_score(y_true, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
                'mae': mean_absolute_error(y_true, y_pred),
                'mse': mean_squared_error(y_true, y_pred),
                'mape': self._calculate_mape(y_true, y_pred),
                'accuracy_within_5': self._calculate_accuracy_within_threshold(y_true, y_pred, 5),
                'accuracy_within_10': self._calculate_accuracy_within_threshold(y_true, y_pred, 10)
            }
            
            self.metrics = metrics
            return metrics
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def _calculate_mape(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate Mean Absolute Percentage Error"""
        return np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    
    def _calculate_accuracy_within_threshold(self, y_true: np.ndarray, y_pred: np.ndarray, threshold: float) -> float:
        """Calculate accuracy within a certain threshold"""
        return np.mean(np.abs(y_true - y_pred) <= threshold) * 100
    
    def print_metrics(self, model_name: str = "Model") -> None:
        """
        Print formatted metrics
        
        Args:
            model_name: Name of the model for display
        """
        if not self.metrics:
            print("No metrics calculated. Run calculate_metrics first.")
            return
        
        print(f"\n{model_name} Performance Metrics:")
        print("=" * 50)
        print(f"R² Score: {self.metrics['r2_score']:.4f}")
        print(f"RMSE: {self.metrics['rmse']:.4f}")
        print(f"MAE: {self.metrics['mae']:.4f}")
        print(f"MSE: {self.metrics['mse']:.4f}")
        print(f"MAPE: {self.metrics['mape']:.2f}%")
        print(f"Accuracy within 5 points: {self.metrics['accuracy_within_5']:.2f}%")
        print(f"Accuracy within 10 points: {self.metrics['accuracy_within_10']:.2f}%")
        print("=" * 50)
    
    def generate_prediction_plots(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                 model_name: str = "Model", save_path: str = None) -> None:
        """
        Generate prediction vs actual plots
        
        Args:
            y_true: True target values
            y_pred: Predicted target values
            model_name: Name of the model for plot title
            save_path: Path to save the plot (optional)
        """
        try:
            plt.figure(figsize=(15, 5))
            
            # Scatter plot
            plt.subplot(1, 3, 1)
            plt.scatter(y_true, y_pred, alpha=0.6)
            plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
            plt.xlabel('Actual Values')
            plt.ylabel('Predicted Values')
            plt.title(f'{model_name}: Predicted vs Actual')
            plt.grid(True, alpha=0.3)
            
            # Residual plot
            plt.subplot(1, 3, 2)
            residuals = y_true - y_pred
            plt.scatter(y_pred, residuals, alpha=0.6)
            plt.axhline(y=0, color='r', linestyle='--')
            plt.xlabel('Predicted Values')
            plt.ylabel('Residuals')
            plt.title(f'{model_name}: Residual Plot')
            plt.grid(True, alpha=0.3)
            
            # Distribution of residuals
            plt.subplot(1, 3, 3)
            plt.hist(residuals, bins=30, alpha=0.7, edgecolor='black')
            plt.xlabel('Residuals')
            plt.ylabel('Frequency')
            plt.title(f'{model_name}: Residual Distribution')
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                logging.info(f"Plot saved to {save_path}")
            
            plt.show()
            
        except Exception as e:
            raise CustomException(e, sys)
    
    def compare_models(self, model_results: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Compare multiple models
        
        Args:
            model_results: Dictionary with model names as keys and metrics as values
            
        Returns:
            DataFrame with comparison results
        """
        try:
            comparison_df = pd.DataFrame(model_results).T
            
            # Sort by R² score (descending)
            comparison_df = comparison_df.sort_values('r2_score', ascending=False)
            
            # Add ranking
            comparison_df['rank'] = range(1, len(comparison_df) + 1)
            
            return comparison_df
            
        except Exception as e:
            raise CustomException(e, sys)

def evaluate_model_performance(model, X_test: np.ndarray, y_test: np.ndarray, 
                             model_name: str = "Model") -> Dict[str, float]:
    """
    Convenience function to evaluate a single model
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test targets
        model_name: Name of the model
        
    Returns:
        Dictionary containing evaluation metrics
    """
    try:
        evaluator = ModelEvaluator()
        y_pred = model.predict(X_test)
        metrics = evaluator.calculate_metrics(y_test, y_pred)
        evaluator.print_metrics(model_name)
        
        return metrics
        
    except Exception as e:
        raise CustomException(e, sys)
