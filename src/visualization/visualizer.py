"""
Visualization module for the Customer Churn Prediction project.
Handles creation of various plots and visualizations for data analysis and model explanations.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from typing import Dict, Any, Optional
import os

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ChurnVisualizer:
    """
    A class to handle all visualization tasks for the churn prediction project.
    """
    
    def __init__(self, save_dir: str):
        """
        Initialize the visualizer with a directory to save plots.
        
        Args:
            save_dir (str): Directory to save generated plots
        """
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        
    def _save_plot(self, filename: str) -> str:
        """
        Save the current plot to file.
        
        Args:
            filename (str): Name of the file to save
            
        Returns:
            str: Path to the saved file
        """
        filepath = os.path.join(self.save_dir, filename)
        plt.savefig(filepath, bbox_inches='tight', dpi=300)
        plt.close()
        return filepath
        
    def plot_feature_distributions(self, df: pd.DataFrame, target_col: str) -> Dict[str, str]:
        """
        Plot distributions of features grouped by target variable.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            target_col (str): Name of target column
            
        Returns:
            Dict[str, str]: Dictionary mapping feature names to plot file paths
        """
        try:
            logger.info("Plotting feature distributions")
            plot_paths = {}
            
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            categorical_cols = df.select_dtypes(include=['object']).columns
            
            # Plot numerical features
            for col in numerical_cols:
                plt.figure(figsize=(10, 6))
                sns.boxplot(x=target_col, y=col, data=df)
                plt.title(f'Distribution of {col} by Churn Status')
                plot_paths[col] = self._save_plot(f'{col}_distribution.png')
                
            # Plot categorical features
            for col in categorical_cols:
                if col != target_col:
                    plt.figure(figsize=(10, 6))
                    df_grouped = df.groupby([col, target_col]).size().unstack()
                    df_grouped.plot(kind='bar', stacked=True)
                    plt.title(f'Distribution of {col} by Churn Status')
                    plt.xticks(rotation=45)
                    plot_paths[col] = self._save_plot(f'{col}_distribution.png')
                    
            return plot_paths
            
        except Exception as e:
            logger.error(f"Error in plotting feature distributions: {str(e)}")
            raise
            
    def plot_correlation_matrix(self, df: pd.DataFrame) -> str:
        """
        Plot correlation matrix for numerical features.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            str: Path to the saved correlation matrix plot
        """
        try:
            logger.info("Plotting correlation matrix")
            
            plt.figure(figsize=(12, 8))
            numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns
            correlation_matrix = df[numerical_cols].corr()
            
            sns.heatmap(
                correlation_matrix,
                annot=True,
                cmap='coolwarm',
                center=0,
                fmt='.2f'
            )
            plt.title('Feature Correlation Matrix')
            
            return self._save_plot('correlation_matrix.png')
            
        except Exception as e:
            logger.error(f"Error in plotting correlation matrix: {str(e)}")
            raise
            
    def plot_feature_importance(self, importance_df: pd.DataFrame, top_n: int = 10) -> str:
        """
        Plot feature importance from the model.
        
        Args:
            importance_df (pd.DataFrame): DataFrame with feature importance scores
            top_n (int): Number of top features to show
            
        Returns:
            str: Path to the saved feature importance plot
        """
        try:
            logger.info("Plotting feature importance")
            
            plt.figure(figsize=(10, 6))
            importance_df.head(top_n).plot(
                x='feature',
                y='importance',
                kind='bar'
            )
            plt.title(f'Top {top_n} Most Important Features')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            return self._save_plot('feature_importance.png')
            
        except Exception as e:
            logger.error(f"Error in plotting feature importance: {str(e)}")
            raise
            
    def plot_shap_summary(self, shap_values: np.ndarray, features: pd.DataFrame) -> str:
        """
        Create SHAP summary plot.
        
        Args:
            shap_values (np.ndarray): SHAP values from the model
            features (pd.DataFrame): Feature values
            
        Returns:
            str: Path to the saved SHAP summary plot
        """
        try:
            logger.info("Creating SHAP summary plot")
            
            plt.figure(figsize=(10, 8))
            shap.summary_plot(shap_values, features, show=False)
            
            return self._save_plot('shap_summary.png')
            
        except Exception as e:
            logger.error(f"Error in creating SHAP summary plot: {str(e)}")
            raise
            
    def plot_metrics(self, metrics: Dict[str, float]) -> str:
        """
        Plot model performance metrics.
        
        Args:
            metrics (Dict[str, float]): Dictionary of metric names and values
            
        Returns:
            str: Path to the saved metrics plot
        """
        try:
            logger.info("Plotting model metrics")
            
            plt.figure(figsize=(10, 6))
            metrics_series = pd.Series(metrics)
            metrics_series.plot(kind='bar')
            plt.title('Model Performance Metrics')
            plt.xticks(rotation=45)
            plt.ylim(0, 1)
            
            for i, v in enumerate(metrics_series):
                plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
                
            plt.tight_layout()
            
            return self._save_plot('model_metrics.png')
            
        except Exception as e:
            logger.error(f"Error in plotting model metrics: {str(e)}")
            raise
