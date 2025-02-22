"""
Model training module for the Customer Churn Prediction project.
Handles model training, evaluation, and prediction with XGBoost.
"""

import pickle
from typing import Tuple, Dict, Any
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)
import shap

from src.config.config import MODEL_PARAMS, MODEL_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class ChurnPredictor:
    """
    A class to handle model training, evaluation, and prediction for customer churn.
    """
    
    def __init__(self, feature_names: list):
        """
        Initialize the ChurnPredictor with model parameters.
        
        Args:
            feature_names (list): List of feature names used in the model
        """
        self.model = XGBClassifier(**MODEL_PARAMS)
        self.feature_names = feature_names
        self.explainer = None
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        """
        Train the XGBoost model.
        
        Args:
            X_train (np.ndarray): Training features
            y_train (np.ndarray): Training target
        """
        try:
            logger.info("Starting model training")
            
            # Create evaluation set
            eval_set = [(X_train, y_train)]
            
            # Train the model with updated parameters
            self.model.fit(
                X_train,
                y_train,
                eval_set=eval_set,
                verbose=True
            )
            
            # Initialize SHAP explainer
            self.explainer = shap.TreeExplainer(self.model)
            
            logger.info("Model training completed successfully")
            
        except Exception as e:
            logger.error(f"Error in model training: {str(e)}")
            raise
            
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        """
        Evaluate the model performance.
        
        Args:
            X_test (np.ndarray): Test features
            y_test (np.ndarray): Test target
            
        Returns:
            Dict[str, float]: Dictionary containing various performance metrics
        """
        try:
            logger.info("Evaluating model performance")
            
            # Make predictions
            y_pred = self.model.predict(X_test)
            y_pred_proba = self.model.predict_proba(X_test)[:, 1]
            
            # Calculate metrics
            metrics = {
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred),
                'recall': recall_score(y_test, y_pred),
                'f1_score': f1_score(y_test, y_pred),
                'roc_auc': roc_auc_score(y_test, y_pred_proba)
            }
            
            logger.info(f"Model evaluation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {str(e)}")
            raise
            
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on new data.
        
        Args:
            X (np.ndarray): Features to predict on
            
        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted classes and probabilities
        """
        try:
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)[:, 1]
            return predictions, probabilities
            
        except Exception as e:
            logger.error(f"Error in making predictions: {str(e)}")
            raise
            
    def explain_predictions(self, X: np.ndarray) -> np.ndarray:
        """
        Generate SHAP values for explaining predictions.
        
        Args:
            X (np.ndarray): Features to explain
            
        Returns:
            np.ndarray: SHAP values for each feature
        """
        try:
            logger.info("Generating SHAP values for predictions")
            
            if self.explainer is None:
                raise ValueError("Model must be trained before generating explanations")
                
            shap_values = self.explainer.shap_values(X)
            return shap_values
            
        except Exception as e:
            logger.error(f"Error in generating SHAP values: {str(e)}")
            raise
            
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance based on the trained model.
        
        Returns:
            pd.DataFrame: DataFrame containing feature importance scores
        """
        try:
            importance_scores = self.model.feature_importances_
            feature_importance = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance_scores
            })
            return feature_importance.sort_values('importance', ascending=False)
            
        except Exception as e:
            logger.error(f"Error in getting feature importance: {str(e)}")
            raise
            
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath (str): Path to save the model
        """
        try:
            with open(filepath, 'wb') as f:
                pickle.dump(self.model, f)
            logger.info(f"Model saved successfully to {filepath}")
            
        except Exception as e:
            logger.error(f"Error in saving model: {str(e)}")
            raise
            
    @classmethod
    def load_model(cls, filepath: str, feature_names: list) -> 'ChurnPredictor':
        """
        Load a trained model from disk.
        
        Args:
            filepath (str): Path to the saved model
            feature_names (list): List of feature names used in the model
            
        Returns:
            ChurnPredictor: Loaded model instance
        """
        try:
            instance = cls(feature_names)
            with open(filepath, 'rb') as f:
                instance.model = pickle.load(f)
            instance.explainer = shap.TreeExplainer(instance.model)
            logger.info(f"Model loaded successfully from {filepath}")
            return instance
            
        except Exception as e:
            logger.error(f"Error in loading model: {str(e)}")
            raise