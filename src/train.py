"""
Main training script for the Customer Churn Prediction project.
Orchestrates the entire training process from data loading to model evaluation.
"""

import os
import pandas as pd
from datetime import datetime

from src.config.config import (
    RAW_DATA_FILE,
    MODEL_DIR,
    LOG_DIR
)
from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ChurnPredictor
from src.visualization.visualizer import ChurnVisualizer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def train_model():
    """
    Main function to train the churn prediction model.
    Handles the entire pipeline from data loading to model evaluation.
    """
    try:
        # Create necessary directories
        os.makedirs(MODEL_DIR, exist_ok=True)
        os.makedirs(LOG_DIR, exist_ok=True)
        
        # Initialize visualizer
        visualizer = ChurnVisualizer(os.path.join(MODEL_DIR, 'plots'))
        
        # Load data
        logger.info("Loading raw data")
        df = pd.read_csv(RAW_DATA_FILE)
        logger.info(f"Loaded dataset with shape: {df.shape}")
        
        # Create visualization of raw data
        logger.info("Creating initial data visualizations")
        visualizer.plot_feature_distributions(df, 'Churn')
        visualizer.plot_correlation_matrix(df)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Prepare features and target
        logger.info("Preprocessing data")
        X, y = preprocessor.prepare_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
        logger.info(f"Training set shape: {X_train.shape}, Test set shape: {X_test.shape}")
        
        # Initialize and train model
        logger.info("Initializing model")
        model = ChurnPredictor(preprocessor.get_feature_names())
        
        logger.info("Training model")
        model.train(X_train, y_train)
        
        # Evaluate model
        logger.info("Evaluating model")
        metrics = model.evaluate(X_test, y_test)
        
        # Generate feature importance plot
        feature_importance = model.get_feature_importance()
        visualizer.plot_feature_importance(feature_importance)
        
        # Generate SHAP values and plot
        logger.info("Generating SHAP values")
        shap_values = model.explain_predictions(X_test)
        visualizer.plot_shap_summary(shap_values, X_test)
        
        # Plot metrics
        visualizer.plot_metrics(metrics)
        
        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(MODEL_DIR, f'churn_model_{timestamp}.pkl')
        model.save_model(model_path)
        
        logger.info(f"Model training completed. Model saved at: {model_path}")
        logger.info(f"Model metrics: {metrics}")
        
        return model_path, metrics
        
    except Exception as e:
        logger.error(f"Error in model training pipeline: {str(e)}")
        raise

if __name__ == "__main__":
    train_model()
