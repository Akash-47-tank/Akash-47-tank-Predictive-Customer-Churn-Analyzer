"""
Configuration settings for the Customer Churn Prediction project.
Contains all the necessary paths, model parameters, and other configurations.
"""

import os
from pathlib import Path

# Base paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
DATA_DIR = os.path.join(BASE_DIR, "Data")
LOG_DIR = os.path.join(BASE_DIR, "logs")
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Data files
RAW_DATA_FILE = os.path.join(DATA_DIR, "WA_Fn-UseC_-Telco-Customer-Churn.csv")
PROCESSED_DATA_FILE = os.path.join(DATA_DIR, "processed_telco_data.csv")

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model parameters optimized for M1 chip
# Model parameters
MODEL_PARAMS = {
    'n_estimators': 500,        # Increased from 200 to 500 epochs for better learning
    'max_depth': 8,             # Increased from 6 to 8 to capture more complex patterns
    'learning_rate': 0.008,     # Further reduced for better generalization
    'subsample': 0.85,          # Slightly increased subsample ratio
    'colsample_bytree': 0.85,   # Slightly increased column sampling
    'min_child_weight': 2,      # Reduced to allow for more tree splits
    'gamma': 0.05,              # Reduced to allow more tree splits while maintaining regularization
    'random_state': RANDOM_STATE,
    'objective': 'binary:logistic',
    'use_label_encoder': False,
    'early_stopping_rounds': 20, # Stop if no improvement after 20 rounds
    'eval_metric': 'auc'        # Use AUC as evaluation metric
}
# Feature configuration
CATEGORICAL_FEATURES = [
    "gender", "InternetService", "Contract", "PaymentMethod",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies"
]

NUMERICAL_FEATURES = [
    "tenure", "MonthlyCharges", "TotalCharges"
]

TARGET_COLUMN = "Churn"

# Logging configuration
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE = os.path.join(LOG_DIR, "churn_predictor.log")
