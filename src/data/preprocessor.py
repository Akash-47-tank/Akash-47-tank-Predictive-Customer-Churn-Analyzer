"""
Data preprocessing module for the Customer Churn Prediction project.
Handles data cleaning, feature engineering, and preparation for model training.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from typing import Tuple, Dict, Any

from src.config.config import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE
)
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

class DataPreprocessor:
    """
    A class to handle all data preprocessing steps for the churn prediction model.
    """
    
    def __init__(self):
        """Initialize the preprocessor with necessary encoders and scalers."""
        self.label_encoders = {}
        self.scaler = StandardScaler()
        self.feature_names = []
        self.CATEGORICAL_FEATURES = CATEGORICAL_FEATURES
        self.NUMERICAL_FEATURES = NUMERICAL_FEATURES
        self.TARGET_COLUMN = TARGET_COLUMN
        
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with handled missing values
        """
        try:
            logger.info("Handling missing values")
            
            # Create a copy of the DataFrame
            df = df.copy()
            
            # Handle TotalCharges column (convert to numeric)
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
            # Fill numeric missing values with median
            numeric_columns = df.select_dtypes(include=['float64', 'int64']).columns
            for col in numeric_columns:
                df[col] = df[col].fillna(df[col].median())
            
            # Fill categorical missing values with mode
            categorical_columns = df.select_dtypes(include=['object']).columns
            for col in categorical_columns:
                df[col] = df[col].fillna(df[col].mode()[0])
                
            return df
            
        except Exception as e:
            logger.error(f"Error in handling missing values: {str(e)}")
            raise
            
    def _encode_categorical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode categorical features using Label Encoding.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with encoded categorical features
        """
        try:
            logger.info("Encoding categorical features")
            
            for feature in self.CATEGORICAL_FEATURES:
                if feature in df.columns:
                    le = LabelEncoder()
                    df[feature] = le.fit_transform(df[feature].astype(str))
                    self.label_encoders[feature] = le
                    
            return df
            
        except Exception as e:
            logger.error(f"Error in encoding categorical features: {str(e)}")
            raise
            
    def _scale_numerical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            df (pd.DataFrame): Input DataFrame
            
        Returns:
            pd.DataFrame: DataFrame with scaled numerical features
        """
        try:
            logger.info("Scaling numerical features")
            
            numerical_data = df[self.NUMERICAL_FEATURES]
            scaled_data = self.scaler.fit_transform(numerical_data)
            
            for i, feature in enumerate(self.NUMERICAL_FEATURES):
                df[feature] = scaled_data[:, i]
                
            return df
            
        except Exception as e:
            logger.error(f"Error in scaling numerical features: {str(e)}")
            raise
            
    def prepare_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare the data for model training.
        
        Args:
            df (pd.DataFrame): Raw input DataFrame
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Preprocessed features and target
        """
        try:
            logger.info("Starting data preparation")
            
            # Create a copy to avoid modifying the original data
            df_copy = df.copy()
            
            # Handle missing values
            df_copy = self._handle_missing_values(df_copy)
            
            # Encode categorical features
            df_copy = self._encode_categorical_features(df_copy)
            
            # Scale numerical features
            df_copy = self._scale_numerical_features(df_copy)
            
            # Prepare features and target
            X = df_copy[self.CATEGORICAL_FEATURES + self.NUMERICAL_FEATURES]
            y = df_copy[self.TARGET_COLUMN].map({'Yes': 1, 'No': 0})
            
            self.feature_names = X.columns.tolist()
            
            logger.info("Data preparation completed successfully")
            return X, y
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}")
            raise
            
    def split_data(self, X: pd.DataFrame, y: pd.Series) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Split the data into training and testing sets.
        
        Args:
            X (pd.DataFrame): Features
            y (pd.Series): Target
            
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Train and test splits
        """
        try:
            logger.info("Splitting data into train and test sets")
            
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE,
                stratify=y
            )
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"Error in splitting data: {str(e)}")
            raise
            
    def get_feature_names(self) -> list:
        """
        Get the list of feature names after preprocessing.
        
        Returns:
            list: List of feature names
        """
        return self.feature_names