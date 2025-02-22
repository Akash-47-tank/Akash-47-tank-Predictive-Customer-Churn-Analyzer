"""
Streamlit application for the Customer Churn Prediction project.
Provides a user interface for making predictions and viewing explanations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import os
import glob
from typing import Tuple

from src.data.preprocessor import DataPreprocessor
from src.models.model_trainer import ChurnPredictor
from src.visualization.visualizer import ChurnVisualizer
from src.config.config import MODEL_DIR
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def load_latest_model() -> Tuple[ChurnPredictor, str]:
    """
    Load the most recently trained model.
    
    Returns:
        Tuple[ChurnPredictor, str]: Loaded model and model path
    """
    model_files = glob.glob(os.path.join(MODEL_DIR, 'churn_model_*.pkl'))
    if not model_files:
        raise FileNotFoundError("No trained model found. Please run training first.")
    
    latest_model = max(model_files, key=os.path.getctime)
    preprocessor = DataPreprocessor()
    model = ChurnPredictor.load_model(latest_model, preprocessor.get_feature_names())
    return model, latest_model

def main():
    """Main function for the Streamlit application."""
    try:
        st.set_page_config(
            page_title="Customer Churn Predictor",
            page_icon="🔄",
            layout="wide"
        )
        
        st.title("Customer Churn Predictor with Explainability")
        st.write("""
        This application predicts customer churn probability and explains the factors
        influencing the prediction. Upload your customer data or use the form to make
        individual predictions.
        """)
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load model
        model, model_path = load_latest_model()
        st.sidebar.info(f"Using model: {os.path.basename(model_path)}")
        
        # Create tabs
        tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])
        
        with tab1:
            st.header("Make Single Prediction")
            
            # Create input form
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tenure = st.number_input("Tenure (months)", min_value=0, max_value=100)
                    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0)
                    total_charges = st.number_input("Total Charges ($)", min_value=0.0)
                    gender = st.selectbox("Gender", ["Male", "Female"])
                    
                with col2:
                    contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
                    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
                    payment_method = st.selectbox("Payment Method", 
                        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
                    online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
                    
                with col3:
                    online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
                    device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
                    tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
                    streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
                    streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
                
                submit_button = st.form_submit_button("Predict Churn")
                
            if submit_button:
                try:
                    # Create a DataFrame from form inputs
                    input_data = pd.DataFrame({
                        'tenure': [tenure],
                        'MonthlyCharges': [monthly_charges],
                        'TotalCharges': [str(total_charges)],  # Convert to string to match training data format
                        'gender': [gender],
                        'Contract': [contract],
                        'InternetService': [internet_service],
                        'PaymentMethod': [payment_method],
                        'OnlineSecurity': [online_security],
                        'OnlineBackup': [online_backup],
                        'DeviceProtection': [device_protection],
                        'TechSupport': [tech_support],
                        'StreamingTV': [streaming_tv],
                        'StreamingMovies': [streaming_movies],
                        'Churn': ['No']  # Add default value for Churn
                    })

                    # Ensure all columns are present and in the correct order
                    required_columns = preprocessor.CATEGORICAL_FEATURES + preprocessor.NUMERICAL_FEATURES + [preprocessor.TARGET_COLUMN]
                    for col in required_columns:
                        if col not in input_data.columns:
                            input_data[col] = ['No' if col != preprocessor.TARGET_COLUMN else 'No']
                    
                    input_data = input_data[required_columns]  # Reorder columns to match training data
                    
                    # Preprocess input data
                    X, _ = preprocessor.prepare_data(input_data)
                    
                    # Make prediction
                    prediction, probability = model.predict(X)
                    
                    # Get SHAP values
                    shap_values = model.explain_predictions(X)
                    
                    # Display results
                    st.subheader("Prediction Results")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric(
                            "Churn Probability",
                            f"{probability[0]:.2%}",
                            delta="High Risk" if probability[0] > 0.5 else "Low Risk"
                        )
                        
                    with col2:
                        st.metric(
                            "Prediction",
                            "Will Churn" if prediction[0] == 1 else "Will Stay",
                            delta="Action Needed" if prediction[0] == 1 else "Stable"
                        )
                    
                    # Display feature importance
                    st.subheader("Feature Importance")
                    feature_importance = model.get_feature_importance()
                    st.bar_chart(feature_importance.set_index('feature')['importance'])
                    
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")
                    logger.error(f"Error in single prediction: {str(e)}")
        
        with tab2:
            st.header("Batch Prediction")
            uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])
            
            if uploaded_file is not None:
                try:
                    # Load and preprocess uploaded data
                    df = pd.read_csv(uploaded_file)
                    if 'Churn' not in df.columns:
                        df['Churn'] = 'No'  # Add default Churn column if not present
                    X, _ = preprocessor.prepare_data(df)
                    
                    # Make predictions
                    predictions, probabilities = model.predict(X)
                    
                                        
                    # Add predictions to DataFrame
                    df['Churn_Probability'] = probabilities
                    df['Predicted_Churn'] = predictions
                    
                    # Display results
                    st.subheader("Prediction Results")
                    st.write(df)
                    
                    # Download results
                    csv = df.to_csv(index=False)
                    st.download_button(
                        "Download Predictions",
                        csv,
                        "predictions.csv",
                        "text/csv",
                        key='download-csv'
                    )
                    
                    # Show summary statistics
                    st.subheader("Summary Statistics")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("Total Customers", len(df))
                        st.metric("Predicted Churn Rate", f"{(predictions == 1).mean():.2%}")
                        
                    with col2:
                        st.metric("High Risk Customers (>50%)", (probabilities > 0.5).sum())
                        st.metric("Average Churn Probability", f"{probabilities.mean():.2%}")
                    
                except Exception as e:
                    st.error(f"Error processing file: {str(e)}")
                    logger.error(f"Error in batch prediction: {str(e)}")
        
        # Add footer
        st.markdown("---")
        st.markdown("""
        ### About this App
        This customer churn predictor uses machine learning to identify customers at risk of churning.
        It provides both individual and batch predictions, along with explanations of what factors
        contribute to the churn risk.
        
        For best results:
        - Ensure your data includes all required fields
        - Use accurate and up-to-date customer information
        - Review both the prediction and the contributing factors
        """)
        
    except Exception as e:
        st.error(f"Application Error: {str(e)}")
        logger.error(f"Application error: {str(e)}")

if __name__ == "__main__":
    main()