# Predictive Customer Churn Analyzer with Explainability

A professional-grade customer churn prediction system that not only predicts customer churn but also provides clear explanations for the predictions. Built with Python, XGBoost, and SHAP, optimized for M1 Macs.

## Features

- 🎯 **Accurate Churn Prediction**: Uses XGBoost classifier optimized for performance
- 📊 **Explainable AI**: Leverages SHAP values to explain each prediction
- 📈 **Interactive Dashboard**: Built with Streamlit for easy interaction
- 🔄 **Batch Processing**: Support for both single and batch predictions
- 📝 **Comprehensive Logging**: Detailed logging for monitoring and debugging
- 🎨 **Data Visualization**: Rich visualizations for better insights
- ⚡ **M1 Optimization**: Specially optimized for Apple Silicon (M1) chips

## Project Structure

```
.
├── Data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── logs/
│   └── churn_predictor.log
├── models/
│   └── plots/
├── src/
│   ├── config/
│   │   └── config.py
│   ├── data/
│   │   └── preprocessor.py
│   ├── models/
│   │   └── model_trainer.py
│   ├── utils/
│   │   └── logger.py
│   ├── visualization/
│   │   └── visualizer.py
│   ├── app.py
│   └── train.py
├── tests/
├── requirements.txt
└── README.md
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Predictive-Customer-Churn-Analyzer
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On macOS/Linux
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Training the Model

Run the training script:
```bash
python src/train.py
```

This will:
- Load and preprocess the data
- Train the XGBoost model
- Generate visualizations
- Save the trained model

### 2. Running the Dashboard

Launch the Streamlit dashboard:
```bash
streamlit run src/app.py
```

The dashboard provides:
- Single customer prediction interface
- Batch prediction capability
- Feature importance visualization
- SHAP value explanations

## Data Requirements

The system expects the following features:
- `tenure`: Number of months the customer has stayed
- `MonthlyCharges`: Monthly charges in dollars
- `TotalCharges`: Total charges in dollars
- `Contract`: Contract type (Month-to-month, One year, Two year)
- `InternetService`: Internet service type (DSL, Fiber optic, No)
- `PaymentMethod`: Payment method used

## Model Performance

The model is evaluated on several metrics:
- Accuracy
- Precision
- Recall
- F1 Score
- ROC AUC

Actual performance metrics will be displayed after training.

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a new Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Dataset source: IBM Sample Data Sets
- Built with Python, XGBoost, SHAP, and Streamlit
- Optimized for Apple Silicon (M1) chips
