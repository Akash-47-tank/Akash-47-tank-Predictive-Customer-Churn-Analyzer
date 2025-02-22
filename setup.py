from setuptools import setup, find_packages

setup(
    name="churn_predictor",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.2",
        "xgboost>=1.5.0",
        "shap>=0.40.0",
        "streamlit>=1.0.0",
        "python-dotenv>=0.19.0",
        "matplotlib>=3.4.3",
        "seaborn>=0.11.2"
    ],
)
