End-to-End Housing Price Prediction API with Docker & FastAPI
🚀 Overview

This project implements a production-grade machine learning pipeline for predicting residential house prices using the Kaggle House Prices dataset.

Unlike a standard notebook workflow, this system is designed with modularity, reproducibility, and deployment readiness in mind.

It includes:

End-to-end data preprocessing pipeline
Feature engineering layer based on domain knowledge
CatBoost regression model
Bayesian hyperparameter optimization using Optuna
Cross-validation evaluation strategy
Inference-ready model export
🎯 Problem Statement

Accurately predicting house prices requires modeling complex nonlinear relationships between:

Property characteristics
Structural attributes
Quality and condition variables
Spatial and temporal features

This project aims to build a robust regression system capable of generalizing to unseen housing data.

🧱 System Architecture

The pipeline is structured into three core layers:

1. Data Layer
Raw Kaggle dataset ingestion
Missing value handling
Outlier filtering (e.g. extreme GrLivArea cases)
Log transformation of target variable
2. Feature Engineering Layer

Domain-driven feature creation:

TotalSF (Total square footage)
Total_Bathrooms (combined bathroom metric)
Qual_Density (interaction feature: quality × area)
Age_at_Sale (temporal feature)
Binary indicators (pool, garage, second floor presence)
Ordinal encoding for quality-related features

This layer is fully reusable and isolated for production consistency.

3. Modeling Layer
Primary model: CatBoost Regressor
Handles categorical variables natively
Optimized for tabular structured data

Supporting techniques:

Stratified K-Fold via binned target distribution
Early stopping to prevent overfitting
Log-space optimization for stability
⚙️ Hyperparameter Optimization

Bayesian optimization implemented using Optuna.

Tuned parameters:
learning_rate
depth
l2_leaf_reg
random_strength
bagging_temperature
Optimization strategy:
Multi-fold cross-validation
RMSE-based objective function
Early stopping per fold

This enables efficient exploration of the hyperparameter space while minimizing overfitting risk.

📊 Evaluation
Metric: Root Mean Squared Error (RMSE)
Validation: 5-fold cross-validation
Target transformation: log1p(SalePrice)
Result:
Competitive Kaggle performance: Top 38%
📦 Inference Pipeline

The trained model is exported for production use:

Model serialized using joblib
Input schema standardized via preprocessing pipeline
Prediction-ready function for new data
🚀 Deployment Ready Design (Optional Extension)

This system is designed to support:

FastAPI inference service
Batch prediction pipelines
Cloud deployment (AWS / Render / GCP)
🧠 Key Engineering Decisions
Log transformation applied to stabilize target variance
CatBoost selected for native categorical handling
Feature engineering prioritized over model complexity
Cross-validation used instead of single split evaluation
Optuna used for scalable hyperparameter tuning
⚠️ Limitations
No ensemble stacking implemented (future improvement)
Limited Optuna trials due to compute constraints
No live API deployment yet
🔮 Future Work
Add model stacking (CatBoost + LightGBM + XGBoost)
Deploy FastAPI inference service
Integrate SHAP explainability layer
Add CI/CD pipeline for training automation
Expand Optuna search with pruning
🛠 Tech Stack
Python
Pandas / NumPy
Scikit-learn
CatBoost
Optuna
Joblib
👤 Author

Machine Learning Engineer (aspiring)
Focused on applied ML systems, regression modeling, and production deployment workflows
