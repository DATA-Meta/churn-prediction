# Model Card - Netflix Churn Prediction

## Model Details
- **Algorithm:** Logistic Regression
- **Training Date:** April 2026
- **Framework:** Scikit-learn

## Model Performance
| Metric | Training | Testing |
|--------|----------|---------|
| Accuracy | 74.62% | 73.66% |
| Precision | 74.62% | 73.66% |
| Recall | 100% | 100% |
| F1-Score | 85.47% | 84.83% |

## Training Data
- Total Samples: 25,000 customers
- Training Set: 20,000 (80%)
- Testing Set: 5,000 (20%)
- Churn Rate: 74.4% (imbalanced)

## Features Used
1. Age
2. Country
3. Subscription Type
4. Watch Time Hours
5. Favorite Genre

## Limitations
- Model predicts all customers as churned (100% recall = overfitting)
- Highly imbalanced dataset (74% churn vs 26% active)
- Limited features - may miss important predictors
- No time-series features (trends, seasonality)

## Recommendations for Improvement
- Handle class imbalance with SMOTE or class weights
- Try ensemble methods (Random Forest, XGBoost)
- Add more features (subscription duration, payment history, etc.)
- Use stratified cross-validation

## Model Usage
import pickle

with open('models/churn_model.pkl', 'rb') as f:
    model = pickle.load(f)

predictions = model.predict(X_test)