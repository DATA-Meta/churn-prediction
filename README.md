# Netflix Customer Churn Prediction

## About This Project
A machine learning project to predict which Netflix customers will cancel their subscriptions based on their activity and subscription type.

## Dataset
- **Size:** 25,000 customers
- **Features:** Age, Country, Subscription Type, Watch Time, Favorite Genre, Last Login
- **Target:** Churn (1 = churned, 0 = active)
- **Time Period:** March 2024 - March 2025

## How Churn Was Defined
Customers who didn't log in for 500+ days are marked as churned.

## What I Did
1. Loaded and explored the Netflix data
2. Created a Churn column based on login activity
3. Encoded text columns (Country, Subscription Type, Genre)
4. Split data into training (80%) and testing (20%)
5. Trained a Logistic Regression model
6. Evaluated model performance with accuracy, precision, recall, F1-score

## Model Results
- **Accuracy:** 73.66%
- **Precision:** 73.66%
- **Recall:** 100%
- **F1-Score:** 84.83%

## What I Learned
- How to prepare data for machine learning
- How to encode categorical variables
- How to split and train a simple ML model
- How to evaluate model performance with multiple metrics

## Files
- `notebooks/01_netflix_eda.ipynb` - Full analysis notebook
- `data/raw/` - Raw datasets (3 CSV files)
- `models/` - Trained model and visualizations

## How to Run
```bash
pip install -r requirements.txt
jupyter notebook notebooks/01_netflix_eda.ipynb

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Jupyter
- Matplotlib

## Next Steps
- Train more advanced models (Random Forest, XGBoost)
- Feature engineering
- Hyperparameter tuning
- Model deployment

## Author
DATA-Meta - Data Science Student