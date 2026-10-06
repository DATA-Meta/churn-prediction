# Model Card: Netflix Churn Prediction (v2)

## Model details
- **Algorithm:** XGBoost classifier inside a scikit-learn pipeline (one-hot encoding + scaling)
- **Class imbalance:** `scale_pos_weight`
- **Decision threshold:** 0.35, tuned for F1 on 5-fold cross-validated training predictions
- **Artifact:** `models/churn_model.joblib` (pipeline, threshold, feature list)
- **Notebook:** `notebooks/02_churn_modeling.ipynb`

## Data
- 25,000 synthetic customers; stratified split into 20,000 train / 5,000 test
- Target: simulated churn label (`netflix_data_realistic_churn.csv`), 45.1% positive
- Features: Age, Country, Subscription_Type, Watch_Time_Hours, Favorite_Genre

## Performance (test set)

| Metric | Majority baseline | XGBoost @0.5 | XGBoost @0.35 |
|---|---|---|---|
| ROC-AUC | 0.500 | 0.686 | 0.686 |
| PR-AUC | 0.451 | 0.634 | 0.634 |
| Precision | 0.000 | 0.588 | 0.516 |
| Recall | 0.000 | 0.658 | 0.867 |
| F1 | 0.000 | 0.621 | 0.647 |
| Balanced accuracy | 0.500 | 0.640 | n/a |

5-fold CV ROC-AUC on train: Logistic Regression 0.689 · Random Forest 0.694 · XGBoost 0.698.

## Feature importance (permutation, drop in ROC-AUC)
Subscription_Type 0.112 · Watch_Time_Hours 0.055 · Favorite_Genre 0.010 · Age 0.003 · Country ≈ 0

## Intended use
A portfolio demonstration of a churn-modelling workflow: ranking subscribers by churn risk for a retention campaign.

## Limitations
- Synthetic data with a simulated label, so scores do not transfer to a real service.
- Few features; real churn models would add tenure, payments, engagement trends and support contacts.
- No time-based validation: the split is random, not by time.

## Version history
- **v1 (April 2026):** Logistic Regression on a label defined by 500+ days since last login. The features had no signal (ROC-AUC ≈ 0.50) and the model predicted every customer as churned (100% recall, 73.7% accuracy = majority rate). Kept in `notebooks/01_netflix_eda.ipynb` for reference.
- **v2 (October 2026):** diagnosis of v1, baselines, stratified CV, class weighting, three model families, threshold tuning, permutation importance.
