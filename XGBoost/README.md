# Forest Fire Prediction using XGBoost

This project uses **XGBoost Classification** to predict whether a fire occurs in a given grid cell based on environmental, weather, and geographical features.

## Project Overview

The model uses historical environmental and weather-related data to classify each grid cell into:

- Fire
- No Fire

This XGBoost model is developed as a **baseline model for forest fire prediction**.

## Algorithm

- XGBoost Classification
- `XGBClassifier`
- RandomizedSearchCV for hyperparameter tuning
- Stratified K-Fold Cross-Validation

## Features Used

The model uses environmental, weather, and geographical features such as:

- NDVI
- NDMI
- Temperature
- Dew Point Temperature
- Precipitation
- Wind Components
- Surface Pressure
- Soil Water
- Elevation
- Slope
- Aspect

## Model Training

The dataset is divided into training and testing sets using a stratified split.

Since forest fire data can contain class imbalance, `scale_pos_weight` is used during model training.

Hyperparameter tuning is performed using **RandomizedSearchCV** with **3-fold Stratified Cross-Validation**.

The model uses **ROC-AUC** as the scoring metric for hyperparameter selection.

## Best Hyperparameters

The best parameters obtained during hyperparameter tuning are:

- `n_estimators`: 250
- `max_depth`: 10
- `learning_rate`: 0.03
- `subsample`: 0.7
- `colsample_bytree`: 0.7

## Model Performance

The tuned XGBoost model achieved:

- **Best CV ROC-AUC:** 0.9299
- **Test Accuracy:** 84.9%
- **Test ROC-AUC:** 0.9305

### Classification Report

| Class | Precision | Recall | F1-Score |
|------|-----------|--------|----------|
| 0 | 0.876 | 0.806 | 0.840 |
| 1 | 0.827 | 0.890 | 0.857 |

### Confusion Matrix

`[[4781, 1148], [675, 5483]]`

## Model Evaluation

The model is evaluated using:

- Accuracy
- ROC-AUC Score
- Classification Report
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance
- Threshold Analysis

## Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

## Model Output

The trained XGBoost model is saved in JSON format for future use.
