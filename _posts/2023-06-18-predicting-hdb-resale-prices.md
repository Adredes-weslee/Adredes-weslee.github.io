---
layout: post
title: "Building an HDB Resale Price Predictor: A Technical Deep Dive into Feature Engineering and Regression"
date: 2023-06-18 14:45:00 +0800 # Retaining original date
categories: [data-science, machine-learning, real-estate]
tags: [housing, singapore, regression, feature-engineering, price-prediction, python, scikit-learn, pandas]
author: Wes Lee
feature_image: /assets/images/2023-06-18-predicting-hdb-resale-prices.jpg # Or a new, more technical image
---

## Introduction: Decoding Singapore's Unique Housing Market with Data

Singapore's public housing (HDB flats) market is a dominant feature of its urban landscape, housing over 80% of the population. Predicting HDB resale prices is not just an academic exercise; it has significant financial implications for citizens and provides valuable insights for urban planning. This post offers a technical walkthrough of a project aimed at building a machine learning model to forecast these prices, detailing the data processing, feature engineering, modeling techniques, and key lessons learned.

> For a higher-level overview of this project's significance, key findings, and policy implications for Singapore, please see the [*HDB Resale Price Insights: A Data-Driven Analysis for Singapore's Housing Market* Project Page](/projects/hdb-resale-prices/).

## The Data Science Approach: From Raw Data to Price Prediction

Our goal was to build an accurate and interpretable model for HDB resale prices. This involved several key stages.

### 1. Dataset Acquisition and Initial Exploration
The project utilized a Kaggle dataset comprising over 60,000 HDB resale transactions. Initial variables included:
-   Physical flat attributes: floor area, flat type (e.g., 3 ROOM, 4 ROOM, EXEC), storey range.
-   Location information: town, block, street name.
-   Lease details: lease commencement year.
-   Transaction details: resale price, transaction month and year.

### 2. Advanced Feature Engineering: The Core of an Accurate Model
Raw data rarely tells the whole story. Creating meaningful features (feature engineering) is crucial, especially in a market as nuanced as Singapore's HDB.

**a. Lease-Related Features:**
The remaining lease on an HDB flat is a critical price determinant. We engineered several lease-based features:

```python
import pandas as pd
import numpy as np

# Assume 'df' is your DataFrame with HDB transaction data
# df['lease_commence_year'] = df['lease_commence_date'].dt.year # If lease_commence_date is datetime
# df['transaction_year'] = df['month'].dt.year # If 'month' is datetime

def create_lease_features(df_input):
    df = df_input.copy() # Work on a copy to avoid SettingWithCopyWarning
    # Calculate remaining lease in years at the time of transaction
    # HDB leases are typically 99 years
    df['remaining_lease_years'] = df['lease_commence_year'] + 99 - df['transaction_year']
    
    # Create binary indicators for critical lease decay thresholds
    # Research and market observations suggest price impacts at these points
    df['lease_lt_60_years'] = (df['remaining_lease_years'] < 60).astype(int)
    df['lease_lt_40_years'] = (df['remaining_lease_years'] < 40).astype(int)
    
    # Calculate percentage of lease remaining
    df['lease_remaining_percentage'] = df['remaining_lease_years'] / 99.0
    
    # Interaction term: floor area * remaining lease (value per sqm might change with lease)
    if 'floor_area_sqm' in df.columns:
        df['floor_area_x_rem_lease_pct'] = df['floor_area_sqm'] * df['lease_remaining_percentage']
        
    return df

# Example usage:
# hdb_data_with_lease_features = create_lease_features(hdb_raw_data)
# print(hdb_data_with_lease_features[['remaining_lease_years', 'lease_lt_60_years', 'lease_remaining_percentage']].head())
```
This captures the non-linear impact of lease decay, where depreciation often accelerates as the lease shortens, particularly below key thresholds like 60 or 40 years remaining.

**b. Location and Proximity Features:**
Location is paramount. Beyond just the 'town', we would ideally engineer features like:
-   Distance to the nearest MRT station.
-   Distance to Central Business District (CBD).
-   Number of schools within a certain radius (e.g., 1km, 2km).
-   Proximity to shopping malls and other key amenities.
*(These often require joining with external geospatial datasets or using APIs, which was abstracted in the original project description but is a key step).*

**c. Flat Characteristics:**
-   **Storey Level:** Convert storey range (e.g., "01 TO 03", "10 TO 12") to an average numerical value or ordinal encoding. Higher floors generally command higher prices.
-   **Flat Type and Model:** One-hot encode or use target encoding for categorical variables like `flat_type` (e.g., '3 ROOM', '4 ROOM', 'EXECUTIVE') and `flat_model` (e.g., 'Model A', 'DBSS', 'Improved').

### 3. Data Preprocessing: Preparing Data for Modeling
Robust preprocessing is essential for model performance and stability.
-   **Handling Missing Values:** For features with missing data, `IterativeImputer` from Scikit-learn was used. This is an advanced imputer that models each feature with missing values as a function of other features, and uses that estimate for imputation.
    ```python
    from sklearn.experimental import enable_iterative_imputer # Enable experimental feature
    from sklearn.impute import IterativeImputer
    
    # Assuming X_train_numeric is your training data with only numeric features
    # imputer = IterativeImputer(max_iter=10, random_state=0)
    # X_train_numeric_imputed = imputer.fit_transform(X_train_numeric)
    # X_test_numeric_imputed = imputer.transform(X_test_numeric) # Use transform on test data
    ```
-   **Categorical Variable Encoding:**
    * For nominal categorical features (like `town` or `flat_model` if no inherent order), One-Hot Encoding is common.
    * For ordinal features (like `storey_range` if converted to ordered categories), Ordinal Encoding can be used.
    * The project mentioned Polynomial Encoding, which can capture interactions between categorical features but can lead to a high number of features.
-   **Feature Scaling:** Numerical features were standardized using `StandardScaler` to bring them to a similar scale, which is important for many regression algorithms, especially those with regularization.
    ```python
    from sklearn.preprocessing import StandardScaler
    
    # scaler = StandardScaler()
    # X_train_scaled = scaler.fit_transform(X_train_numeric_imputed) # Fit on train, then transform
    # X_test_scaled = scaler.transform(X_test_numeric_imputed)   # Transform test
    ```
-   **Multicollinearity Analysis:** Variance Inflation Factor (VIF) analysis was conducted to identify and potentially remove or combine highly correlated features, which can destabilize linear models.
-   **Feature Selection:**
    * **Mutual Information:** To assess the relationship between each feature and the target variable (resale price).
    * **LASSO Regularization:** Used not only for modeling but also implicitly for feature selection, as it tends to shrink coefficients of less important features to zero.

### 4. Model Development and Evaluation: A Progressive Approach

We adopted an iterative modeling strategy:

**a. Baseline Model:**
A simple Multiple Linear Regression model was implemented first with a basic set of features to establish a performance baseline.

**b. Regularized Regression Models:**
To prevent overfitting and handle potential multicollinearity from the many engineered features, Ridge (L2 regularization) and Lasso (L1 regularization) regression models were employed.
-   **Ridge Regression:** Adds a penalty equal to the square of the magnitude of coefficients. It shrinks coefficients but rarely to zero.
-   **Lasso Regression:** Adds a penalty equal to the absolute value of the magnitude of coefficients. It can shrink some coefficients exactly to zero, effectively performing feature selection.

**c. Hyperparameter Tuning:**
The regularization strength (alpha) for Ridge and Lasso is a critical hyperparameter. `GridSearchCV` was used to find the optimal alpha value by evaluating model performance across a range of alpha values using k-fold cross-validation.

```python
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

# Assume X_processed, y_target are your fully preprocessed features and target variable
# X_train, X_test, y_train, y_test = train_test_split(X_processed, y_target, test_size=0.2, random_state=42)

# Ridge Regression with GridSearchCV
ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]} # Range of alpha values
ridge_model = Ridge()
ridge_cv = GridSearchCV(ridge_model, ridge_params, cv=5, scoring='neg_mean_squared_error')
# ridge_cv.fit(X_train_scaled, y_train) # Use scaled training data

# best_ridge_alpha = ridge_cv.best_params_['alpha']
# print(f"Best alpha for Ridge: {best_ridge_alpha}")
# best_ridge_model = ridge_cv.best_estimator_

# Lasso Regression with GridSearchCV (similar setup)
# lasso_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]} # Lasso often needs smaller alphas
# lasso_model = Lasso(max_iter=10000) # Increase max_iter if it doesn't converge
# lasso_cv = GridSearchCV(lasso_model, lasso_params, cv=5, scoring='neg_mean_squared_error')
# lasso_cv.fit(X_train_scaled, y_train)
# best_lasso_model = lasso_cv.best_estimator_

# Evaluate the best performing model (e.g., Ridge)
# y_pred_ridge = best_ridge_model.predict(X_test_scaled)
# rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
# r2_ridge = r2_score(y_test, y_pred_ridge)

# print(f"Final Ridge Model RMSE: {rmse_ridge:.2f} SGD") # Original: ~39,180 SGD
# print(f"Final Ridge Model R-squared: {r2_ridge:.4f}")   # Original: 0.9261
```

**d. Validation Strategy:**
K-fold cross-validation (typically 5 or 10 folds) was used during hyperparameter tuning to ensure that the model's performance estimates were robust and not overly dependent on a particular train-test split.

**e. Performance Metrics:**
-   **Root Mean Squared Error (RMSE):** Measures the standard deviation of the residuals (prediction errors). An RMSE of ~39,180 SGD was achieved.
-   **R-squared (R²):** Represents the proportion of the variance in the dependent variable that is predictable from the independent variables. An R² of 0.9261 indicates that the model explains about 92.61% of the variability in HDB resale prices.

## Technical Lessons Learned and Best Practices

1.  **Domain Knowledge is Gold:** Understanding the nuances of the HDB market (e.g., the 99-year lease structure, flat types, town maturity) was paramount for effective feature engineering. The "lease decay cliffs" are a prime example.
2.  **Iterative Feature Engineering:** The process wasn't linear. We often revisited feature engineering based on model performance and error analysis.
3.  **Handling Multicollinearity:** In real estate data, many features are inherently correlated (e.g., flat size and number of rooms, proximity to MRT and proximity to CBD). Regularization (Ridge/Lasso) and VIF analysis are crucial.
4.  **Importance of Scaling:** For distance-based algorithms or those using regularization, feature scaling (like StandardScaler) is essential for fair coefficient penalization and faster convergence.
5.  **Interpretability vs. Accuracy Trade-off:** While more complex models (e.g., Gradient Boosting, Neural Networks) might offer marginal improvements in RMSE, linear models (especially regularized ones) provide better interpretability of feature importance, which is often valuable for stakeholders and policy insights.
6.  **Robust Validation:** K-fold cross-validation is more reliable than a single train-test split for assessing model generalization and tuning hyperparameters.

## Future Technical Enhancements

While the model performed well, there's always room for improvement:
-   **Geospatial Features:** More sophisticated geospatial features (e.g., using GIS data for precise amenity distances, view quality proxies, noise levels).
-   **Time Series Modeling:** Explicitly modeling temporal trends and seasonality in prices (e.g., using ARIMA with exogenous variables or Prophet) if forecasting future prices is a goal, rather than just explaining current price variations.
-   **Non-Linear Models:** Exploring models like Random Forests, Gradient Boosting (XGBoost, LightGBM), or Neural Networks to capture more complex, non-linear relationships, potentially at the cost of some interpretability.
-   **Interaction Terms:** Systematically exploring interaction terms between key features (e.g., `remaining_lease * is_central_region`).
-   **Hyperlocal Models:** Training separate models for distinct regions or towns if data permits, as price dynamics can vary significantly.

## Conclusion

Predicting HDB resale prices is a challenging yet rewarding data science problem. This project demonstrated that a combination of thoughtful feature engineering grounded in domain knowledge, robust preprocessing, and appropriate regularized regression techniques can yield highly accurate and interpretable models. The insights derived not only aid individual homebuyers but also provide a quantitative basis for housing policy and urban development in Singapore.

---

*This post details the technical methodologies used in the HDB Resale Price Prediction project. For more on the project's significance, key findings, and policy implications, please visit the [project page](/projects/hdb-resale-prices/). The source code is available on [GitHub](https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price).*
