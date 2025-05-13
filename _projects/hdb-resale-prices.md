---
layout: project
title: "Predicting HDB Resale Prices in Singapore"
categories: machine-learning regression real-estate
image: /assets/images/hdb-resale-prices.jpg
technologies: [Python, Linear Regression, Regularization, Scikit-Learn, Pandas, Feature Engineering]
github: https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price
blog_post: /data-science/machine-learning/real-estate/2023/06/18/predicting-hdb-resale-prices.html
---

## Project Overview

Built a machine learning model to forecast HDB flat resale prices in Singapore, where over 80% of residents live in public housing. Using a dataset of 60,000+ transactions, this project combined Singapore's unique property characteristics with advanced regression techniques to achieve highly accurate price predictions.

## Background & Significance

In Singapore, public housing (HDB flats) forms the backbone of the residential property market, with homeownership rates around 90% - one of the highest in the world. Housing affordability and price prediction are crucial topics affecting citizens' financial planning, investment decisions, and quality of life.

This project addresses several key questions:
- Which factors most significantly influence HDB resale prices?
- How does remaining lease impact property valuation?
- What role do location and proximity to amenities play in price determination?
- How can data science provide transparency to the housing market?

## Methodology

### Data Collection and Integration
- Integrated 60,000+ resale transactions from Kaggle
- Engineered 30+ features including flat type, location, lease length, and proximity to amenities
- Applied domain knowledge of Singapore's housing market to create meaningful features

### Data Preprocessing and Feature Engineering
- **Advanced Imputation**: Used IterativeImputer to handle missing values
- **Feature Encoding**: Implemented polynomial encoding for categorical variables
- **Feature Scaling**: Normalized numerical features for model consistency
- **Multicollinearity Analysis**: Conducted VIF analysis to identify and handle correlated features
- **Feature Selection**: Applied mutual information and LASSO regularization for dimensionality reduction

### Model Development and Evaluation
- **Base Models**: Implemented multiple linear regression as baseline
- **Regularization**: Applied Ridge and Lasso regression to prevent overfitting
- **Hyperparameter Tuning**: Grid search for optimal regularization parameters
- **Validation Strategy**: K-fold cross-validation for reliable performance estimates
- **Performance Metrics**: Achieved RMSE of ~39,180 SGD and R² = 0.9261

## Key Insights

The project revealed several important factors affecting HDB resale prices:

1. **Lease Decay Effect**: Quantified how prices decline as lease falls below 60 years
2. **Flat Model Premiums**: Identified that DBSS and Model A flats fetch higher prices
3. **Location Premium**: Central regions commanded significant price premiums
4. **Floor Level Impact**: Higher floors consistently correlated with higher prices
5. **Size Matters**: Floor area remained a dominant factor in price determination
6. **Amenity Value**: Proximity to MRT stations and schools added measurable value
7. **Policy Effects**: Identified price patterns following policy changes

## Technical Implementation

```python
# Feature Engineering for HDB Resale Price Prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

# Create lease-related features
def create_lease_features(df):
    # Calculate remaining lease in years
    df['remaining_lease'] = df['lease_commence_year'] + 99 - df['transaction_year']
    
    # Create lease decay indicators
    df['lease_decay_60'] = df['remaining_lease'] < 60
    df['lease_decay_40'] = df['remaining_lease'] < 40
    
    # Calculate percentage of lease remaining
    df['lease_remaining_pct'] = df['remaining_lease'] / 99
    
    return df

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ridge regression with hyperparameter tuning
ridge_params = {'alpha': [0.01, 0.1, 1.0, 10.0, 100.0]}
ridge_cv = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='neg_mean_squared_error')
ridge_cv.fit(X_train, y_train)
ridge_best = ridge_cv.best_estimator_

# Evaluate best model
y_pred = ridge_best.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
```

## Policy Implications

Based on the analysis, several recommendations emerged:

1. **Lease Decay Transparency**: Implement clearer guidelines on valuation impact of remaining lease
2. **Differentiated Subsidies**: Consider targeted subsidies for lower-lease resale flats to maintain market stability
3. **Data-Driven Urban Planning**: Use predictive insights to guide future development
4. **Amenity Distribution**: Balance amenity development across regions to reduce price disparities
5. **Market Monitoring**: Apply predictive models to detect potential pricing anomalies or bubbles

## Future Work

- Incorporate additional features like renovation status and interior quality
- Develop neighborhood-specific models for more localized predictions
- Create an interactive web application for homebuyers to estimate prices
- Analyze price trends over time to identify emerging patterns
- Integrate macroeconomic factors to enhance prediction accuracy

## References

The project drew on extensive resources including HDB policies, Singapore urban planning documents, and statistical methodologies. A comprehensive bibliography is available in the project repository.
