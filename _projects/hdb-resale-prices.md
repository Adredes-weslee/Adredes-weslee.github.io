---
layout: project
title: "Predicting HDB Resale Prices in Singapore"
categories: machine-learning regression real-estate
image: /assets/images/placeholder.svg
technologies: [Python, Linear Regression, Regularization, Scikit-Learn, Pandas, Feature Engineering]
github: https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price
---

## Project Overview

Built a machine learning model to forecast HDB flat resale prices using multivariate regression and domain-specific feature engineering. This project combined Singapore's unique property characteristics with advanced regression techniques to achieve highly accurate price predictions.

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
- **Feature Selection**: Applied LASSO regularization for dimensionality reduction

### Model Development and Evaluation
- **Base Models**: Implemented multiple linear regression as baseline
- **Regularization**: Applied Ridge and Lasso regression to prevent overfitting
- **Hyperparameter Tuning**: Grid search for optimal regularization parameters
- **Validation Strategy**: K-fold cross-validation for reliable performance estimates
- **Performance Metrics**: Achieved RMSE of ~39,180 SGD and R² = 0.9261

## Key Insights

The project revealed several important factors affecting HDB resale prices:

1. **Location Premium**: Central regions commanded significant price premiums
2. **Lease Decay Effect**: Quantified the impact of remaining lease on property values
3. **Floor Level Impact**: Higher floors consistently correlated with higher prices
4. **Amenity Value**: Proximity to MRT stations and schools added measurable value
5. **Policy Effects**: Identified price patterns following policy changes

## Code Sample

```python
# Feature Engineering for HDB Resale Price Prediction
import pandas as pd
import numpy as np
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.impute import IterativeImputer
from sklearn.linear_model import Ridge, Lasso, LinearRegression

# Create proximity features
def create_proximity_features(df):
    # Calculate distances to nearest amenities
    df['dist_to_mrt'] = calculate_distance(df['latitude'], df['longitude'], 
                                          mrt_stations['latitude'], mrt_stations['longitude'])
    df['dist_to_school'] = calculate_distance(df['latitude'], df['longitude'],
                                             schools['latitude'], schools['longitude'])
    
    # Create binary features for amenities within walking distance
    df['near_mrt'] = df['dist_to_mrt'] < 0.5  # Within 500m
    df['near_school'] = df['dist_to_school'] < 1.0  # Within 1km
    
    return df

# Lease remaining calculation
df['lease_remaining'] = df['lease_commence_date'] + 99 - df['transaction_year']
df['lease_remaining_pct'] = df['lease_remaining'] / 99

# Model training and evaluation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Try different regression models
models = {
    'Linear': LinearRegression(),
    'Ridge': Ridge(alpha=1.0),
    'Lasso': Lasso(alpha=0.01)
}

results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results[name] = {
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred)),
        'R2': r2_score(y_test, y_pred)
    }
```

## Results and Impact

The final model achieved impressive prediction accuracy with an RMSE of ~39,180 SGD, representing less than 8% error on the average HDB resale price. The high R² value of 0.9261 indicates that the model explains over 92% of the price variance.

### Recommendations Derived
1. **Buyer Insights**: Quantified the price impact of different features to help buyers make informed decisions
2. **Policy Implications**: Identified how housing policies affect market pricing
3. **Affordability Analysis**: Projected price trends to assess future affordability
4. **Lease Decay Insights**: Provided data-driven understanding of lease decay effect on property values

## Technologies Used
- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - Machine learning implementation
- **NumPy** - Numerical computing
- **Matplotlib/Seaborn** - Data visualization
- **Statsmodels** - Statistical modeling
