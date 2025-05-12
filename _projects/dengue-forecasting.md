---
layout: project
title: "Forecasting Dengue Cases with Time Series Analysis"
categories: time-series public-health machine-learning
image: /assets/images/placeholder.svg
technologies: [Python, Time Series, Prophet, Statsmodels, Cost-Benefit Analysis]
github: https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis
---

## Project Overview

Built a time series forecasting model and conducted a health economics analysis to support dengue intervention planning in Singapore. This project combined predictive modeling with cost-benefit analysis to provide actionable public health recommendations.

## Forecasting Methodology

### Data Integration
- Preprocessed and harmonized weather, population, Google Trends, and infectious disease datasets 
- Created a unified weekly time series spanning 2012–2022
- Applied statistical tests to verify stationarity and seasonality patterns

### Model Development and Evaluation
- Developed and compared 7 time series models:
  - ARIMA & SARIMA
  - SARIMAX with exogenous variables
  - Holt-Winters exponential smoothing
  - BATS & TBATS 
  - Facebook's Prophet
- Evaluated performance using MAPE (Mean Absolute Percentage Error)
- Achieved MAPE of 0.0952 on test set using Facebook's Prophet model
- Forecast dengue cases 16 weeks ahead with confidence intervals

## Cost-Benefit Analysis

### Intervention Comparison
- Evaluated two major dengue control strategies:
  1. **Wolbachia mosquito release program**
  2. **Dengvaxia® vaccination campaign**
- Calculated cost per DALY (Disability-Adjusted Life Year) averted

### Economic Findings
- Found Wolbachia mosquito program to be 6× more cost-effective 
- Cost per DALY averted:
  - Wolbachia: USD 60,000
  - Dengvaxia®: USD 360,000
- Compared results to WHO cost-effectiveness thresholds

## Code Sample

```python
# Time series forecasting with Prophet
from prophet import Prophet

# Prepare data for Prophet
prophet_df = df_weekly[['date', 'cases']].rename(columns={'date': 'ds', 'cases': 'y'})

# Initialize and fit Prophet model
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    seasonality_mode='multiplicative',
    changepoint_prior_scale=0.05
)

# Add additional regressors
for regressor in ['temperature', 'rainfall', 'google_trends']:
    if regressor in df_weekly.columns:
        model.add_regressor(regressor)

# Fit model
model.fit(prophet_df)

# Create future dataframe for forecasting
future = model.make_future_dataframe(periods=16, freq='W')
for regressor in ['temperature', 'rainfall', 'google_trends']:
    if regressor in df_weekly.columns:
        # Add known values for the regressors
        future[regressor] = df_weekly[regressor]

# Generate forecast
forecast = model.predict(future)
```

## Results and Recommendations

The project provided actionable insights for public health planning:

1. **Forecasting Tool**: Delivered a reliable forecasting model capable of predicting dengue outbreaks 16 weeks in advance
2. **Intervention Strategy**: Recommended prioritizing Wolbachia mosquito programs over vaccination based on cost-effectiveness
3. **Resource Allocation**: Provided data-driven guidance on timing and scale of interventions
4. **Economic Justification**: Supplied cost-benefit analysis for public health budget allocation

## Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation
- **Statsmodels** - Statistical modeling and ARIMA/SARIMA implementation
- **Prophet** - Facebook's time series forecasting tool
- **Matplotlib/Seaborn** - Data visualization
- **Scikit-learn** - Model evaluation metrics
