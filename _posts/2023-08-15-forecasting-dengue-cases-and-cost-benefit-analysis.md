---
layout: post
title: "Dengue Outbreak Prediction: A Technical Guide to Time Series Forecasting and Health Economic Analysis"
date: 2023-08-15 10:00:00 +0800 # Retaining original date
categories: [time-series, public-health, economics]
tags: [forecasting, prophet, arima, sarimax, bats, tbats, health-economics, cost-benefit-analysis, python, pandas, statsmodels, singapore, dengue]
author: Wes Lee
feature_image: /assets/images/2023-08-15-forecasting-dengue-cases-and-cost-benefit-analysis.jpg # Or a new, more technical image
---

## Introduction: The Dual Challenge of Dengue Prediction and Policy

Dengue fever is a persistent and escalating public health concern in many tropical regions, including Singapore. The 2020 outbreak, with over 30,000 cases, underscored the urgent need for robust predictive tools and economically sound intervention strategies. This post offers a technical walkthrough of a project that tackled this dual challenge by combining advanced time series forecasting of dengue cases with a detailed health economic cost-benefit analysis of control interventions.

> For a higher-level overview of this project's public health context, strategic findings, and policy recommendations for Singapore, please see the [*Strategic Dengue Control: Forecasting & Cost-Benefit Analysis for Public Health Interventions in Singapore* Project Page](/projects/dengue-forecasting/).

## Phase 1: Building the Foundation - Data Integration and Preparation

Effective forecasting starts with comprehensive and well-prepared data. Our approach involved several key steps:

### 1. Sourcing Diverse Data Streams
To capture the multifaceted nature of dengue transmission, we integrated data from various sources:
-   **Epidemiological Data:** Weekly dengue case counts from the Ministry of Health, Singapore (2012-2022).
-   **Meteorological Data:** Weekly averages/totals for temperature, relative humidity, and rainfall from the Meteorological Service Singapore.
-   **Demographic Data:** Population statistics from the Department of Statistics Singapore (used for context and potential rate calculations).
-   **Digital Behavioral Data:** Google Trends data for dengue-related search terms, as a proxy for public awareness or early symptom reporting.

### 2. Data Preprocessing and Harmonization
The raw data required significant preprocessing:
-   **Aggregation & Resampling:** All data was converted to a unified weekly time series.
-   **Timestamp Alignment:** Ensured consistent indexing across all datasets.
-   **Missing Value Imputation:** Handled missing data points using appropriate techniques (e.g., interpolation, forward/backward fill).
-   **Feature Engineering:** Created lagged variables for weather data, as their impact on mosquito breeding cycles and viral incubation is often delayed.
-   **Dataset Splitting:** The final dataset (152 weekly data points) was split into a training set (114 points) and a test set (38 points) for model evaluation.

## Phase 2: Understanding Dengue Dynamics - Time Series Analysis

Before modeling, we thoroughly analyzed the time series properties of dengue cases in Singapore.

### 1. Identifying Seasonality and Trends
Dengue incidence exhibits complex seasonality:
-   **Annual Cycles:** Peaks often occur in the warmer, wetter months (typically June-October).
-   **Multi-year Epidemic Cycles:** Larger outbreaks tend to occur every 5-6 years, potentially driven by shifts in dominant dengue virus serotypes and population immunity.
-   **Weekly Patterns:** Minor fluctuations can be related to reporting artifacts.

Correlation analysis revealed lagged relationships: temperature showed significant correlation with a 3-4 week lag, while rainfall's impact was more complex with 1-3 week lags.

### 2. Stationarity Testing: A Critical Step
Most time series models assume stationarity (i.e., statistical properties like mean and variance are constant over time). We used two common tests:
-   **Augmented Dickey-Fuller (ADF) Test:** Null hypothesis (H0) is that the series is non-stationary.
-   **Kwiatkowski-Phillips-Schmidt-Shin (KPSS) Test:** Null hypothesis (H0) is that the series is stationary around a deterministic trend.

```python
# Python code for stationarity testing
from statsmodels.tsa.stattools import adfuller, kpss
import pandas as pd # Assuming df is your DataFrame with a 'cases' column

# Load your weekly dengue cases data into df['cases']
# df = pd.read_csv('your_dengue_data.csv', parse_dates=['date_column'], index_col='date_column')
# df_weekly_cases = df['cases'].resample('W').sum() # Example resampling

# Augmented Dickey-Fuller Test
# print("Results of Dickey-Fuller Test:")
# adf_result = adfuller(df_weekly_cases, autolag='AIC')
# adf_output = pd.Series(adf_result[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
# for key,value in adf_result[4].items():
#    adf_output['Critical Value (%s)'%key] = value
# print(adf_output)
# If p-value < 0.05, reject H0 (series is stationary). Original p=0.0037.

# KPSS Test
# print("\nResults of KPSS Test:")
# kpss_result = kpss(df_weekly_cases, regression='c', nlags="auto") # 'c' for constant trend
# kpss_output = pd.Series(kpss_result[0:3], index=['Test Statistic','p-value','Lags Used'])
# for key,value in kpss_result[3].items():
#    kpss_output['Critical Value (%s)'%key] = value
# print(kpss_output)
# If p-value > 0.05, fail to reject H0 (series is stationary). Original p=0.1.
```
Our data (ADF p=0.0037, KPSS p=0.1) suggested difference stationarity, meaning differencing the series could make it stationary. This informed the choice of `d` or `D` parameters in ARIMA/SARIMA models.

### 3. Autocorrelation Analysis (ACF and PACF)
Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF) plots helped identify the order of AR (p) and MA (q) terms for ARIMA models. Strong weekly dependence (ACF at lag 1 = 0.9768) was observed, diminishing over time.

## Phase 3: Developing and Evaluating Forecasting Models

We developed and compared seven different time series models to find the best predictor for dengue cases.

### 1. Overview of Models Tested:
-   **ARIMA (AutoRegressive Integrated Moving Average):** A classic model for univariate time series.
-   **SARIMA (Seasonal ARIMA):** Extends ARIMA to handle seasonality.
-   **SARIMAX (SARIMA with eXogenous variables):** Allows inclusion of external predictors like weather data.
-   **Holt-Winters Exponential Smoothing:** A method that explicitly models level, trend, and seasonality.
-   **BATS (Box-Cox transform, ARMA errors, Trend, and Seasonal components):** Handles complex seasonalities and other components.
-   **TBATS (Trigonometric BATS):** An extension of BATS that can model multiple complex seasonalities using trigonometric functions.
-   **Prophet (by Facebook):** A decomposable model that explicitly models trend, multiple seasonalities (yearly, weekly), and holidays/special events. It's robust to missing data and shifts in trend.

### 2. Spotlight on Prophet: The Best Performing Model
The Prophet model demonstrated the best performance in our evaluations.

**Implementation with Prophet:**
```python
# Python code for Prophet model implementation
from prophet import Prophet
import pandas as pd # Ensure pandas is imported

# Assume 'df_train' is the training DataFrame with 'ds' (datetime) and 'y' (cases) columns,
# and potentially other regressor columns like 'temperature', 'relative_humidity', 'rainfall'.
# df_train = pd.DataFrame({
#     'ds': pd.to_datetime(['2022-01-01', '2022-01-08', ...]),
#     'y': [10, 12, ...],
#     'temperature': [28, 29, ...],
#     # ... other regressors
# })


# Initialize Prophet model
# Seasonality_mode can be 'additive' or 'multiplicative'
# Changepoint_prior_scale controls flexibility of trend changes
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True, # Captures weekly reporting patterns if any
    daily_seasonality=False, # Not applicable for weekly data
    seasonality_mode='multiplicative', # Assumes seasonal effects multiply the trend
    changepoint_prior_scale=0.05 # Default is 0.05; adjusts trend flexibility
)

# Add exogenous regressors (e.g., weather variables)
# These regressors must be known for future periods as well for forecasting
# model.add_regressor('temperature')
# model.add_regressor('relative_humidity')
# model.add_regressor('rainfall')
# model.add_regressor('google_trends_dengue') # Example

# Fit the model to the training data
# model.fit(df_train)

# Create a future DataFrame for forecasting (e.g., 16 weeks ahead)
# future_periods = 16
# future_df = model.make_future_dataframe(periods=future_periods, freq='W') # 'W' for weekly frequency

# Add future values for regressors to future_df
# This is crucial: you need forecasts or actual future values for your regressors.
# For example, if using weather forecasts:
# future_df['temperature'] = future_weather_forecasts['temperature'] 
# ... and so on for other regressors.
# If regressors are not available for the future, they cannot be used directly in this way,
# or you might need to forecast them separately.

# Generate forecast
# forecast_df = model.predict(future_df)

# Display key forecast components
# print(forecast_df[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(future_periods))

# Plotting the forecast
# fig1 = model.plot(forecast_df)
# plt.show()

# Plotting components (trend, yearly seasonality, weekly seasonality)
# fig2 = model.plot_components(forecast_df)
# plt.show()
```

### 3. Model Evaluation and Cross-Validation
We used Mean Absolute Percentage Error (MAPE), Root Mean Square Error (RMSE), and Mean Absolute Error (MAE) for evaluation. Prophet's built-in cross-validation tools were also utilized.

```python
# Prophet's cross-validation and performance metrics
from prophet.diagnostics import cross_validation, performance_metrics

# Perform cross-validation
# 'initial': size of the initial training period
# 'period': spacing between cutoff dates
# 'horizon': forecast horizon
# df_cv = cross_validation(model, initial='730 days', period='90 days', horizon = '112 days') # Example parameters

# Calculate performance metrics
# performance_metrics_df = performance_metrics(df_cv)
# print("Prophet Model Performance Metrics (from CV):")
# print(performance_metrics_df.head())
# print(f"Mean MAPE from CV: {performance_metrics_df['mape'].mean():.4f}") # Target MAPE was 0.0952
```
The Prophet model achieved a MAPE of 0.0952 on the test set, outperforming other models, particularly in handling multiple seasonalities and incorporating external regressors.

## Phase 4: Health Economic Analysis - The "How To"

Beyond prediction, the project aimed to evaluate the economic viability of dengue control interventions.

### 1. Defining Interventions for Comparison:
-   **Project Wolbachia:** A biocontrol method using *Wolbachia*-infected mosquitoes.
-   **Dengvaxia® Vaccination:** A population-level immunization campaign.

### 2. Calculating Key Health Economic Metrics:
The core of the cost-benefit analysis involved:
-   **Estimating Implementation Costs:** Gathering data on the costs associated with rolling out each intervention (e.g., mosquito rearing and release, vaccine procurement, administration, public awareness campaigns, serological screening for Dengvaxia®).
-   **Calculating DALYs (Disability-Adjusted Life Years) Averted:** This involved estimating the reduction in disease burden (morbidity and mortality) due to each intervention. DALYs = Years of Life Lost (YLL) due to premature mortality + Years Lived with Disability (YLD).
-   **Cost per DALY Averted:** Calculated as: `Total Intervention Cost / Total DALYs Averted`. This is a standard metric for comparing the cost-effectiveness of health interventions.
-   **Benefit-Cost Ratio (BCR):** Calculated as: `Total Monetized Benefits / Total Intervention Costs`. Benefits include averted healthcare costs, productivity losses prevented, etc.

```python
# Python snippet for simplified cost-effectiveness calculation

def calculate_cost_per_daly(total_intervention_cost, total_dalys_averted):
    """Calculates the cost per DALY averted."""
    if total_dalys_averted == 0:
        return float('inf') # Avoid division by zero
    return total_intervention_cost / total_dalys_averted

# Example data (replace with actuals from the study)
# For Project Wolbachia in Singapore (annualized)
wolbachia_annual_cost_usd = 27000000 
wolbachia_annual_dalys_averted = 449.7 # Example value

cost_per_daly_wolbachia = calculate_cost_per_daly(wolbachia_annual_cost_usd, wolbachia_annual_dalys_averted)
# print(f"Project Wolbachia - Cost per DALY averted: USD {cost_per_daly_wolbachia:,.0f}")
# Expected: USD 60,039

# For Dengvaxia® Vaccination in Singapore (annualized, hypothetical campaign)
dengvaxia_annual_cost_usd = 36320000 
dengvaxia_annual_dalys_averted = 100.6 # Example value

cost_per_daly_dengvaxia = calculate_cost_per_daly(dengvaxia_annual_cost_usd, dengvaxia_annual_dalys_averted)
# print(f"Dengvaxia Vaccination - Cost per DALY averted: USD {cost_per_daly_dengvaxia:,.0f}")
# Expected: USD 360,876
```

### 3. Comparing with WHO Cost-Effectiveness Thresholds
The calculated cost per DALY averted was then compared against WHO-recommended thresholds (e.g., <1x GDP per capita is highly cost-effective, 1-3x GDP per capita is cost-effective).

## Technical Lessons and Future Directions

**Key Technical Learnings:**
-   **Multi-Model Evaluation:** Testing a diverse suite of time series models is crucial, as no single model is universally superior.
-   **Feature Engineering for Time Series:** Incorporating relevant exogenous variables (weather, search trends) and their appropriate lags significantly improves forecast accuracy.
-   **Robustness of Prophet:** Prophet's ability to handle multiple seasonalities and trend changes made it particularly suitable for complex epidemiological data.
-   **Economic Modeling Complements Forecasting:** Combining predictive analytics with economic evaluation provides a much richer basis for policy decisions.

**Future Technical Enhancements:**
-   **Spatio-temporal Modeling:** Incorporating geographical data to predict localized outbreaks.
-   **Genomic Surveillance Data:** Integrating viral serotype information to improve accuracy during serotype shifts.
-   **Real-time Model Updating:** Developing a pipeline for continuous model retraining and forecast generation as new data becomes available.
-   **Ensemble Forecasting:** Combining predictions from multiple top-performing models to potentially achieve even greater accuracy and robustness.

## Conclusion: The Power of Integrated Analytics

This project demonstrated a comprehensive workflow for tackling a complex public health issue like dengue. By meticulously integrating diverse data, applying a range of time series forecasting techniques, and performing a rigorous health economic analysis, we were able to generate actionable, evidence-based recommendations. The journey from raw data to policy insights highlights the power of a multi-disciplinary data science approach.

---

*For a deeper dive into the specific findings, policy recommendations, and overall impact of this work, please visit the [project page](/projects/dengue-forecasting/). The source code and datasets are available on [GitHub](https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis).*
