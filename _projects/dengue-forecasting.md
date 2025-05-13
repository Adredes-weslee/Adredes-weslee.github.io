---
layout: project
title: "Forecasting Dengue Cases with Time Series Analysis"
categories: time-series public-health machine-learning
image: /assets/images/dengue-forecasting.jpg
technologies: [Python, Time Series, Prophet, ARIMA, SARIMA, SARIMAX, BATS, TBATS, Health Economics, Cost-Benefit Analysis]
github: https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis
blog_post: /time-series/public-health/economics/2023/08/15/forecasting-dengue-cases-and-cost-benefit-analysis.html
---

## Project Overview

Built a comprehensive time series forecasting model and conducted a detailed health economics analysis to support evidence-based dengue intervention planning in Singapore. This project combines advanced predictive modeling with rigorous cost-benefit analysis to provide actionable public health recommendations for dengue prevention and control strategies.

> Read my detailed blog post: [Forecasting Dengue Cases and Conducting Cost-Benefit Analysis](/time-series/public-health/economics/2023/08/15/forecasting-dengue-cases-and-cost-benefit-analysis.html)

## Public Health Context

Dengue fever poses a significant public health challenge in tropical urban environments, including Singapore. With rising temperatures and evolving viral serotypes, predicting case loads and evaluating intervention strategies have become critical to public health planning.

- Singapore recorded over 30,000 dengue cases in 2020, the worst outbreak year
- Dengue control costs Singapore approximately SGD 100 million annually
- Mortality rates, economic costs, and healthcare burden make this a priority health concern

## Forecasting Methodology

### Data Integration
- Preprocessed and harmonized multiple datasets:
  - Weekly dengue case counts from Ministry of Health (2012-2022)
  - Weather data (temperature, humidity, rainfall) from Meteorological Service
  - Population statistics from Department of Statistics
  - Google Trends data for search terms related to dengue symptoms
- Created a unified weekly time series with 114 data points for training, 38 points for testing
- Applied statistical tests (ADF, KPSS, Seasonal Strength) to verify stationarity and seasonality patterns

### Statistical Testing & Time Series Properties
- Conducted comprehensive stationarity testing with contrasting ADF (p=0.0037) and KPSS (p=0.1) tests
- Identified difference stationarity in the time series
- Autocorrelation analysis showed strong weekly dependence (0.9768) with diminishing influence over time
- Determined appropriate model parameters through ACF and PACF plots

### Model Development and Evaluation
- Developed and rigorously compared 7 time series models:
  - ARIMA & SARIMA (Box-Jenkins methodology)
  - SARIMAX with exogenous weather variables
  - Holt-Winters exponential smoothing
  - BATS & TBATS (handling complex seasonality)
  - Facebook's Prophet (decomposable model with trend, seasonality components)
- Evaluated performance using multiple metrics:
  - MAPE (Mean Absolute Percentage Error)
  - RMSE (Root Mean Square Error)
  - MAE (Mean Absolute Error)
- Achieved best performance with Prophet model:
  - MAPE of 0.0952 on test set
  - Superior handling of seasonality and trend components
- Generated forecasts 16 weeks ahead with confidence intervals
- Conducted feature importance analysis to identify key predictive factors

## Cost-Benefit Analysis

### Intervention Comparison
- Performed comprehensive health economic evaluation of two major dengue control strategies:
  1. **Wolbachia mosquito release program** - A biocontrol approach using mosquitoes infected with Wolbachia bacteria to reduce dengue transmission
  2. **Dengvaxia® vaccination campaign** - Population-level immunization with the dengue vaccine
- Calculated key health economic metrics:
  - Implementation costs for each intervention
  - Disability-Adjusted Life Years (DALYs) averted
  - Cost per DALY averted
  - Benefit-Cost Ratio (BCR)
- Compared Singapore's results to regional implementations in Vietnam and Jakarta

### Economic Findings
- Found Project Wolbachia to be substantially more cost-effective than vaccination:
  - Cost per DALY averted with Wolbachia: USD 60,039
  - Cost per DALY averted with Dengvaxia®: USD 360,876 (6× higher)
- Benefit-Cost Ratio for Project Wolbachia in Singapore: 2.90
  - Annual cost: USD 27 million
  - Annual benefit: USD 78.4 million
- Compared results to WHO cost-effectiveness thresholds:
  - 0.5× GDP per capita threshold: USD 30,364
  - 3× Gross National Income threshold: USD 166,255
- Concluded that while Project Wolbachia exceeds the conservative 0.5× GDP threshold, it falls well within the 3× GNI threshold appropriate for high HDI countries like Singapore

## Policy Implications & Recommendations

Based on our combined forecasting and economic analysis:

1. **Expand Project Wolbachia** implementation island-wide as a cost-effective dengue control measure
2. **Target interventions seasonally** based on forecasting model predictions
3. **Reserve vaccination** for high-risk individuals rather than population-level implementation
4. **Integrate forecasting tools** into public health planning processes
5. **Continue monitoring** intervention effectiveness and adjust strategies accordingly

## Code Sample

```python
# Time series forecasting with Prophet
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics

# Initialize model with desired parameters
model = Prophet(
    yearly_seasonality=True,
    weekly_seasonality=True,
    daily_seasonality=False,
    changepoint_prior_scale=0.05
)

# Add regressors for weather variables
model.add_regressor('temperature')
model.add_regressor('relative_humidity')
model.add_regressor('rainfall')

# Fit the model to historical data
model.fit(df_train)

# Create future dataframe for forecasting
future = model.make_future_dataframe(periods=16, freq='W')
future = add_regressor_values(future, weather_data)  # Add weather data

# Generate forecast
forecast = model.predict(future)

# Evaluate model performance on test set
df_cv = cross_validation(model, initial='730 days', period='30 days', horizon='112 days')
metrics = performance_metrics(df_cv)
print(f"MAPE: {metrics['mape'].mean():.4f}")
```

[Read the detailed blog post on methodology and findings](/time-series/public-health/economics/2023/08/15/forecasting-dengue-cases-and-cost-benefit-analysis.html)

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
