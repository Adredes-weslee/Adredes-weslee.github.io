---
layout: post
title: "Decoding Heat Stress: A Data Scientist's Guide to Wet-Bulb Temperature Analysis"
date: 2023-05-15 10:30:00 +0800 # Retaining original date
categories: [data-science, climate, public-health]
tags: [wet-bulb-temperature, climate-change, regression-analysis, python, pandas, scikit-learn, data-integration, time-series]
author: Wes Lee
feature_image: /assets/images/2023-05-15-predicting-heat-stress-with-wet-bulb-temperature.jpg # Or a new, more technical image
---

## The Hidden Danger of Heat Stress: A Data Perspective

When we discuss rising global temperatures, the common metric is the dry-bulb temperature from weather forecasts. However, this figure alone doesn't capture the full picture of heat stress on the human body, especially in humid environments. As data scientists, we can delve deeper. This is where wet-bulb temperature (WBT) becomes a critical measure, combining both heat and humidity to quantify how effectively our bodies can cool down through perspiration.

This post details the technical journey of a data science project focused on analyzing WBT in Singapore, from data integration challenges to modeling and deriving insights.

<div class="callout interactive-demo">
  <h4><i class="fas fa-rocket"></i> Try It Yourself!</h4>
  <p>Explore wet-bulb temperature trends in Singapore and climate factor correlations in this interactive dashboard:</p>
  <a href="https://adredes-weslee-data-analysis-of-wet-bulb-te-dashboardapp-mwqkey.streamlit.app/" class="callout-button" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-play-circle"></i> Launch Interactive Dashboard
  </a>
</div>

> For a higher-level overview of this project's policy implications and key findings for Singapore, please see the [*Wet-Bulb Temperature & Climate Resilience: A Policy-Focused Data Study for Singapore* Project Page](/projects/wet-bulb-temperature-analysis/).

## Understanding Wet-Bulb Temperature (WBT)

WBT is the lowest temperature air can reach through evaporative cooling. It's a direct indicator of heat stress:
-   WBT > 31°C: Makes most physical labor dangerous.
-   WBT > 35°C: Surpasses the limit of human survivability for extended periods, even for healthy individuals at rest in shade with ample water.

Given predictions that Singapore might see peak temperatures of 40°C by 2045, analyzing WBT trends is crucial. Stull's formula (2011) is one common way to approximate WBT:

```python
import numpy as np # Make sure to import numpy

def calculate_wetbulb_stull(temperature, relative_humidity):
    """
    Calculate wet-bulb temperature using Stull's formula (2011).
    Args:
      temperature (float or np.array): Dry-bulb temperature in Celsius.
      relative_humidity (float or np.array): Relative humidity in percent (e.g., 70 for 70%).
    Returns:
      float or np.array: Wet-bulb temperature in Celsius.
    """
    term1 = temperature * np.arctan(0.151977 * np.power(relative_humidity + 8.313659, 0.5))
    term2 = np.arctan(temperature + relative_humidity)
    term3 = np.arctan(relative_humidity - 1.676331)
    term4 = 0.00391838 * np.power(relative_humidity, 1.5) * np.arctan(0.023101 * relative_humidity)
    tw = term1 + term2 - term3 + term4 - 4.686035
    return tw
```

## The Data Science Workflow: Analyzing WBT

### 1. Data Sourcing and Integration Challenges

A key challenge was integrating diverse datasets:
* Hourly WBT (Data.gov.sg, 1982-2023)
* Monthly surface air temperature (Data.gov.sg, 1982-2023)
* Monthly climate variables like rainfall, sunshine, humidity (SingStat, 1975-2023)
* Global greenhouse gas (GHG) concentrations (CO₂, CH₄, N₂O, SF₆) from NOAA (various periods starting from 1958-2001).

**Key Preprocessing Steps:**
* **Aggregation:** Hourly WBT was aggregated to monthly averages to align with other climate variables.
* **Timestamp Alignment:** Ensured all datasets were on a common monthly time index.
* **Quality Control:** Applied filters and checks for anomalous readings or missing data.
* **Unit Standardization:** Converted all measurements to consistent units.

```python
import pandas as pd

# Conceptual function for preprocessing and merging
def preprocess_and_merge_data(wbt_df, temp_df, climate_df, ghg_data_dict):
    """
    Preprocesses and merges various climate and GHG datasets.
    Args:
        wbt_df (pd.DataFrame): DataFrame with hourly wet-bulb temperature. Must have a datetime index or 'date' column.
        temp_df (pd.DataFrame): DataFrame with monthly surface air temperature. Must have a 'month' (period/datetime) column.
        climate_df (pd.DataFrame): DataFrame with other monthly climate variables. Must have a 'month' column.
        ghg_data_dict (dict): Dictionary of DataFrames for GHG data, e.g., {'co2': co2_df, 'ch4': ch4_df}. Each df must have a 'month' column.
    Returns:
        pd.DataFrame: A merged and preprocessed DataFrame.
    """
    # Ensure 'date' or 'month' columns are datetime objects
    # Example for wbt_df, assuming 'date' column exists
    if 'date' in wbt_df.columns:
        wbt_df['date'] = pd.to_datetime(wbt_df['date'])
        wbt_df.set_index('date', inplace=True)
    
    # Aggregate hourly WBT to monthly means
    # Ensure 'wet_bulb_temperature' is the column name for WBT values
    wbt_monthly = wbt_df['wet_bulb_temperature'].resample('MS').mean().reset_index() # MS for Month Start
    wbt_monthly.rename(columns={'date': 'month', 'wet_bulb_temperature': 'mean_wet_bulb_temperature'}, inplace=True)

    # Standardize 'month' column for merging (if not already datetime)
    # Example for temp_df
    if 'month' in temp_df.columns and not pd.api.types.is_datetime64_any_dtype(temp_df['month']):
        temp_df['month'] = pd.to_datetime(temp_df['month'])
    if 'month' in climate_df.columns and not pd.api.types.is_datetime64_any_dtype(climate_df['month']):
        climate_df['month'] = pd.to_datetime(climate_df['month'])

    # Merge WBT with surface air temperature
    combined_df = pd.merge(wbt_monthly, temp_df, on='month', how='inner')
    
    # Merge with other climate variables
    combined_df = pd.merge(combined_df, climate_df, on='month', how='inner')
    
    # Merge with greenhouse gas data
    for gas_name, gas_df in ghg_data_dict.items():
        if 'month' in gas_df.columns and not pd.api.types.is_datetime64_any_dtype(gas_df['month']):
            gas_df['month'] = pd.to_datetime(gas_df['month'])
        combined_df = pd.merge(combined_df, gas_df, on='month', how='left') # Use left merge to keep all climate data
        
    # Handle missing values (e.g., forward fill, interpolate, or drop)
    # This is a critical step and strategy depends on the data
    combined_df.ffill(inplace=True) # Example: Forward fill
    combined_df.bfill(inplace=True) # Example: Backward fill for any remaining at the start
    combined_df.dropna(inplace=True) # Drop rows if critical data is still missing

    return combined_df

# Example Usage (assuming DataFrames are loaded):
# ghg_dfs = {'co2': co2_df, 'ch4': ch4_df, 'n2o': n2o_df, 'sf6': sf6_df}
# final_climate_data = preprocess_and_merge_data(raw_wbt_df, surface_temp_df, other_climate_df, ghg_dfs)
# print(final_climate_data.head())
```

### 2. Exploratory Data Analysis (EDA)

EDA was performed to understand trends, seasonality, and correlations.
* **Correlation Analysis:** Mean air temperature showed the strongest positive correlation with WBT.
* **Counterintuitive Finding:** Relative humidity exhibited a negative correlation with WBT in Singapore's tropical context. This is likely due to complex interactions where periods of very high humidity might coincide with cloud cover and rain, which can lower air temperature, a dominant factor in WBT.
* **Greenhouse Gas Impact:** All measured GHGs (CO₂, CH₄, N₂O, SF₆) showed positive correlations with WBT, with N₂O and SF₆ being particularly significant in some models. However, strong multicollinearity was observed among GHGs, reflecting their shared anthropogenic origins.
* **Seasonality:** WBT showed clear seasonal patterns, aligning with Singapore's monsoon cycles, typically peaking during inter-monsoon periods. Time series decomposition helped isolate these patterns.

### 3. Modeling Wet-Bulb Temperature

Multiple linear regression was chosen as the primary modeling approach to identify key drivers of WBT.

**Feature Selection:**
Based on EDA and domain knowledge, features included: mean surface air temperature, mean relative humidity, total rainfall, daily mean sunshine hours, and average concentrations of CO₂, CH₄, N₂O, and SF₆.

**Model Training and Evaluation:**
```python
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np # Ensure numpy is imported

# Assume 'final_climate_data' is the merged and preprocessed DataFrame from the previous step
# And it contains 'mean_wet_bulb_temperature' as the target and other features

# Define features (X) and target (y)
# Ensure these column names exactly match your DataFrame
feature_columns = [
    'mean_surface_air_temp', # Example column name, adjust to your actual data
    'mean_relative_humidity',
    'total_rainfall',
    'mean_daily_sunshine_hours', # Example column name
    'average_co2_ppm', 
    'average_ch4_ppb',
    'average_n2o_ppb',
    'average_sf6_ppt'
]
# Verify all feature_columns exist in final_climate_data
existing_features = [col for col in feature_columns if col in final_climate_data.columns]
if len(existing_features) != len(feature_columns):
    print(f"Warning: Some feature columns are missing. Using: {existing_features}")

X = final_climate_data[existing_features]
y = final_climate_data['mean_wet_bulb_temperature']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=False) # Time series data, so shuffle=False is often preferred

# Initialize and train the Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = lr_model.predict(X_test)

# Evaluate the model
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"Model R-squared: {r2:.4f}")
print(f"Model RMSE: {rmse:.4f}")

# Feature importance (coefficients for Linear Regression)
coefficients = pd.DataFrame(lr_model.coef_, X.columns, columns=['Coefficient'])
print("\nFeature Coefficients:")
print(coefficients)
```
The model achieved a good R² score, indicating that the selected features could explain a significant portion of WBT variance. Cross-validation techniques were used for robust validation. Residual analysis helped check model assumptions.

## Technical Lessons Learned from the Project

1.  **Domain Knowledge is Key:** A solid understanding of climate science, thermodynamics (especially evaporative cooling), and the specific local meteorology of Singapore was invaluable for feature engineering, model interpretation, and explaining counterintuitive findings (like the negative RH-WBT correlation).
2.  **Data Integration is Non-Trivial:** Harmonizing datasets with varying temporal resolutions, measurement units, and data collection methodologies requires meticulous attention to detail and robust preprocessing pipelines.
3.  **Multicollinearity Management:** GHGs are highly correlated. While this doesn't necessarily degrade predictive power, it makes interpreting individual coefficients challenging. Techniques like Principal Component Analysis (PCA) or using domain knowledge to select representative GHGs could be explored. For this policy-focused study, showing the collective association was still insightful.
4.  **Correlation vs. Causation:** It's crucial to communicate that while the model identifies strong statistical relationships (e.g., between certain GHGs and WBT), it doesn't inherently prove direct causation for each variable in isolation, especially with complex Earth systems.
5.  **Model Simplicity for Policy:** For a policy-focused study, a well-understood model like linear regression can be more effective for communication than a complex black-box model, provided it performs adequately. The interpretability of coefficients (though needing care with multicollinearity) is a plus.

## Future Technical Directions

While this project provided valuable insights, further technical work could include:
* **Non-Linear Models:** Exploring machine learning models like Random Forest, Gradient Boosting (XGBoost), or Support Vector Regression to capture potential non-linear relationships between WBT and its drivers.
* **Time Series Models:** Employing more sophisticated time series models (e.g., SARIMA, Prophet) to better account for autocorrelation and seasonality in WBT predictions.
* **Feature Engineering:** Incorporating interaction terms, lagged variables (e.g., previous month's WBT), and other derived features like dew point.
* **Spatial Analysis:** Developing heat risk maps by integrating WBT predictions with geographical data and urban characteristics (e.g., green cover, building density).
* **Real-time Monitoring Tools:** Building a dashboard or API for real-time WBT calculation and short-term forecasting using current meteorological data.

## Conclusion

Analyzing wet-bulb temperature through a data science lens offers a more nuanced understanding of heat stress than relying on dry-bulb temperature alone. This project demonstrated a practical workflow for integrating diverse climate data, building predictive models, and extracting technically sound insights that can inform public health and environmental policy. The journey highlighted the importance of careful data handling, robust EDA, and appropriate model selection in tackling complex real-world problems.

---

*This post details the technical execution of the Wet-Bulb Temperature Analysis project. To learn more about the project's background, findings, and policy implications for Singapore, please visit the [project page on GitHub](https://github.com/Adredes-weslee/wet-bulb-temperature-analysis).*
