---
layout: post
title: "Predicting Heat Stress with Wet-Bulb Temperature Analysis"
date: 2023-05-15 10:30:00 +0800
categories: [data-science, climate, public-health]
tags: [wet-bulb-temperature, climate-change, regression-analysis, singapore]
author: Wes Lee
feature_image: /assets/images/placeholder.svg
---

## The Hidden Danger of Heat Stress

When most people think about climate change and rising temperatures, they focus on simple thermometer readings—the dry-bulb temperature we see in weather forecasts. However, as a data scientist working on climate analytics, I've learned that this common metric misses a crucial factor that determines human survivability: humidity.

This is where wet-bulb temperature (WBT) comes in—a vital but often overlooked measurement that combines heat and humidity to indicate how effectively the human body can cool itself through sweating. In this post, I'll share my journey through a recent data science project analyzing WBT in Singapore and its implications for public health and policy.

## Why Wet-Bulb Temperature Matters

Unlike the standard temperature reading, wet-bulb temperature represents the lowest temperature to which air can be cooled by water evaporation. For humans, this is critical because:

- At WBT > 35°C, even healthy humans in shade with unlimited water cannot survive extended exposure
- At WBT > 31°C, most physical labor becomes dangerous
- Even at WBT of 28°C, vulnerable populations face significant health risks

With Singapore potentially facing days with peak temperatures of 40°C as early as 2045, understanding WBT trends isn't just academic—it's essential for future planning.

## The Data Science Approach

### Data Integration Challenges

One of the most challenging aspects of this project was integrating data from multiple sources with different formats, time periods, and measurement frequencies:

1. **Hourly WBT readings** from Data.gov.sg (1982-2023)
2. **Monthly climate variables** from SingStat (1975-2023)
3. **Global greenhouse gas measurements** from NOAA (various periods)

To create a unified dataset for analysis, I:

- Aggregated hourly readings to monthly averages
- Aligned timestamps across datasets
- Applied quality control measures to identify and handle anomalies
- Created a normalized dataset with consistent units and formats

```python
# Example of data preprocessing and integration
def preprocess_climate_data(wbt_df, temp_df, climate_df, ghg_dfs):
    # Aggregate hourly WBT to monthly
    wbt_monthly = wbt_df.resample('MS', on='date').mean()
    
    # Align timestamps and merge datasets
    combined_df = pd.merge(wbt_monthly, temp_df, on='month', how='inner')
    combined_df = pd.merge(combined_df, climate_df, on='month', how='inner')
    
    # Add greenhouse gas data
    for gas_name, gas_df in ghg_dfs.items():
        combined_df = pd.merge(combined_df, gas_df, on='month', how='left')
    
    # Handle missing values
    combined_df = handle_missing_values(combined_df)
    
    return combined_df
```

### Exploratory Analysis Insights

The exploratory data analysis revealed several interesting patterns:

1. **Counterintuitive humidity relationship**: In Singapore's context, relative humidity showed a negative correlation with WBT—contrary to what many might expect. This highlights the complex interplay of meteorological factors in a tropical island environment.

2. **Greenhouse gas correlations**: All four measured greenhouse gases showed positive correlations with WBT, with nitrous oxide (N₂O) and sulfur hexafluoride (SF₆) displaying particularly significant relationships.

3. **Seasonal patterns**: Clear seasonal variations in WBT followed Singapore's monsoon cycles, with peaks during the inter-monsoon periods.

### Modeling Approach

For this project, I chose multiple linear regression as my primary modeling approach:

```python
# Feature selection for the model
features = ['mean_surface_airtemp', 'mean_relative_humidity', 
            'total_rainfall', 'daily_mean_sunshine',
            'average_co2_ppm', 'average_ch4_ppb', 
            'average_n2o_ppb', 'average_sf6_ppt']

# Train-test split
X = climate_data[features]
y = climate_data['mean_wet_bulb_temperature']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
```

The model achieved promising results with an R² score indicating that a significant portion of WBT variance could be explained by the selected features.

## From Analysis to Action

The technical insights from this project translate into several actionable recommendations:

1. **Integrate WBT into public health systems**: Singapore should include WBT in heat advisories and weather forecasts to better inform the public about heat stress risks.

2. **Targeted interventions**: Resources should be directed toward protecting vulnerable populations during high WBT periods.

3. **Urban planning considerations**: Future infrastructure development should incorporate heat resilience strategies, particularly in dense urban areas.

4. **Labor regulations**: Outdoor work safety guidelines should be updated to reflect WBT thresholds rather than simple temperature readings.

## Technical Lessons Learned

From a data science perspective, this project reinforced several important principles:

1. **Domain knowledge is crucial**: Understanding the physiology of heat stress and climate science was essential for meaningful analysis.

2. **Data integration requires care**: Merging datasets from different sources necessitates careful validation and harmonization.

3. **Correlation isn't causation**: While we found significant correlations between greenhouse gases and WBT, establishing causal relationships requires additional research.

4. **Communication is key**: Translating technical findings into actionable insights for policymakers requires clear communication of complex concepts.

## Next Steps

This project is just the beginning of what could be a more comprehensive analysis of climate change impacts on Singapore. Future work could include:

- Developing real-time WBT monitoring tools
- Creating detailed heat risk maps for urban planning
- Integrating public health data to correlate WBT with health outcomes
- Implementing more sophisticated models including non-linear approaches

## Conclusion

As climate change continues to alter our environment, data science tools offer powerful ways to understand and prepare for these changes. The wet-bulb temperature analysis demonstrates how combining multiple data sources with careful analysis can provide insights that go beyond simple temperature measurements—insights that could literally be life-saving as our world warms.

By bringing these advanced metrics into public awareness and policy considerations, we can better prepare for a future where extreme heat may become a regular challenge to human health and productivity.

---

*Want to learn more about this project or discuss other climate data applications? Connect with me on [LinkedIn](https://www.linkedin.com/in/wes-lee/) or check out the full project on [GitHub](https://github.com/Adredes-weslee/wet-bulb-temperature-analysis).*
