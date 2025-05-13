---
layout: project
title: "Predicting Wet-Bulb Temperature for Heat Stress Analysis"
categories: climate-science machine-learning data-analysis
image: /assets/images/wet-bulb-temperature.jpg
technologies: [Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, Time Series Analysis]
github: https://github.com/Adredes-weslee/wet-bulb-temperature-analysis
blog_post: /data-science/climate/public-health/2023/05/15/predicting-heat-stress-with-wet-bulb-temperature.html
---

## Project Overview

Commissioned as a hypothetical policy study for the Singapore government, this project investigates the relationship between wet-bulb temperature (WBT)—a crucial indicator of heat stress—and climate change drivers such as greenhouse gases and meteorological factors. Using regression modeling and time series analysis, I aimed to identify key contributors to extreme heat conditions in tropical environments and provide actionable insights for public health planning.

> Read my detailed blog post: [Predicting Heat Stress with Wet-Bulb Temperature Analysis](/data-science/climate/public-health/2023/05/15/predicting-heat-stress-with-wet-bulb-temperature.html)

## Background & Significance

Wet-bulb temperature (WBT) combines heat and humidity into a single value measuring heat stress on the human body. At wet-bulb temperatures above 35°C, even healthy humans cannot survive extended exposure as the body cannot cool itself through sweating.

With climate change accelerating, identifying regions approaching critical wet-bulb thresholds has significant public health implications:

- At WBT > 31°C: Most physical labor becomes dangerous
- At WBT > 35°C: Even resting humans cannot thermoregulate effectively
- Even at lower WBT values (28°C), vulnerable populations face significant health risks

As Minister for Sustainability and Environment Grace Fu noted, Singapore could experience days with peak temperatures of 40°C as early as 2045, making WBT assessment crucial for future planning.

## Data Integration

This study integrated data from multiple authoritative sources:

| Data Source | Variables | Period |
|-------------|-----------|--------|
| Data.gov.sg | Wet-bulb temperature (hourly) | 1982-2023 |
| Data.gov.sg | Surface air temperature (monthly) | 1982-2023 |
| SingStat | Rainfall, sunshine hours, humidity | 1975-2023 |
| NOAA | CO₂ concentration (ppm) | 1958-2023 |
| NOAA | CH₄ concentration (ppb) | 1983-2023 |
| NOAA | N₂O concentration (ppb) | 2001-2023 |
| NOAA | SF₆ concentration (ppt) | 1997-2023 |

## Methodology

### Exploratory Data Analysis
- Performed correlation analysis between WBT and all climate variables
- Conducted time series decomposition to identify seasonal patterns
- Created visualizations to understand variable relationships and trends
- Examined multicollinearity among greenhouse gases

### Data Processing
- Aggregated hourly WBT readings to monthly averages
- Aligned timestamps across datasets for proper integration
- Applied quality control filters to handle anomalous readings
- Standardized units across different measurement systems

### Model Development
- Built multiple linear regression models to predict WBT
- Assessed feature importance to understand key drivers
- Validated models using cross-validation techniques
- Evaluated performance using R², RMSE, and residual analysis

## Key Findings

1. **Critical Variables**: Mean air temperature showed the strongest positive correlation with WBT, followed by several greenhouse gases
2. **Negative Correlation**: Relative humidity surprisingly showed negative correlation with WBT in Singapore's context
3. **Greenhouse Gas Impact**: Nitrous oxide (N₂O) and sulfur hexafluoride (SF₆) displayed significant positive correlations with WBT
4. **Multicollinearity**: Strong correlations between greenhouse gases reflect their shared anthropogenic sources
5. **Extreme Values**: While no clear year-over-year WBT trend was observed, there's evidence of increasing extreme values

## Policy Implications

The findings suggest several actionable policy recommendations:

1. **Heat Monitoring**: Integrate WBT into Singapore's heat advisory systems rather than relying solely on conventional temperature metrics
2. **Public Education**: Develop education campaigns on heat stress risks, particularly for vulnerable populations
3. **Urban Planning**: Design future infrastructure with heat resilience in mind
4. **Labor Regulations**: Consider WBT thresholds for outdoor work safety guidelines
5. **Climate Action**: Continue efforts to reduce greenhouse gas emissions as part of heat stress mitigation

## Technical Implementation

```python
# Sample code for wet-bulb temperature calculation using Stull's formula
def calculate_wetbulb(temperature, relative_humidity):
    """
    Calculate wet-bulb temperature using Stull's formula (2011)
    temperature: dry-bulb temperature (°C)
    relative_humidity: relative humidity (%)
    """
    tw = temperature * np.arctan(0.151977 * np.power(relative_humidity + 8.313659, 0.5)) + \
         np.arctan(temperature + relative_humidity) - \
         np.arctan(relative_humidity - 1.676331) + \
         0.00391838 * np.power(relative_humidity, 1.5) * np.arctan(0.023101 * relative_humidity) - \
         4.686035
    return tw
```

## Future Work

- Expand the analysis with additional data sources including wind patterns and urbanization metrics
- Implement more sophisticated models including non-linear approaches (Random Forest, XGBoost)
- Develop real-time WBT monitoring tools for public health applications
- Integrate public health data to correlate WBT with health outcomes
- Create detailed heat risk maps for urban planning

## References

This project drew on extensive research from scientific literature, including studies from:
- NASA's Jet Propulsion Laboratory
- National Institute of Health
- Journal of Applied Physiology
- Singapore's National Climate Change Study

For a comprehensive bibliography, please visit the project repository.
