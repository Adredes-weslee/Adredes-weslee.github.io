---
layout: post
title: "Forecasting Dengue Cases and Conducting Cost-Benefit Analysis of Public Health Interventions"
date: 2023-08-15 10:00:00 +0800
categories: [time-series, public-health, economics]
tags: [forecasting, prophet, arima, health-economics, cost-benefit-analysis, singapore, dengue]
author: Wes Lee
feature_image: /assets/images/placeholder.svg
---

## Tackling a Growing Public Health Challenge

Dengue fever represents one of the most significant mosquito-borne viral diseases globally, with dramatic increases in incidence over the past 20 years. In Singapore, where I conducted this analysis, the disease poses a persistent public health challenge despite the country's advanced healthcare system and mosquito control efforts.

The record-breaking outbreak in 2020 with over 30,000 cases served as a stark reminder that we need better predictive tools and cost-effective intervention strategies. This project emerged from that need, combining advanced time series forecasting with health economic analysis to provide evidence-based recommendations for dengue management.

## A Multi-faceted Forecasting Approach

### Comprehensive Data Integration

Building effective forecasting models requires rich, multi-dimensional datasets. For this project, I integrated several data streams:

- **Epidemiological data**: Weekly dengue case counts spanning 10 years (2012-2022)
- **Meteorological variables**: Temperature, relative humidity, and rainfall patterns
- **Demographic information**: Population statistics and density metrics
- **Digital signals**: Google Trends data for dengue-related search terms

This integrated dataset provided a foundation for examining relationships between environmental factors and dengue incidence, while also capturing seasonal patterns and long-term trends.

### Time Series Characteristics and Challenges

Analyzing dengue case data presented several interesting challenges:

**1. Complex Seasonality**: Dengue in Singapore shows multiple seasonal patterns:
   - Annual cycles with peaks typically in June-October
   - Multi-year epidemic cycles with large outbreaks every 5-6 years
   - Weekly variations related to reporting patterns

**2. Environmental Drivers**: Correlation analysis revealed lag relationships between weather variables and case counts:
   - Temperature showed significant correlation with a 3-4 week lag
   - Rainfall exhibited complex relationships with 1-3 week lags
   - Relative humidity had more immediate associations with case increases

**3. Stationarity Considerations**: The series exhibited difference stationarity rather than trend stationarity, informing our modeling decisions:

```python
# Testing for stationarity
from statsmodels.tsa.stattools import adfuller, kpss

# Augmented Dickey-Fuller test (H0: non-stationary)
result = adfuller(df['cases'])
print(f'ADF Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
# Result: p-value 0.0037 - reject null hypothesis, indicating stationarity

# KPSS test (H0: stationary)
result = kpss(df['cases'])
print(f'KPSS Statistic: {result[0]:.4f}')
print(f'p-value: {result[1]:.4f}')
# Result: p-value 0.10 - fail to reject null hypothesis, also indicating stationarity
```

The contrasting results from ADF and KPSS tests suggested different characteristics at various time scales, requiring models that could handle both short-term fluctuations and longer-term patterns.

### Model Development and Selection

I implemented and evaluated seven distinct time series forecasting approaches:

1. **ARIMA (AutoRegressive Integrated Moving Average)**: Baseline performance with simple temporal dependencies
2. **SARIMA (Seasonal ARIMA)**: Extension with explicit seasonal components
3. **SARIMAX**: Incorporating exogenous variables (weather data) for improved predictions
4. **Holt-Winters Exponential Smoothing**: Triple exponential smoothing for level, trend, and seasonal components
5. **BATS (Box-Cox transform, ARMA errors, Trend, and Seasonal components)**: Handling complex seasonality
6. **TBATS**: Extension that accommodates multiple seasonal periods
7. **Prophet**: Facebook's decomposable time series model with trend, multiple seasonality components, and holiday effects

After rigorous cross-validation and performance comparison, Prophet emerged as the superior model for several reasons:

- **Lower MAPE** (Mean Absolute Percentage Error): 9.52% on the test set
- **Superior handling of multiple seasonalities** without overfitting
- **Effective incorporation of external regressors** like temperature and rainfall
- **Interpretable components** allowing for insights into trend and seasonal effects
- **Reliable uncertainty intervals** for forecasting

The model selection process highlighted the importance of balancing statistical performance with practical interpretability when building forecasting tools for public health applications.

## From Prediction to Action: Cost-Benefit Analysis

Accurate forecasting provides the foundation for evidence-based decision making, but translating predictions into policy requires economic evaluation. I conducted a comprehensive cost-benefit analysis of two major dengue control strategies:

### Evaluating Intervention Options

**1. Project Wolbachia**
   - A biocontrol strategy using *Wolbachia*-infected mosquitoes that reduce transmission
   - Successfully piloted in Singapore since 2016 with demonstrated effectiveness
   - Requires ongoing releases but creates sustained population reduction

**2. Dengvaxia® Vaccination Program**
   - First licensed dengue vaccine with demonstrated efficacy
   - Requires individual serological screening due to risks in seronegative individuals
   - Implementation complexity in mass vaccination scenarios

The analysis incorporated:
- Implementation costs (scaling estimates from pilot programs)
- Hospitalization costs averted
- Productivity losses prevented
- Disease burden reduction (calculated in Disability-Adjusted Life Years)

### Economic Findings and Implications

The results revealed substantial differences in cost-effectiveness between the two interventions:

```python
# Simplified cost-effectiveness calculation
def calculate_cost_effectiveness(intervention_cost, dalys_averted):
    cost_per_daly = intervention_cost / dalys_averted
    return cost_per_daly

# Project Wolbachia
wolbachia_cost_per_daly = calculate_cost_effectiveness(27000000, 449.7)  # USD 60,039

# Dengvaxia vaccination program
dengvaxia_cost_per_daly = calculate_cost_effectiveness(36320000, 100.6)  # USD 360,876
```

The key findings:

1. **Project Wolbachia** showed dramatically better cost-effectiveness:
   - Cost per DALY averted: USD 60,039
   - Benefit-Cost Ratio: 2.90 (every dollar spent returns $2.90 in benefits)

2. **Dengvaxia® vaccination** was substantially less cost-effective:
   - Cost per DALY averted: USD 360,876 (six times higher than Wolbachia)
   - Additional screening costs and administrative complexity
   
3. **WHO Cost-Effectiveness Thresholds**:
   - The conservative threshold (0.5× GDP per capita): USD 30,364
   - Standard threshold for developed nations (3× GNI): USD 166,255
   - Wolbachia falls between these thresholds, representing reasonable value for a high-income country

The analysis demonstrated that while Project Wolbachia exceeds the most conservative cost-effectiveness threshold, it falls well within acceptable ranges for high Human Development Index countries like Singapore, especially when considering additional benefits not captured in the primary analysis.

## Practical Applications and Future Directions

The combined forecasting and economic analysis yielded several actionable recommendations:

1. **Epidemic Preparedness**: Using the 16-week forecast horizon to calibrate resource allocation for upcoming outbreaks
2. **Targeted Interventions**: Concentrating Wolbachia releases in high-risk areas identified through spatio-temporal analysis
3. **Vaccination Policy**: Reserving vaccination for specific high-risk populations rather than universal implementation
4. **Integration with Public Health Systems**: Embedding forecasting tools in routine surveillance activities
5. **Ongoing Monitoring**: Continuous evaluation of intervention effectiveness with model recalibration

### Technical Lessons Learned

From a data science perspective, this project reinforced several key insights:

- **Multi-model approach**: The value of implementing multiple models rather than assuming one methodology fits all problems
- **Feature engineering importance**: How domain knowledge can improve model performance through targeted feature creation
- **Uncertainty quantification**: The critical nature of providing prediction intervals rather than point estimates for public health planning
- **Economics + Data Science**: The power of combining predictive analytics with economic evaluation for decision support

### Future Enhancements

As I continue refining this work, several extensions are planned:

1. **Spatio-temporal modeling**: Incorporating geographic variation to identify localized outbreak patterns
2. **Genomic surveillance integration**: Adding viral serotype data to improve prediction accuracy during serotype shifts
3. **Real-time updating**: Developing a pipeline for continuous model updating as new case data becomes available
4. **Ensemble methods**: Combining forecasts from multiple models to further improve accuracy
5. **Extended cost modeling**: Incorporating additional economic factors including tourism impacts and property value effects

## Conclusion

Dengue fever remains a formidable public health challenge in Singapore and across tropical regions worldwide. This project demonstrates how combining advanced time series forecasting with rigorous economic analysis can provide evidence-based guidance for intervention strategies.

By quantifying both the predictive power of various models and the economic implications of different control strategies, we create a powerful decision-support framework that balances statistical rigor with practical applicability. The resulting recommendations not only address immediate public health needs but also provide a sustainable roadmap for long-term dengue management.

As climate change potentially expands the geographic range of dengue and other mosquito-borne diseases, such integrated analytical approaches will become increasingly valuable for public health planners and policymakers worldwide.

---

*For a more detailed look at the methodology and implementation, check out the [complete project page](/projects/dengue-forecasting/) or view the [source code on GitHub](https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis).*
