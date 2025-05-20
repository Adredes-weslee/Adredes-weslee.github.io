---
layout: project
title: "Wet-Bulb Temperature & Climate Resilience: A Policy-Focused Data Study for Singapore"
categories: [climate-science, public-health, data-analysis, policy]
image: /assets/images/wet-bulb-temperature.jpg # Or a new, more policy-focused image
technologies: [Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, Time Series Analysis, Regression Modeling]
github: https://github.com/Adredes-weslee/wet-bulb-temperature-analysis
blog_post: /data-science/climate/public-health/2023/05/15/predicting-heat-stress-with-wet-bulb-temperature.html # Link to the new blog post
streamlit_app: https://adredes-weslee-data-analysis-of-wet-bulb-te-dashboardapp-mwqkey.streamlit.app/
---

## Project Overview

This project, framed as a hypothetical policy study for the Singapore government, investigates the critical relationship between **wet-bulb temperature (WBT)**—a key indicator of heat stress on the human body—and various climate change drivers. By analyzing long-term meteorological data and greenhouse gas concentrations, this study aims to identify key factors contributing to extreme heat conditions in tropical urban environments like Singapore and provide data-driven insights for public health planning and climate resilience strategies.

* **Comprehensive Data Integration**: Combined climate data from multiple sources including temperature, humidity, rainfall, and greenhouse gas concentrations.
* **Advanced Time Series Analysis**: Decomposed trends, seasonality, and anomalies in wet-bulb temperature patterns.
* **Statistical Modeling**: Developed regression models to identify key drivers of wet-bulb temperature changes.
* **Policy Recommendations**: Formulated evidence-based recommendations for climate resilience strategies.
* **Interactive Dashboard**: Created a Streamlit application for visualizing WBT trends, drivers, and projections.

<div class="demo-link-container">
  <a href="https://adredes-weslee-data-analysis-of-wet-bulb-te-dashboardapp-mwqkey.streamlit.app/" class="demo-button" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-play-circle"></i> Try the Live Demo
  </a>
</div>

## Background: The Significance of Wet-Bulb Temperature

Wet-bulb temperature (WBT) is a more comprehensive measure of heat stress than standard air temperature because it accounts for humidity. High WBT conditions severely limit the human body's ability to cool itself through sweating. This has profound implications:

* **At WBT > 31°C:** Sustained physical labor becomes dangerous.
* **At WBT > 35°C:** Represents the theoretical limit of human survivability for extended exposure, as effective thermoregulation via sweating ceases.
* **Vulnerable Populations:** Even at lower WBT values (e.g., 28°C), elderly individuals, children, and those with pre-existing health conditions face significant health risks.

As noted by Singapore's Minister for Sustainability and the Environment, Grace Fu, the nation could face days with peak temperatures reaching 40°C as early as 2045. This makes the assessment and prediction of WBT crucial for safeguarding public health and ensuring urban livability.

## Data Sources & Integration

A robust analysis requires integrating data from multiple authoritative sources. This study utilized:

| Data Source          | Variables                                       | Period    |
| -------------------- | ----------------------------------------------- | --------- |
| Data.gov.sg          | Wet-bulb temperature (hourly)                   | 1982-2023 |
| Data.gov.sg          | Surface air temperature (monthly)               | 1982-2023 |
| SingStat             | Rainfall, sunshine hours, relative humidity (monthly) | 1975-2023 |
| NOAA (ESRL/GML)      | CO₂ concentration (ppm, monthly global average)   | 1958-2023 |
| NOAA (ESRL/GML)      | CH₄ concentration (ppb, monthly global average)   | 1983-2023 |
| NOAA (ESRL/GML)      | N₂O concentration (ppb, monthly global average)   | 2001-2023 |
| NOAA (ESRL/GML)      | SF₆ concentration (ppt, monthly global average)   | 1997-2023 |

*Note: Global GHG averages were used as proxies for atmospheric concentrations influencing regional climate.*

## Methodology: A Phased Analytical Approach

The project followed a structured methodology:

1.  **Data Preprocessing & Integration:**
    * Aggregated hourly WBT readings to monthly averages.
    * Aligned all datasets to a common monthly timestamp.
    * Performed quality control, handled missing values (e.g., through imputation or forward/backward fill where appropriate), and standardized units.

2.  **Exploratory Data Analysis (EDA):**
    * Conducted correlation analysis between monthly average WBT and all climate variables (temperature, humidity, rainfall, sunshine) and GHG concentrations.
    * Performed time series decomposition (trend, seasonality, residuals) for WBT and key drivers.
    * Utilized visualizations (time series plots, scatter plots, correlation heatmaps) to understand relationships and trends.
    * Examined multicollinearity among predictor variables, particularly the greenhouse gases.

3.  **Predictive Model Development:**
    * Built multiple linear regression models to predict monthly average WBT based on the identified meteorological factors and GHG concentrations.
    * Assessed feature importance using model coefficients and statistical significance (p-values).
    * Validated models using standard techniques like train-test splits and evaluating performance metrics (R², RMSE).
    * Analyzed model residuals to check for assumptions of linear regression.

## Key Findings & Insights

The analysis yielded several important findings regarding WBT drivers in Singapore:

1.  **Dominance of Air Temperature:** Mean surface air temperature exhibited the strongest positive correlation with WBT, confirming its primary role in heat stress.
2.  **Humidity's Complex Role:** Contrary to common intuition that higher humidity always means higher WBT, relative humidity showed a *negative* correlation with WBT in Singapore's specific context when analyzed alongside other factors in the regression. This suggests that periods of extremely high humidity might also be associated with increased cloud cover or rainfall, which can lower ambient temperature, a more dominant WBT driver.
3.  **Significant Greenhouse Gas Impact:**
    * Nitrous oxide (N₂O) and sulfur hexafluoride (SF₆) concentrations showed statistically significant positive correlations with WBT in several model specifications.
    * Carbon dioxide (CO₂) and Methane (CH₄) also showed positive correlations, though their individual impact was sometimes masked by multicollinearity.
4.  **Multicollinearity Among GHGs:** Strong inter-correlations were observed between the various greenhouse gases, reflecting their common anthropogenic sources and long atmospheric lifetimes. This makes isolating the individual impact of each GHG complex but underscores their collective influence.
5.  **Trends in Extreme Values:** While a clear, consistent year-over-year upward trend in *average* monthly WBT was not strongly evident over the entire period after accounting for seasonality and other factors, the analysis suggested a potential increase in the frequency or magnitude of *extreme* WBT values, warranting further investigation with more granular (daily/hourly extreme) data.

## Policy Implications & Recommendations

The findings from this study support several actionable policy recommendations for Singapore:

1.  **Enhance Heat Advisory Systems:** Integrate WBT as a primary metric into public heat warning systems, alongside or in place of solely relying on dry-bulb temperature, to provide a more accurate measure of heat stress.
2.  **Strengthen Public Health Campaigns:** Develop targeted public education initiatives to raise awareness about WBT, its health risks (especially for vulnerable groups like the elderly, children, and outdoor workers), and preventive measures.
3.  **Climate-Resilient Urban Planning:** Incorporate WBT considerations into urban design and infrastructure development. Promote strategies like increasing green cover, cool pavements, and building designs that mitigate urban heat island effects and improve ventilation.
4.  **Review Labor Regulations for Outdoor Work:** Establish or update occupational safety and health guidelines for outdoor workers based on WBT thresholds to prevent heat-related illnesses and ensure worker safety.
5.  **Reinforce Climate Change Mitigation Efforts:** Continue and strengthen national and international efforts to reduce greenhouse gas emissions, as the study links rising GHG concentrations to increased WBT. This includes transitioning to renewable energy, improving energy efficiency, and sustainable practices.
6.  **Invest in Further Research:** Support ongoing research into localized WBT impacts, including detailed heat mapping, vulnerability assessments for different communities, and the effectiveness of various cooling strategies.

## Technical Implementation Snapshot

While the focus of this page is policy, the analysis was underpinned by data science techniques. For instance, Stull's formula (2011) can be used to calculate WBT from temperature and humidity:

```python
# Sample code for wet-bulb temperature calculation using Stull's formula
import numpy as np

def calculate_wetbulb_stull(temperature, relative_humidity):
    """
    Calculate wet-bulb temperature using Stull's formula (2011).
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
Regression models were then built using Python libraries such as Pandas for data manipulation and Scikit-Learn for model training and evaluation.

## Future Work & Research Directions

This study lays the groundwork for more extensive research:
* **Granular Data Analysis:** Incorporate daily or hourly extreme WBT values rather than monthly averages for a more precise risk assessment.
* **Additional Climate Variables:** Include data on wind speed, solar radiation, and urban heat island effect proxies (e.g., land surface temperature, NDVI).
* **Advanced Modeling:** Implement more sophisticated machine learning models (e.g., Random Forest, XGBoost, Neural Networks) and time series forecasting techniques (e.g., SARIMA, Prophet) to improve predictive accuracy and capture non-linearities.
* **Health Impact Correlation:** Integrate anonymized public health data (e.g., hospital admissions for heatstroke, cardiovascular events) to directly correlate WBT levels with health outcomes.
* **High-Resolution Heat Risk Mapping:** Develop detailed WBT risk maps for Singapore at a neighborhood level to identify hotspots and guide targeted interventions.

## Conclusion

Understanding and predicting wet-bulb temperature is paramount for building climate resilience, particularly in tropical urban environments like Singapore. This data-driven analysis has identified key meteorological and greenhouse gas drivers influencing WBT, providing a foundation for evidence-based policymaking. By proactively addressing the risks associated with rising WBT, Singapore can better protect public health, enhance urban livability, and adapt to the challenges of a warming climate.

---

*For a detailed technical walkthrough of the data processing, modeling, and analysis, please refer to the [accompanying blog post](/data-science/climate/public-health/2023/05/15/predicting-heat-stress-with-wet-bulb-temperature.html). The full codebase and data sources are available on [GitHub](https://github.com/Adredes-weslee/wet-bulb-temperature-analysis).*
