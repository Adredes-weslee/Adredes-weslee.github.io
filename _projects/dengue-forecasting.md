---
layout: project
title: "Strategic Dengue Control: Forecasting & Cost-Benefit Analysis for Public Health Interventions in Singapore"
categories: [time-series, public-health, machine-learning, health-economics, policy]
image: /assets/images/dengue-forecasting.jpg # Or a new, more strategic image
technologies: [Python, Time Series Analysis, Prophet, ARIMA, SARIMA, SARIMAX, BATS, TBATS, Health Economics, Cost-Benefit Analysis, Pandas, Statsmodels, Scikit-learn]
github: https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis
blog_post: /time-series/public-health/economics/2023/08/15/forecasting-dengue-cases-and-cost-benefit-analysis.html # Link to the new blog post
---

## Project Overview

This project presents a comprehensive approach to support evidence-based dengue intervention planning in Singapore by integrating **advanced time series forecasting** of dengue case loads with a **detailed health economics analysis** of control strategies. The aim was to provide actionable public health recommendations, optimize resource allocation, and enhance Singapore's resilience against dengue fever, a significant and recurring public health challenge.

## The Public Health Imperative: Dengue in Singapore

Dengue fever poses a substantial and growing public health burden in tropical urban environments like Singapore. Key facts underscoring the urgency include:
-   **High Incidence:** Singapore experienced its worst dengue outbreak in 2020, with over 30,000 reported cases.
-   **Significant Economic Costs:** Dengue control and treatment cost Singapore approximately SGD 100 million annually.
-   **Health Burden:** The disease contributes to morbidity, mortality (though rare in Singapore, still a risk), and places a strain on healthcare resources.
-   **Climate Change Impact:** Rising global temperatures and changing weather patterns are expected to exacerbate dengue transmission dynamics.

Effective public health planning requires accurate forecasting of future outbreaks and rigorous evaluation of intervention strategies to ensure optimal use of resources.

## Strategic Approach: Combining Prediction with Economic Evaluation

Our methodology was twofold:

1.  **Dengue Case Forecasting:** Develop and validate robust time series models to predict weekly dengue cases up to 16 weeks in advance. This involved:
    * Integrating diverse datasets (epidemiological, meteorological, demographic, Google Trends).
    * Performing comprehensive time series analysis to understand data characteristics (stationarity, seasonality, autocorrelation).
    * Developing and comparing multiple forecasting models (ARIMA, SARIMA, SARIMAX, Holt-Winters, BATS, TBATS, Prophet).
    * Selecting the best model based on performance metrics (MAPE, RMSE, MAE) and its ability to handle complex data features.

2.  **Health Economic Cost-Benefit Analysis:** Evaluate the economic viability and public health impact of two major dengue control interventions:
    * The **Wolbachia mosquito release program**.
    * A hypothetical **Dengvaxia® vaccination campaign**.
    * This involved calculating implementation costs, Disability-Adjusted Life Years (DALYs) averted, cost per DALY averted, and Benefit-Cost Ratios (BCR), benchmarked against WHO cost-effectiveness thresholds.

## Key Findings: Forecasting Dengue Outbreaks

### Data Insights & Model Performance
-   **Data Integration:** Successfully created a unified weekly time series from 2012-2022 by harmonizing epidemiological, weather, population, and Google Trends data.
-   **Time Series Properties:** Statistical tests (ADF p=0.0037, KPSS p=0.1) indicated difference stationarity. Strong weekly autocorrelation (0.9768) was observed.
-   **Model Champion:** Facebook's **Prophet model** emerged as the most accurate and robust, achieving a Mean Absolute Percentage Error (MAPE) of **0.0952** on the test set. It excelled at handling multiple seasonalities and incorporating exogenous weather variables.
-   **Predictive Horizon:** The selected model reliably forecasts dengue cases up to **16 weeks in advance**, providing a crucial window for proactive public health action.
-   **Key Predictors:** Feature importance analysis (from models like SARIMAX and Prophet with regressors) highlighted temperature, rainfall, and relative humidity (with appropriate lags) as significant drivers of dengue incidence.

## Key Findings: Economic Evaluation of Interventions

The cost-benefit analysis provided clear insights into the economic viability of the two primary dengue control strategies:

### Intervention Cost-Effectiveness Comparison:

| Metric                      | Project Wolbachia (Singapore) | Dengvaxia® Vaccination (Singapore - Hypothetical) |
| --------------------------- | ----------------------------- | ------------------------------------------------- |
| Annual Implementation Cost  | USD 27 million                | USD 36.32 million                                 |
| Annual DALYs Averted        | 449.7                         | 100.6                                             |
| **Cost per DALY Averted** | **USD 60,039** | **USD 360,876** (approx. 6x higher)             |
| **Benefit-Cost Ratio (BCR)**| **2.90** | Not explicitly calculated but implied lower       |

*Note: Costs and DALYs are based on specific study assumptions and modeling for the Singapore context.*

### Comparison with WHO Thresholds:
-   **Highly Cost-Effective Threshold (e.g., <0.5–1x GDP per capita for Singapore):** Approx. USD 30,364 – USD 60,728.
-   **Cost-Effective Threshold (e.g., <3x GNI per capita for Singapore):** Approx. USD 166,255.

**Conclusion on Cost-Effectiveness:**
-   **Project Wolbachia**, with a cost per DALY averted of USD 60,039, is considered **cost-effective** for a high-income country like Singapore, falling near the upper limit of "highly cost-effective" and well within the "cost-effective" range. Its BCR of 2.90 indicates that for every dollar invested, USD 2.90 in societal benefits (averted healthcare costs, productivity loss, etc.) are realized.
-   **Dengvaxia® vaccination**, at USD 360,876 per DALY averted, significantly exceeds these WHO cost-effectiveness thresholds for a population-level campaign in the Singaporean context, suggesting it is not an economically favorable primary intervention strategy for general use.

## Actionable Policy Implications & Recommendations

The combined findings from the forecasting models and the economic analysis led to the following strategic recommendations for Singapore's dengue control program:

1.  **Prioritize and Expand Project Wolbachia:** Given its superior cost-effectiveness and positive BCR, island-wide expansion of the Wolbachia mosquito release program is recommended as a primary dengue control measure.
2.  **Leverage Forecasts for Targeted Action:** Utilize the 16-week dengue case forecasts to proactively allocate resources, intensify vector control measures, and launch public awareness campaigns in anticipation of seasonal peaks or predicted outbreaks.
3.  **Strategic Use of Vaccination:** Reserve the Dengvaxia® vaccine for targeted use in specific high-risk individuals or groups as defined by health authorities, rather than pursuing a costly population-level campaign, due to its lower cost-effectiveness and serostatus-dependent risk profile.
4.  **Integrate Predictive Tools into Public Health Workflow:** Embed the developed forecasting models and dashboards into routine public health surveillance and planning processes to enable dynamic and data-driven decision-making.
5.  **Continuous Monitoring and Adaptive Management:** Regularly monitor the effectiveness of implemented interventions and recalibrate forecasting models and economic evaluations as new data (epidemiological, cost, intervention efficacy) becomes available.

## Overall Impact of the Project

This project provides a robust, evidence-based framework for enhancing Singapore's dengue prevention and control efforts:
-   **Improved Preparedness:** Delivers a reliable forecasting tool for early warning of potential outbreaks.
-   **Optimized Intervention Strategy:** Offers clear, data-driven guidance on prioritizing cost-effective interventions like Project Wolbachia.
-   **Efficient Resource Allocation:** Enables better timing and scaling of public health responses based on predicted needs.
-   **Strong Economic Justification:** Supplies a rigorous cost-benefit analysis to support public health budget allocations and policy decisions.

## Technologies Utilized

The project leveraged a range of Python-based data science and statistical modeling tools:
-   **Core Data Handling:** Pandas, NumPy
-   **Time Series Modeling:** Statsmodels (for ARIMA, SARIMA, SARIMAX), Facebook Prophet, other libraries for BATS/TBATS and Holt-Winters.
-   **Model Evaluation:** Scikit-learn (for metrics like MAPE, RMSE, MAE).
-   **Visualization:** Matplotlib, Seaborn.
-   **Health Economics:** Custom calculations for DALYs, cost-effectiveness ratios, and BCR based on established methodologies.

## Conclusion

By synergizing advanced time series forecasting with comprehensive health economic analysis, this project offers a powerful, data-driven pathway to strengthen dengue control in Singapore. The findings strongly advocate for the strategic expansion of cost-effective measures like Project Wolbachia, guided by predictive insights, to mitigate the public health and economic burden of dengue fever. This integrated approach serves as a model for evidence-based public health planning in the face of complex infectious disease challenges.

---

*For a detailed technical walkthrough of the forecasting methodologies and cost-benefit calculations, please refer to the [accompanying blog post](/time-series/public-health/economics/2023/08/15/forecasting-dengue-cases-and-cost-benefit-analysis.html). The full codebase is available on [GitHub](https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis).*
