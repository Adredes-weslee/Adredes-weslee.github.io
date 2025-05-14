---
layout: project
title: "HDB Resale Price Insights: A Data-Driven Analysis for Singapore's Housing Market"
categories: [machine-learning, regression, real-estate, public-policy, urban-planning]
image: /assets/images/hdb-resale-prices.jpg # Or a new, more strategic image
technologies: [Python, Scikit-Learn, Pandas, Linear Regression, Ridge Regression, Lasso Regression, Feature Engineering, Data Visualization]
github: https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price
blog_post: /data-science/machine-learning/real-estate/2023/06/18/predicting-hdb-resale-prices.html # Link to the new blog post
---

## Project Overview

This project developed a robust machine learning model to predict Housing & Development Board (HDB) flat resale prices in Singapore. Utilizing a comprehensive dataset of over 60,000 transactions, the model combines advanced feature engineering specific to Singapore's unique public housing context with regularized regression techniques. The primary goal was to achieve high predictive accuracy (R² of 0.9261 and RMSE of ~SGD 39,180) and extract actionable insights into the key drivers of HDB resale values, thereby offering valuable information for homebuyers, policymakers, and urban planners.

## The Significance: Understanding Singapore's Housing Backbone

In Singapore, over 80% of the resident population lives in HDB flats, making public housing a cornerstone of the nation's social fabric and economy. The HDB resale market is dynamic and deeply impacts citizens' financial well-being, investment choices, and housing affordability. Accurate price prediction and a clear understanding of value drivers are crucial for:

-   **Informed Homebuyers:** Empowering individuals to make sound financial decisions when buying or selling flats.
-   **Effective Policymaking:** Providing data-driven insights for housing policies, subsidies, and urban development.
-   **Market Transparency:** Increasing clarity and reducing information asymmetry in the resale market.
-   **Urban Planning:** Guiding the development of amenities and infrastructure to support sustainable and equitable housing.

This project sought to answer critical questions such as: What are the most significant factors influencing HDB resale prices? How does the diminishing lease affect property valuation? What is the premium associated with location and amenities?

## Methodology: A Data-Driven Approach to Price Prediction

The project followed a structured data science methodology:

1.  **Data Collection & Integration:**
    * Utilized a Kaggle dataset containing 60,000+ HDB resale transactions.
    * Initial data included flat type, location (town, block), floor area, storey range, lease commencement date, and transaction date/price.

2.  **Domain-Specific Feature Engineering:**
    * Created over 30 new features tailored to the Singapore HDB context. Key engineered features included:
        * **Remaining Lease:** Calculated in years, with binary indicators for critical decay thresholds (e.g., <60 years, <40 years).
        * **Percentage of Lease Remaining.**
        * **Proximity to Amenities:** (Conceptual) Distance to nearest MRT stations, schools, shopping centers.
        * **Flat Age at Transaction.**
        * Encoded categorical variables like `flat_type`, `flat_model`, and `town`.

3.  **Rigorous Data Preprocessing:**
    * **Advanced Imputation:** Employed `IterativeImputer` to intelligently handle missing values.
    * **Categorical Encoding:** Used techniques like one-hot encoding or polynomial encoding.
    * **Feature Scaling:** Normalized numerical features using `StandardScaler`.
    * **Multicollinearity Management:** Conducted Variance Inflation Factor (VIF) analysis to identify and address highly correlated predictors.
    * **Feature Selection:** Applied Mutual Information analysis and LASSO regularization to select the most impactful features and reduce dimensionality.

4.  **Model Development & Validation:**
    * **Baseline Models:** Started with Multiple Linear Regression.
    * **Regularized Models:** Implemented Ridge (L2) and Lasso (L1) regression to prevent overfitting and improve generalization.
    * **Hyperparameter Optimization:** Used `GridSearchCV` with k-fold cross-validation to find optimal regularization parameters (alpha).
    * **Performance Evaluation:** Assessed models using Root Mean Squared Error (RMSE) and R-squared (R²). The final selected model (Ridge Regression) achieved an **RMSE of approximately SGD 39,180** and an **R² of 0.9261**.

## Key Findings & Insights into HDB Price Drivers

The analysis and predictive model uncovered several critical factors influencing HDB resale prices:

1.  **The Lease Decay Effect:**
    * A non-linear relationship was observed: prices decline more rapidly as the remaining lease falls, particularly below thresholds like 60 and 40 years. This quantitative insight is vital for understanding long-term value.

2.  **Flat Model & Type Premiums:**
    * Specific flat models, such as Design, Build and Sell Scheme (DBSS) flats and "Model A" types, consistently fetched higher prices, even after accounting for other factors.
    * Larger flat types (e.g., 5 ROOM, EXECUTIVE) naturally commanded higher prices, with floor area being a dominant price determinant.

3.  **The Power of Location ("Location, Location, Location"):**
    * Flats in central regions and mature estates showed significant price premiums.
    * Proximity to key amenities like MRT stations and reputable schools measurably increased resale values.

4.  **Storey Level Impact:**
    * Higher floor levels consistently correlated with higher prices, likely due to better views, ventilation, and perceived prestige. The premium per floor, however, could vary by estate age and building height.

5.  **Impact of Policy & Market Sentiment (Inferred):**
    * While not directly modeled as event studies, price patterns over time can reflect the impact of government housing policies (e.g., cooling measures, new grant schemes) and broader market sentiment.

## Technical Implementation Highlights

The project relied on a Python-based data science stack:

* **Data Manipulation:** Pandas for data loading, cleaning, and feature engineering.
* **Numerical Computation:** NumPy for efficient array operations.
* **Machine Learning:** Scikit-learn for:
    * Preprocessing (`StandardScaler`, `IterativeImputer`).
    * Modeling (`LinearRegression`, `Ridge`, `Lasso`).
    * Model selection and evaluation (`train_test_split`, `GridSearchCV`, `mean_squared_error`, `r2_score`).

A key feature engineering function:
```python
# Simplified example of lease feature creation
def create_lease_features(df):
    df_copy = df.copy()
    df_copy['remaining_lease_years'] = df_copy['lease_commence_year'] + 99 - df_copy['transaction_year']
    df_copy['lease_lt_60_years'] = (df_copy['remaining_lease_years'] < 60).astype(int)
    df_copy['lease_remaining_percentage'] = df_copy['remaining_lease_years'] / 99.0
    return df_copy
```
The final model, a Ridge Regression, was selected after hyperparameter tuning using `GridSearchCV` to optimize the `alpha` parameter, balancing model complexity and fit.

## Policy & Societal Implications

The insights derived from this HDB resale price prediction model have several important implications:

1.  **Enhanced Market Transparency:** Provides homebuyers and sellers with a data-driven tool for assessing fair market value.
2.  **Informed Policymaking on Lease Decay:** Offers quantitative evidence on the financial impact of diminishing leases, informing discussions on policies like VERS (Voluntary Early Redevelopment Scheme) and lease buyback options.
3.  **Targeted Housing Subsidies:** Allows for more nuanced design of housing grants and subsidies, potentially differentiating based on lease status or location-specific price pressures.
4.  **Data-Driven Urban Planning:** Insights into amenity valuation can guide future development plans, ensuring equitable distribution of facilities and infrastructure.
5.  **Financial Planning for Citizens:** Helps Singaporeans better understand the long-term value trajectory of their primary asset, aiding in retirement and financial planning.
6.  **Market Stability Monitoring:** Predictive models can serve as an early warning system for potential market anomalies or unsustainable price trends.

## Future Work & Potential Enhancements

This project lays a strong foundation for further research and development:
-   **Incorporate More Granular Data:** Add features like specific floor number (not just range), unit facing, renovation status, and detailed amenity proximity using GIS data.
-   **Develop Time-Aware Models:** Implement time series models (e.g., ARIMA, Prophet with regressors) to explicitly forecast future price trends.
-   **Build Neighborhood-Specific Models:** Create more localized models to capture unique micro-market dynamics within different HDB towns or estates.
-   **Create an Interactive Web Application:** Develop a user-friendly tool for prospective buyers and sellers to get instant price estimations and understand value drivers for specific flats.
-   **Integrate Macroeconomic Factors:** Include variables like interest rates, GDP growth, and inflation to improve long-range prediction accuracy.

## Conclusion

Predicting HDB resale prices in Singapore is a complex task due to the market's unique characteristics and the interplay of numerous factors. This project successfully demonstrated that by combining domain-specific feature engineering with robust machine learning techniques, it's possible to build highly accurate predictive models. The resulting R² of 0.9261 and RMSE of ~SGD 39,180 provide a strong quantitative tool. More importantly, the insights into price drivers—particularly lease decay, location, and flat attributes—offer significant value for individual decision-making, public policy formulation, and the pursuit of a more transparent and efficient housing market in Singapore.

---

*For a detailed technical walkthrough of the feature engineering and modeling process, please refer to the [accompanying blog post](/data-science/machine-learning/real-estate/2023/06/18/predicting-hdb-resale-prices.html). The full codebase is available on [GitHub](https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price).*
