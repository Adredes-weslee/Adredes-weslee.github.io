---
layout: post
title: "Predicting HDB Resale Prices in Singapore"
date: 2023-06-18 14:45:00 +0800
categories: [data-science, machine-learning, real-estate]
tags: [housing, singapore, regression, feature-engineering, price-prediction]
author: Wes Lee
feature_image: /assets/images/placeholder.svg
---

## The Unique Singapore Housing Market

Singapore's public housing market is one of the most fascinating real estate ecosystems in the world. With over 80% of residents living in Housing Development Board (HDB) flats, these properties represent far more than just shelter—they're crucial financial assets for most Singaporean families and a cornerstone of the nation's economic policy.

As a data scientist with an interest in real estate markets, I recently completed a project analyzing and predicting HDB resale prices using machine learning techniques. In this post, I'll share my approach, insights, and the practical implications of this work for homebuyers, policymakers, and urban planners.

## Why HDB Price Prediction Matters

Housing costs represent the single largest expenditure for most Singaporeans. With average prices for resale flats exceeding SGD 500,000, even small percentage improvements in price prediction accuracy can translate to significant financial impact for buyers. Beyond individual decisions, understanding price drivers also informs:

- **Policy decisions** about housing grants and subsidies
- **Urban planning** for amenity distribution and transportation networks
- **Investment strategies** for property owners
- **Financial planning** for prospective buyers

## The Data Science Approach

### Dataset and Features

For this project, I leveraged a Kaggle dataset containing 60,000+ HDB resale transactions. Key variables included:

- Physical attributes (floor area, flat type, storey range)
- Location information (town, region)
- Lease details (commencement year, remaining lease)
- Derived proximity features (distance to MRT stations, schools)

### Feature Engineering

Some of the most valuable predictors came not from the raw data but from engineered features:

```python
# Example of feature engineering for lease decay
def create_lease_features(df):
    # Calculate remaining lease in years
    df['remaining_lease'] = df['lease_commence_year'] + 99 - df['transaction_year']
    
    # Create lease decay indicators - research shows pricing cliffs at certain thresholds
    df['lease_decay_60'] = df['remaining_lease'] < 60
    df['lease_decay_40'] = df['remaining_lease'] < 40
    
    # Calculate percentage of lease remaining
    df['lease_remaining_pct'] = df['remaining_lease'] / 99
    
    return df
```

This approach to lease remaining was particularly important because of the non-linear relationship between lease decay and property values—a phenomenon unique to leasehold property systems like Singapore's.

### Modeling Process

I employed a progressive modeling strategy:

1. **Baseline model**: Simple linear regression with basic features
2. **Feature optimization**: Iterative feature selection using mutual information and domain knowledge
3. **Regularization**: Ridge and Lasso regression to prevent overfitting and handle multicollinearity
4. **Hyperparameter tuning**: Grid search with cross-validation to find optimal regularization strength

The final Ridge regression model achieved an R² score of 0.9261 and RMSE of approximately SGD 39,180—representing less than 8% error on the average transaction price.

## Key Insights from the Analysis

### 1. The Lease Decay Effect

One of the most interesting findings involved the relationship between remaining lease and resale value. Rather than a linear decline, the data revealed distinct "cliffs" in valuation:

- Properties with less than 60 years remaining showed accelerated depreciation
- Flats with less than 40 years saw even steeper price drops
- This non-linear pattern challenges simplistic assumptions about lease depreciation

This insight is particularly relevant given ongoing national discussions about lease decay and potential Voluntary Early Redevelopment Scheme (VERS) policies.

### 2. Location Premium Patterns

The analysis quantified what many Singaporeans intuitively know—central region properties command significant premiums. However, the data revealed more nuanced patterns:

- Mature estates maintain value better than expected given their older lease profiles
- Proximity to multiple MRT lines creates multiplicative rather than additive value
- Specific towns (like Marine Parade and Queenstown) show price resilience beyond what their amenities would predict

### 3. Floor Level Economics

Higher floors consistently commanded price premiums, but the relationship wasn't uniform:

- The premium for high floors (>10th story) was more pronounced in newer estates
- Buildings with more stories showed lower per-floor premium increments
- Corner units and units with unblocked views commanded additional premiums beyond the floor effect

### 4. The DBSS and Premium Flat Effect

Design, Build and Sell Scheme (DBSS) flats and premium models showed significant price premiums:

```python
# Feature importance visualization for flat models
coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_
})

# Sort by absolute coefficient value
coefficients = coefficients.reindex(
    coefficients['Coefficient'].abs().sort_values(ascending=False).index
)

# Plot top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(data=coefficients.head(10), x='Coefficient', y='Feature')
plt.title('Top 10 Features by Importance')
plt.tight_layout()
```

This analysis showed that DBSS flats and Premium Apartments commanded the highest model-specific premiums, even after controlling for size, location, and other factors.

## Practical Applications

### For Homebuyers

The model offers several practical tools for prospective buyers:

1. **Value assessment**: Determine if a listing is fairly priced based on comparable properties
2. **Feature prioritization**: Understand which features deliver the most value for your budget
3. **Negotiation insights**: Identify properties that may be overvalued due to specific features
4. **Long-term planning**: Project how future lease decay might affect resale potential

### For Policymakers

The findings also have implications for housing policy:

1. **Lease decay management**: Develop targeted interventions for flats approaching critical lease thresholds
2. **Amenity distribution**: Balance amenity development to reduce regional price disparities
3. **Grant optimization**: Structure housing grants to address specific market inefficiencies
4. **Market monitoring**: Detect potential pricing bubbles or anomalies

## Challenges and Limitations

Despite the model's strong performance, several challenges remain:

1. **Hidden variables**: Factors like unit orientation, view quality, and renovation state aren't captured in the public dataset
2. **Temporal dynamics**: Market conditions and buyer preferences evolve over time
3. **Policy intervention effects**: Changes in cooling measures or grant structures can create structural breaks in pricing patterns
4. **Hyperlocal factors**: Neighborhood-level developments that affect only small clusters of properties

## Technical Lessons Learned

From a data science perspective, this project reinforced several key lessons:

1. **Domain knowledge matters**: Understanding Singapore's unique housing policies was essential for effective feature engineering
2. **Multicollinearity management**: In real estate, many predictors are naturally correlated, requiring careful handling through regularization
3. **Feature interaction importance**: The interaction between variables (like floor × view or location × amenities) often contains more signal than individual features
4. **Interpretability vs. accuracy**: While more complex models like random forests achieved marginally better performance, the interpretability of linear models provided more valuable insights

## Next Steps

As I continue to refine this work, several extensions are planned:

1. **Time series modeling**: Incorporating temporal trends to forecast future price movements
2. **Hyperlocal analysis**: Developing neighborhood-specific models to capture micromarket dynamics
3. **Interactive application**: Creating a user-friendly tool for homebuyers to estimate fair values
4. **Macroeconomic integration**: Adding broader economic indicators to improve long-range forecasting

## Conclusion

Singapore's HDB market represents a fascinating intersection of social policy, urban planning, and real estate economics. Through data science, we can decode the complex factors driving property values, helping both individuals and policymakers make more informed decisions.

As housing continues to be both a necessity and a major investment for Singaporeans, data-driven insights will become increasingly valuable in navigating this complex market. By quantifying the impact of various factors on resale prices, we move toward a more transparent, efficient, and equitable housing ecosystem.

---

*Want to explore this project in more detail? Check out the [complete project page](/projects/hdb-resale-prices/) or view the [source code on GitHub](https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price).*
