---
layout: post
title: "Unlocking Revenue: A Technical Walkthrough of Customer Segmentation and Price Optimization for CS Tay"
date: 2024-08-15 10:30:00 +0800 
categories: [data-science, pricing-strategy, business-analytics, commercial-strategy]
tags: [rfm-analysis, k-means-clustering, price-elasticity, revenue-optimization, python, pandas, scikit-learn, gurobi, capstone-project]
author: Wes Lee
feature_image: /assets/images/2024-08-15-customer-segmentation-price-optimization.jpg 
---

## Introduction: A Data Science Journey in Commercial Strategy

As part of my BCG RISE 2.0 program capstone, I undertook a data science project with CS Tay, a leading frozen food distributor in Singapore. The goal was to leverage data to address real-world commercial challenges. This post provides a technical deep dive into the methodologies we employed, from customer segmentation using RFM and K-Means to price elasticity modeling and revenue optimization with Gurobi.

> For a higher-level overview of the business context, outcomes, and impact of this project, please see the [Strategic Growth Engine Project Page](/projects/customer-segmentation-price-optimization-project-page/).

## The Starting Point: Business Challenges & Data

CS Tay, like many mature businesses, faced stagnating revenue, margin pressures, and the limitations of a one-size-fits-all commercial approach. They had extensive transaction data (2+ years, 200+ customers, 800+ SKUs) but lacked actionable insights. Our mission was to transform this data into a data-driven commercial strategy.

## Phase 1: Deep Dive into Customer Segmentation

Understanding *who* the customers are and *how* they behave is fundamental. We employed a multi-step approach.

### 1. Data Preparation & Enhanced RFM Metrics

First, we cleaned and integrated the transaction data. Then, we calculated enhanced RFM (Recency, Frequency, Monetary) metrics, adding Total Quantity as a fourth dimension to capture purchase volume.

```python
# Python (Illustrative - assumes 'cs' is the preloaded DataFrame)
import pandas as pd
from datetime import timedelta

# Calculate snapshot date (one day after last transaction)
snapshot_date = cs['Transaction Date'].max() + timedelta(days=1)
print(f"Snapshot Date: {snapshot_date}")

# Calculate Recency: Days since last purchase for each customer
cs['Recency'] = (snapshot_date - cs['Transaction Date']).dt.days
recency_df = cs.groupby('Customer Code')['Recency'].min().reset_index()

# Calculate Frequency: Number of unique sales orders (transactions)
frequency_df = cs.groupby('Customer Code')['Sales Order No.'].nunique().reset_index()
frequency_df.columns = ['Customer Code', 'Frequency']

# Calculate Monetary Value: Total sum of 'Total Base Amt'
monetary_df = cs.groupby('Customer Code')['Total Base Amt'].sum().reset_index()
monetary_df.columns = ['Customer Code', 'Monetary']

# Calculate Total Quantity: Sum of 'Qty'
total_quantity_df = cs.groupby('Customer Code')['Qty'].sum().reset_index()
total_quantity_df.columns = ['Customer Code', 'Total Quantity']

# Merge all RFM-Q metrics into a single DataFrame
rfm_q_df = pd.merge(recency_df, frequency_df, on='Customer Code', how='inner')
rfm_q_df = pd.merge(rfm_q_df, monetary_df, on='Customer Code', how='inner')
rfm_q_df = pd.merge(rfm_q_df, total_quantity_df, on='Customer Code', how='inner')

print("RFM-Q DataFrame head:")
print(rfm_q_df.head())
```

### 2. Two-Stage K-Means Clustering for Nuanced Segments

To identify customer segments, we used K-Means clustering. Given potential outliers in business data, `RobustScaler` was chosen for feature scaling.

**Determining Optimal Cluster Numbers (k):**
We used silhouette analysis to find the optimal number of clusters. The silhouette score measures how similar an object is to its own cluster compared to other clusters.

```python
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
# from yellowbrick.cluster import KElbowVisualizer # KElbowVisualizer can also be used

features_for_segmentation = ['Recency', 'Frequency', 'Monetary', 'Total Quantity']
rfm_features = rfm_q_df[features_for_segmentation]

# Scale features using RobustScaler
scaler = RobustScaler()
scaled_rfm_features = scaler.fit_transform(rfm_features)

# Silhouette analysis to find optimal k for the first stage
silhouette_scores_stage1 = {}
k_range = range(2, 7) # Test k from 2 to 6

for k in k_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels = kmeans.fit_predict(scaled_rfm_features)
    silhouette_avg = silhouette_score(scaled_rfm_features, cluster_labels)
    silhouette_scores_stage1[k] = silhouette_avg
    print(f"For k={k}, the average silhouette_score is: {silhouette_avg}")

# Optimal k for first stage was found to be 3 (based on original analysis)
optimal_k_stage1 = 3
kmeans_stage1 = KMeans(n_clusters=optimal_k_stage1, init='k-means++', n_init=10, random_state=42)
rfm_q_df['KMeans_Segment_Stage1'] = kmeans_stage1.fit_predict(scaled_rfm_features) + 1 # Segments 1, 2, 3
```

**Second-Stage Clustering:**
The largest segment from the first stage was often too broad. We performed a second K-Means clustering step on this segment to achieve more granular insights.

```python
# Assuming Segment 1 from Stage 1 is the largest and needs further clustering
largest_segment_data = rfm_q_df[rfm_q_df['KMeans_Segment_Stage1'] == 1].copy() # Use .copy() to avoid SettingWithCopyWarning
scaled_largest_segment_features = scaler.transform(largest_segment_data[features_for_segmentation]) # Use transform, not fit_transform

# Silhouette analysis for the second stage on the largest segment
silhouette_scores_stage2 = {}
for k in k_range: # Can use the same k_range or a different one
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    cluster_labels_s2 = kmeans.fit_predict(scaled_largest_segment_features)
    if len(set(cluster_labels_s2)) > 1: # Silhouette score requires at least 2 labels
        silhouette_avg_s2 = silhouette_score(scaled_largest_segment_features, cluster_labels_s2)
        silhouette_scores_stage2[k] = silhouette_avg_s2
        print(f"For k={k} (Stage 2), the average silhouette_score is: {silhouette_avg_s2}")

# Optimal k for second stage was found to be 3 (based on original analysis)
optimal_k_stage2 = 3
kmeans_stage2 = KMeans(n_clusters=optimal_k_stage2, init='k-means++', n_init=10, random_state=42)
# Assign new segment labels starting from where Stage 1 left off (e.g., 4, 5, 6)
largest_segment_data['KMeans_Segment_Final'] = kmeans_stage2.fit_predict(scaled_largest_segment_features) + optimal_k_stage1 + 1

# Combine segments: Non-largest segments from Stage 1 keep their labels
# Other segments from Stage 1 are assigned their original Stage 1 labels
final_segments = []
for index, row in rfm_q_df.iterrows():
    if row['KMeans_Segment_Stage1'] == 1: # If it was part of the largest segment
        # Find its new label from largest_segment_data
        final_segments.append(largest_segment_data.loc[index, 'KMeans_Segment_Final'])
    else:
        final_segments.append(row['KMeans_Segment_Stage1'])
rfm_q_df['KMeans_Segment_Final'] = final_segments

print("Final K-Means Segments head:")
print(rfm_q_df[['Customer Code', 'KMeans_Segment_Final']].head())
```
This two-stage approach led to five key segments: Champions, Potential Loyalists, New Customers, Hibernating, and Lost.

### 3. Visualizing and Profiling Segments
Visualizations like treemaps and count plots helped in understanding segment sizes and their distribution across business categories (Supermarkets, Cafes, etc.).

```python
import matplotlib.pyplot as plt
import seaborn as sns
import squarify # for treemaps

# Example: Treemap of final customer segments
segment_counts_final = rfm_q_df['KMeans_Segment_Final'].value_counts() # Use the final segment column
plt.figure(figsize=(12, 8))
squarify.plot(sizes=segment_counts_final.values, 
              label=[f'Segment {i}' for i in segment_counts_final.index], 
              alpha=0.7, color=sns.color_palette("viridis", len(segment_counts_final)))
plt.axis('off')
plt.title('Final Customer Segments Treemap (K-Means)')
plt.show()

# Example: Distribution by customer category (assuming 'cust_cat_df' is merged)
# cust_cat_df = cs[['Customer Code', 'Customer Category Desc']].drop_duplicates()
# rfm_q_df_merged = pd.merge(rfm_q_df, cust_cat_df, on='Customer Code', how='left')

# plt.figure(figsize=(15, 10))
# sns.countplot(y='Customer Category Desc', hue='KMeans_Segment_Final', data=rfm_q_df_merged)
# plt.title('Customer Category Distribution by Final K-Means Segment')
# plt.xlabel('Count')
# plt.tight_layout()
# plt.show()
```

## Phase 2: Modeling Price Elasticity of Demand

With segments defined, we aimed to understand their price sensitivity.

### 1. Data Aggregation and Log-Log Regression
Sales data was aggregated by segment, product, and time. We used a log-log regression model (log(Quantity) ~ log(Price)) because the coefficient of log(Price) directly gives the price elasticity.

```python
import statsmodels.api as sm
import numpy as np

# Assume 'sales_data_agg' is a DataFrame with:
# 'Segment', 'Product_ID', 'Period', 'log_Quantity', 'log_Price'

def calculate_price_elasticity(df, segment_col, product_col, log_qty_col, log_price_col):
    """Calculates price elasticity for each product within each segment."""
    elasticities = {}
    for segment in df[segment_col].unique():
        elasticities[segment] = {}
        segment_data = df[df[segment_col] == segment]
        for product in segment_data[product_col].unique():
            product_segment_data = segment_data[segment_data[product_col] == product]
            if len(product_segment_data) > 10: # Need enough data points
                Y = product_segment_data[log_qty_col]
                X = product_segment_data[log_price_col]
                X = sm.add_constant(X) # Add intercept
                
                model = sm.OLS(Y, X, missing='drop').fit()
                price_elasticity_coefficient = model.params[log_price_col]
                # Optionally, store p-value: model.pvalues[log_price_col]
                elasticities[segment][product] = price_elasticity_coefficient
            else:
                elasticities[segment][product] = np.nan # Not enough data
    return elasticities

# Example:
# elasticity_results = calculate_price_elasticity(sales_data_agg, 
#                                               'KMeans_Segment_Final', 
#                                               'Inventory Code', 
#                                               'log_Qty', # Ensure these columns exist
#                                               'log_UnitPrice') # Ensure these columns exist
# print(elasticity_results)
```

### 2. Analyzing Own-Price and Cross-Price Elasticities
This revealed how demand for a product changes with its own price and with the price changes of other (substitute or complementary) products. For instance, Raw products like SP01 Skinless Chicken Breast showed high price elasticity (-2.61) in the Supermarket segment.

```python
# Conceptual: Cross-price elasticity (requires careful data alignment)
def calculate_cross_price_elasticity(df, product_a_id, product_b_id, segment_col, log_qty_a_col, log_price_b_col):
    """Calculates cross-price elasticity of product A with respect to product B's price."""
    # Data needs to be prepared such that for each observation, we have qty_A and price_B
    # This often involves merging or careful time-series alignment.
    # For simplicity, assuming 'merged_data_for_cross_el' is pre-prepared:
    # Columns: 'Segment', log_qty_A_col, log_price_B_col
    
    cross_elasticities = {}
    # for segment in merged_data_for_cross_el[segment_col].unique():
    #     segment_data = merged_data_for_cross_el[merged_data_for_cross_el[segment_col] == segment]
    #     if len(segment_data) > 10:
    #         Y = segment_data[log_qty_a_col]
    #         X = segment_data[log_price_b_col]
    #         X = sm.add_constant(X)
    #         model = sm.OLS(Y, X, missing='drop').fit()
    #         cross_elasticities[segment] = model.params[log_price_b_col]
    #     else:
    #         cross_elasticities[segment] = np.nan
    return cross_elasticities
```

## Phase 3: Optimizing Revenue with Gurobi

The final phase involved using these insights for price optimization. We built a constrained optimization model using Gurobi.

### 1. Optimization Model Formulation
The objective was to maximize total revenue, subject to business constraints (e.g., maximum allowable price changes for products, average price increase limits).

```python
from gurobipy import Model, GRB, quicksum

def optimize_prices_gurobi(product_elasticities, base_prices, base_quantities, 
                           min_price_multiplier=0.9, max_price_multiplier=1.2, 
                           max_avg_increase_multiplier=1.05,
                           premium_products_list=None, premium_max_multiplier=1.1):
    """
    Optimizes prices to maximize revenue using Gurobi.
    Args:
        product_elasticities (dict): {product_id: elasticity_value}
        base_prices (dict): {product_id: current_price}
        base_quantities (dict): {product_id: current_quantity_sold}
        min_price_multiplier (float): Min allowed price change factor (e.g., 0.9 for -10%)
        max_price_multiplier (float): Max allowed price change factor (e.g., 1.2 for +20%)
        max_avg_increase_multiplier (float): Max allowed average price increase factor
        premium_products_list (list, optional): List of premium product_ids.
        premium_max_multiplier (float): Max price multiplier for premium products.
    Returns:
        dict: Optimal price multipliers {product_id: multiplier}
        float: Max projected revenue
    """
    m = Model("PriceOptimization")

    products = list(product_elasticities.keys())

    # Decision Variables: Price Multiplier for each product
    # p_mult[i] is the factor by which base_price[i] is multiplied
    p_mult = m.addVars(products, lb=min_price_multiplier, ub=max_price_multiplier, name="price_multiplier")

    # Objective Function: Maximize Total Revenue
    # Revenue_i = New_Price_i * New_Quantity_i
    # New_Price_i = Base_Price_i * p_mult[i]
    # New_Quantity_i = Base_Quantity_i * (p_mult[i] ^ Elasticity_i)  (from log-log model)
    # So, Revenue_i = (Base_Price_i * p_mult[i]) * (Base_Quantity_i * (p_mult[i] ** Elasticity_i))
    # Revenue_i = Base_Price_i * Base_Quantity_i * (p_mult[i] ** (1 + Elasticity_i))
    
    # Gurobi's general power constraint: x_res = x_base ** exponent (requires x_base >= 0)
    # We need to introduce auxiliary variables for p_mult[i] ** (1 + Elasticity_i)
    
    # For simplicity in this example, let's assume Gurobi can handle this directly if exponent is constant.
    # If not, reformulation or piecewise linear approximation would be needed for non-integer/complex exponents.
    # Gurobi's addGenConstrPow() is the way for var ** constant.
    
    # Let's use an auxiliary variable for the quantity multiplier
    q_mult_factor = m.addVars(products, name="quantity_multiplier_factor")
    for prod_id in products:
        # q_mult_factor[prod_id] = p_mult[prod_id] ** product_elasticities[prod_id]
        m.addGenConstrPow(p_mult[prod_id], q_mult_factor[prod_id], product_elasticities[prod_id], 
                          name=f"pow_constr_{prod_id}")

    total_revenue = quicksum(
        base_prices[prod_id] * p_mult[prod_id] * base_quantities[prod_id] * q_mult_factor[prod_id]
        for prod_id in products
    )
    m.setObjective(total_revenue, GRB.MAXIMIZE)

    # Constraint: Average price increase limit
    if len(products) > 0:
        m.addConstr(quicksum(p_mult[prod_id] for prod_id in products) / len(products) <= max_avg_increase_multiplier, 
                    "AvgPriceIncreaseLimit")

    # Constraint: Max price change for premium products
    if premium_products_list:
        for prod_id in premium_products_list:
            if prod_id in products: # Ensure product is in the optimization scope
                m.addConstr(p_mult[prod_id] <= premium_max_multiplier, f"PremiumLimit_{prod_id}")
                
    m.params.NonConvex = 2 # Allow Gurobi to solve non-convex quadratic objectives/constraints

    m.optimize()

    if m.status == GRB.OPTIMAL:
        optimal_multipliers = {prod_id: p_mult[prod_id].X for prod_id in products}
        return optimal_multipliers, m.ObjVal
    else:
        print("Optimization was not successful. Status:", m.status)
        return None, None

# Example usage:
# optimal_multipliers, max_revenue = optimize_prices_gurobi(
#     product_elasticities_segment_x, 
#     base_prices_segment_x, 
#     base_quantities_segment_x,
#     premium_products_list=['SKU001', 'SKU005']
# )
# if optimal_multipliers:
#     print("Optimal Price Multipliers:", optimal_multipliers)
#     print("Projected Max Revenue:", max_revenue)
```

### 2. Scenario Analysis and Strategy Development
The model allowed us to test different constraints and develop segment-specific pricing strategies, identifying significant revenue uplift potential (e.g., SGD ~4M annually in the Supermarket segment).

## Key Technical Lessons Learned

1.  **Iterative Segmentation:** Simple RFM scores are a good start, but multi-stage clustering with robust scaling and silhouette analysis provides deeper, more actionable segments.
2.  **Elasticity Nuances:** Price elasticity varies significantly not just by product but critically by customer segment. Cross-price elasticities are vital for understanding substitution/complementarity.
3.  **Constrained Optimization is Key:** Unconstrained optimization can yield unrealistic prices. Incorporating business rules (max/min price changes, category-level adjustments) is essential for practical implementation.
4.  **Tooling Matters:** Libraries like Scikit-learn, Statsmodels, and Gurobi (or other solvers like PuLP/CBC for open-source alternatives) are powerful enablers for each stage of such a project.
5.  **Data Quality and Granularity:** The success of elasticity and optimization models heavily depends on the quality, granularity (transaction-level), and length of historical sales data.

This project underscored how a systematic, multi-phase data science approach, combining statistical modeling with optimization, can translate raw business data into tangible commercial strategies and significant revenue impact.

---

*This post details the technical methodologies used in the CS Tay Commercial Strategy project, a capstone for the BCG RISE 2.0 program. For more on the business impact, see the [project page](/projects/customer-segmentation-price-optimization-project-page/).*
