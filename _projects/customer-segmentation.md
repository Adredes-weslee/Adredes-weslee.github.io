---
layout: project
title: "Customer Segmentation & Price Optimization"
date: 2024-08-15
categories: [data-science, business-analytics, pricing-strategy, commercial-strategy]
image: /assets/images/customer-segmentation.jpg
technologies: [Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, Yellowbrick, Squarify, RobustScaler, Gurobi, RFM Analysis, K-Means, Silhouette Analysis, Log-Log Regression, Constrained Optimization]
github: https://github.com/Adredes-weslee/BCG-RISE-2.0/tree/main/capstone
blog_post: /data-science/pricing-strategy/business-analytics/commercial-strategy/2024/08/15/customer-segmentation-price-optimization.html
---

## Project Overview

Led a full-cycle data science project for CS Tay, a leading frozen food distributor in Singapore, as part of the BCG RISE 2.0 program capstone. Our team delivered a data-driven commercial strategy to address stagnating revenue growth and declining profit margins in a competitive market environment.

> Read my detailed blog post: [Transforming Commercial Strategy with Data Science: Customer Segmentation and Price Optimization](/data-science/pricing-strategy/business-analytics/commercial-strategy/2024/08/15/customer-segmentation-price-optimization.html)

The project followed a systematic approach combining advanced analytics and business strategy:

1. **Comprehensive Market Analysis:** Examined market trends and competitive forces in Singapore's frozen food distribution sector
2. **Customer Behavior Analysis:** Analyzed 2+ years of transaction data covering 200+ customers and 800+ SKUs
3. **Commercial Strategy Development:** Created actionable recommendations with potential SGD ~4M revenue uplift

## Methodology

### Customer Segmentation
- Implemented RFM (Recency, Frequency, Monetary) analysis to evaluate customer behavior
- Applied K-Means clustering with silhouette analysis to identify optimal customer segments
- Conducted two-stage clustering: first separating major segments, then further refining within clusters
- Classified customers into personas including Champions, New Customers, Hibernating, and Lost
- Cross-analyzed segments with business categories (Supermarkets, Cafes, Wholesalers) for targeted strategies

### Demand Modeling
- Built a log-log regression model to estimate price elasticities
- Calculated own-price and cross-price elasticities across top SKUs
- Identified product substitution patterns for strategic pricing

### Price Optimization
- Formulated a constrained optimization model using Gurobi
- Simulated revenue-maximizing price points under real-world constraints
- Developed targeted marketing playbooks based on segment behavior

## Results and Impact

The project delivered significant potential business value:

- Identified SGD ~4M/year in potential revenue uplift through optimized pricing
- Surfaced SKU substitution patterns within the Retail segment to inform bundling and discounting
- Provided actionable marketing playbooks for each customer segment
- Presented findings to CS Tay stakeholders and BCG consultants

## Technical Implementation

### Customer Segmentation (RFM Analysis & Two-Stage K-Means Clustering)

```python
# Customer Segmentation using advanced RFM analysis and two-stage K-Means clustering
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
from yellowbrick.cluster import KElbowVisualizer

# Calculate RFM metrics - More comprehensive approach
def calculate_rfm(df):
    # Define snapshot date (one day after last transaction)
    snapshot_date = df['Transaction Date'].max() + timedelta(days=1)
    
    # Calculate recency (days since last purchase)
    df['Recency'] = (snapshot_date - df['Transaction Date']).dt.days
    recency = df.groupby('Customer Code')['Recency'].min().reset_index()
    
    # Calculate frequency (number of transactions)
    frequency = df.groupby('Customer Code')['Sales Order No.'].nunique().reset_index()
    
    # Calculate monetary value (total spend)
    monetary = df.groupby('Customer Code')['Total Base Amt'].sum().reset_index()
    
    # Calculate total quantity purchased
    quantity = df.groupby('Customer Code')['Qty'].sum().reset_index()
    
    # Merge all metrics
    rfm = pd.merge(recency, frequency, on='Customer Code', how='inner')
    rfm = pd.merge(rfm, monetary, on='Customer Code', how='inner')
    rfm = pd.merge(rfm, quantity, on='Customer Code', how='inner')
    
    return rfm

# Two-stage clustering approach for more nuanced segmentation
def two_stage_clustering(rfm_data, features):
    # Scale data using RobustScaler (better with outliers)
    scaler = RobustScaler()
    scaled_data = scaler.fit_transform(rfm_data[features])
    
    # First clustering stage - Find optimal k using silhouette scores
    silhouette_scores = {}
    for k in range(2, 13):
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(scaled_data)
        silhouette = silhouette_score(scaled_data, labels)
        silhouette_scores[k] = silhouette
        
    # Apply first stage clustering with optimal k=3
    kmeans_stage1 = KMeans(n_clusters=3, random_state=42)
    clusters_stage1 = kmeans_stage1.fit_predict(scaled_data)
    rfm_data["KMeans_Segment"] = clusters_stage1 + 1  # Start from 1 instead of 0
    
    # Second clustering stage for largest cluster
    cluster_1 = rfm_data[rfm_data['KMeans_Segment'] == 1]
    cluster_1_scaled = scaler.fit_transform(cluster_1[features])
    
    # Apply second stage clustering to largest segment
    kmeans_stage2 = KMeans(n_clusters=3, random_state=42)
    clusters_stage2 = kmeans_stage2.fit_predict(cluster_1_scaled)
    cluster_1["KMeans_Segment"] = clusters_stage2 + 4  # New segment IDs: 4,5,6
    
    # Combine results from both stages
    return pd.concat([cluster_1, rfm_data[rfm_data['KMeans_Segment'].isin([2, 3])]], axis=0)

# Main segmentation execution
features = ['Recency', 'Frequency', 'Monetary', 'Total Quantity']
rfm = calculate_rfm(transactions)

# Execute two-stage clustering
segmented_customers = two_stage_clustering(rfm, features)

# Add RFM score-based segments for comparison
def rfm_segment(customer):
    if customer['Score'] >= 27:
        return 'Champions'
    elif customer['Score'] >= 21:
        return 'Loyal_Customers'
    elif customer['Score'] >= 18:
        return 'Potential_Loyalists'
    elif customer['Score'] >= 15:
        return 'New_Customers'
    elif customer['Score'] >= 12:
        return 'At_Risk'
    elif customer['Score'] >= 6:
        return 'Hibernating'
    else:
        return 'Lost'

segmented_customers['RFM_Segment'] = segmented_customers.apply(rfm_segment, axis=1)
```

### Customer Segmentation Visualizations

```python
# Visualize segment distribution and characteristics
import matplotlib.pyplot as plt
import seaborn as sns
import squarify

# Create treemap of customer segments
segment_counts = segmented_customers['RFM_Segment'].value_counts()
plt.figure(figsize=(12, 8))
squarify.plot(sizes=segment_counts.values, label=segment_counts.index, alpha=0.7)
plt.axis('off')
plt.title('Customer Segments Distribution')

# Analyze segment characteristics across business categories
fig, axes = plt.subplots(1, 2, figsize=(18, 8))
# Distribution by customer category
sns.countplot(y='Customer Category Desc', hue='RFM_Segment', data=customer_data, ax=axes[0])
axes[0].set_title('Customer Categories by Segment')
# Distribution by product type
sns.countplot(y='Item Category', hue='RFM_Segment', data=customer_data, ax=axes[1])
axes[1].set_title('Product Types by Segment')
plt.tight_layout()
```

### Price Elasticity Modeling

```python
# Log-log regression model for price elasticity
import statsmodels.api as sm

# Prepare data for elasticity modeling
def prepare_elasticity_data(df):
    # Aggregate data by product, customer segment and time period
    agg_df = df.groupby(['product_id', 'segment', 'period']).agg({
        'quantity': 'sum',
        'price': 'mean'
    }).reset_index()
    
    # Log transform for elasticity calculation
    agg_df['log_quantity'] = np.log(agg_df['quantity'])
    agg_df['log_price'] = np.log(agg_df['price'])
    
    return agg_df

# Run regression for each product in each segment
def calculate_elasticity(df, segment):
    segment_df = df[df['segment'] == segment]
    elasticities = {}
    
    for product in segment_df['product_id'].unique():
        product_df = segment_df[segment_df['product_id'] == product]
        X = sm.add_constant(product_df['log_price'])
        model = sm.OLS(product_df['log_quantity'], X).fit()
        elasticities[product] = model.params['log_price']
        
    return elasticities
```

### Revenue Optimization

```python
# Revenue optimization using Gurobi
from gurobipy import Model, GRB, quicksum

def optimize_prices(elasticities, base_prices, base_quantities, constraints):
    m = Model("price_optimization")
    
    # Decision variables - price multipliers
    price_vars = {}
    for product in elasticities.keys():
        price_vars[product] = m.addVar(
            lb=constraints['min_mult'], 
            ub=constraints['max_mult'], 
            name=f"price_mult_{product}"
        )
    
    # Objective function: maximize revenue
    revenue = quicksum(
        base_quantities[p] * base_prices[p] * price_vars[p] * 
        (price_vars[p] ** elasticities[p]) 
        for p in elasticities.keys()
    )
    
    m.setObjective(revenue, GRB.MAXIMIZE)
    
    # Add constraints on average price increase
    avg_increase = quicksum(price_vars[p] for p in elasticities.keys()) / len(elasticities)
    m.addConstr(avg_increase <= constraints['max_avg_increase'])
    
    # Solve the model
    m.optimize()
    
    # Extract results
    results = {p: price_vars[p].X for p in elasticities.keys()}
    projected_revenue = m.ObjVal
    
    return results, projected_revenue
```

## Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - For clustering algorithms
- **Matplotlib** - Data visualization
- **Seaborn** - Advanced data visualization
- **StatsModels** - Statistical modeling and hypothesis testing
- **Gurobi** - Mathematical optimization
- **RFM Analysis** - Customer segmentation framework
- **K-Means** - Clustering algorithm
- **Log-Log Regression** - Price elasticity modeling
- **Constrained Optimization** - Revenue maximization while meeting business constraints

## Key Learnings

- **Business-Technology Integration:** Successfully bridged business strategy and data science to create actionable commercial recommendations
- **Stakeholder Communication:** Translated complex technical findings into clear business insights for non-technical stakeholders
- **End-to-End Project Management:** Led a complete data science project from problem definition to implementation planning
- **Constraint-Based Modeling:** Incorporated real-world business constraints into mathematical optimization models
- **Commercial Strategy:** Developed deep understanding of how data science can drive pricing and segmentation strategies
