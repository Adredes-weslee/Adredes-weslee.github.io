---
layout: project
title: "Customer Segmentation & Price Optimization"
categories: data-science business-analytics pricing-strategy
image: /assets/images/placeholder.svg
technologies: [Python, Pandas, Scikit-Learn, Matplotlib, Gurobi, RFM Analysis, K-Means]
github: https://github.com/Adredes-weslee/BCG-RISE-2.0/tree/main/capstone
---

## Project Overview

Led a full-cycle data science project for CS Tay, a leading frozen food distributor in Singapore, as part of the BCG RISE 2.0 program. Our team delivered actionable commercial strategies using customer segmentation, demand modeling, and pricing optimization.

## Methodology

### Customer Segmentation
- Implemented RFM (Recency, Frequency, Monetary) analysis to evaluate customer behavior
- Applied K-Means clustering to identify distinct customer segments
- Defined customer personas like Champions, Hibernating, and New customers

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

```python
# Customer Segmentation using RFM and K-Means
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Calculate RFM metrics
def create_rfm(df):
    # Recency
    max_date = df['date'].max()
    rfm = df.groupby('customer_id').agg({
        'date': lambda x: (max_date - x.max()).days,  # Recency
        'order_id': 'count',                         # Frequency
        'sales': 'sum'                             # Monetary
    }).reset_index()
    
    # Rename columns
    rfm.columns = ['customer_id', 'recency', 'frequency', 'monetary']
    return rfm

# Standardize and cluster
rfm = create_rfm(transactions)
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['recency', 'frequency', 'monetary']])

# Find optimal number of clusters
wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(rfm_scaled)
    wcss.append(kmeans.inertia_)

# Apply K-Means clustering
kmeans = KMeans(n_clusters=5, random_state=42)
rfm['cluster'] = kmeans.fit_predict(rfm_scaled)
```

## Technologies Used

- **Python** - Core programming language
- **Pandas** - Data manipulation and analysis
- **Scikit-learn** - For clustering algorithms
- **Matplotlib** - Data visualization
- **Gurobi** - Mathematical optimization
- **RFM Analysis** - Customer segmentation framework
- **K-Means** - Clustering algorithm
