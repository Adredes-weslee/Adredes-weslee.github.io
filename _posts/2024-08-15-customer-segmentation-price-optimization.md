---
layout: post
title: "Transforming Commercial Strategy with Data Science: Customer Segmentation and Price Optimization"
date: 2024-08-15 10:30:00 +0800
categories: [data-science, pricing-strategy, business-analytics, commercial-strategy]
tags: [rfm-analysis, k-means-clustering, price-elasticity, revenue-optimization, commercial-strategy]
author: Wes Lee
feature_image: /assets/images/2024-08-15-customer-segmentation-price-optimization.jpg
---

# Transforming Commercial Strategy with Data Science

As part of my BCG RISE 2.0 program capstone, I had the opportunity to lead a data science project that delivered tangible business impact for CS Tay, a leading frozen food distributor in Singapore. This blog post details our approach to solving real-world commercial challenges through customer segmentation, price elasticity modeling, and revenue optimization.

## The Business Challenge

CS Tay faced typical challenges of mature businesses in competitive markets:

- **Stagnating Revenue Growth:** Despite a solid customer base, revenue growth had plateaued
- **Margin Pressure:** Increasing competition and costs were compressing profit margins
- **Limited Customer Insights:** Despite having extensive transaction data, the company lacked actionable customer insights
- **Pricing Inefficiency:** Prices were set based on cost-plus formulas rather than customer willingness to pay
- **One-Size-Fits-All Approach:** The same commercial strategies were applied across diverse customer segments

The company needed a data-driven approach to revitalize its commercial strategy and drive profitable growth.

## Our Three-Phase Approach

We implemented a systematic, three-phase approach that combined business strategy with advanced data science techniques:

### Phase 1: Customer Segmentation & Analysis

The foundation of our approach was understanding customer behavior through advanced segmentation techniques:

1. **Data Preparation:** We cleaned and integrated 2+ years of transaction data covering 200+ customers and 800+ SKUs
2. **Enhanced RFM Analysis:** We calculated Recency, Frequency, Monetary and Total Quantity metrics for each customer
3. **Two-Stage K-Means Clustering:** We implemented a sophisticated clustering approach:
   - Used RobustScaler to handle outliers in the data
   - Applied silhouette analysis to determine optimal cluster counts
   - Executed first-stage clustering to identify major segments 
   - Performed second-stage clustering on the largest segment for more granular insights
4. **Cross-Category Analysis:** We analyzed segment distributions across customer categories (Supermarkets, Cafes, etc.) and product types (RTC, Raw, RTE)
5. **Segment Profiling:** We developed detailed profiles including business category distributions within each segment

Our analysis revealed a more nuanced segmentation than initially expected. Using a two-stage K-Means clustering approach with silhouette analysis to determine optimal cluster counts, we identified five key segments:

- **Champions (0.3%):** Elite customers with extremely high frequency (9,680+ transactions), high monetary value (over SGD 13M), and recent activity
- **Potential Loyalists (0.1%):** Recent customers with high-frequency patterns (200+ transactions) and significant monetary value (SGD 10M+)
- **New Customers (80%):** Recent purchasers with varying frequency and spending patterns
- **Hibernating (11%):** Previously active customers with moderate recency scores (average 280 days) and modest spending
- **Lost (9%):** Inactive customers with very poor recency scores (average 511 days) and low frequency

Visualization of the segmentation revealed clear distinctions between clusters:

![Customer Segment Distribution](/assets/images/Customer Segment Distribution.jpg)

Importantly, we identified distinct customer category distributions across segments:
- **Cafes** made up the largest portion of our New Customers segment (240 businesses)
- **Schools** represented the second largest group (100 businesses) in New Customers
- **Wholesalers** and **Wet Markets** showed potential for growth with 39 and 42 businesses respectively
- **Supermarkets** were disproportionately represented in our Champions segment

```python
# Advanced RFM Calculation with Total Quantity
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from yellowbrick.cluster import KElbowVisualizer
from datetime import timedelta

# Calculate snapshot date (one day after last transaction)
snapshot_date = cs['Transaction Date'].max() + timedelta(days=1)
print(f"Last Invoice Date: {cs['Transaction Date'].max()}")
print(f"Snapshot Date: {snapshot_date}")  # 2024-02-01

# Calculate recency (days since last purchase)
cs['Recency'] = (snapshot_date - cs['Transaction Date']).dt.days
recency = cs.groupby('Customer Code')['Recency'].min().reset_index()

# Calculate frequency (number of transactions)
frequency = cs.groupby('Customer Code')['Sales Order No.'].nunique().reset_index()

# Calculate monetary value (total spend)
monetary = cs.groupby('Customer Code')['Total Base Amt'].sum().reset_index()

# Calculate total quantity purchased
total_quantity = cs.groupby('Customer Code')['Qty'].sum().reset_index()

# Merge all RFM metrics
rfm = pd.merge(recency, frequency, on='Customer Code', how='inner')
rfm = pd.merge(rfm, monetary, on='Customer Code', how='inner')
rfm = pd.merge(rfm, total_quantity, on='Customer Code', how='inner')
rfm.columns = ['Customer Code', 'Recency', 'Frequency', 'Monetary', 'Total Quantity']

# Advanced two-stage clustering
# First stage: optimal cluster detection using silhouette scores
features_for_segmentation = ['Recency', 'Frequency', 'Monetary', 'Total Quantity']
scaler = RobustScaler()
scaled_data = scaler.fit_transform(rfm[features_for_segmentation])

# Find optimal k using silhouette analysis
silhouette_scores = {}
for k in range(2, 7):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(scaled_data)
    silhouette = silhouette_score(scaled_data, labels)
    silhouette_scores[k] = silhouette
    print(f"K={k}, Silhouette Score={silhouette}")

# First-stage clustering with k=3 (best silhouette score)
kmeans = KMeans(n_clusters=3, random_state=42)
rfm["KMeans_Segment"] = kmeans.fit_predict(scaled_data) + 1

# Second-stage clustering on largest segment
cluster_1 = rfm[rfm['KMeans_Segment'] == 1]
cluster_1_scaled = scaler.fit_transform(cluster_1[features_for_segmentation])
kmeans_stage2 = KMeans(n_clusters=3, random_state=42)
cluster_1["KMeans_Segment"] = kmeans_stage2.fit_predict(cluster_1_scaled) + 4

# Create enhanced RFM score segments
def rfm_segment(customer):
    if customer['Score'] >= 27:
        return 'Champions'
    elif customer['Score'] >= 15:
        return 'New_Customers'
    elif customer['Score'] >= 6:
        return 'Hibernating'
    else:
        return 'Lost'

rfm['RFM_Segment'] = rfm.apply(rfm_segment, axis=1)

# Visualizations for customer segments
import squarify
import matplotlib.pyplot as plt
import seaborn as sns

# Create treemap visualization of customer segments
segment_counts = rfm['RFM_Segment'].value_counts()
plt.figure(figsize=(12, 8))
squarify.plot(sizes=segment_counts.values, label=segment_counts.index, alpha=0.7)
plt.axis('off')
plt.title('Customer Segments Treemap')

# Analyze segment distribution across customer categories
# Merge segmentation with customer category data
cust_cat = cs[['Customer Code', 'Customer Category Desc', 'Item Category']]
cust_cat = cust_cat.drop_duplicates(subset='Customer Code', keep='first')
segmented_with_categories = pd.merge(rfm, cust_cat, on='Customer Code', how='inner')

# Create visualizations of business categories by segment
plt.figure(figsize=(15, 10))
sns.countplot(y='Customer Category Desc', hue='RFM_Segment', data=segmented_with_categories)
plt.title('Customer Category Distribution by Segment')
plt.xlabel('Count')
plt.tight_layout()
```

### Phase 2: Price Elasticity Modeling

With customer segments defined, we moved to understand how each segment responds to price changes:

1. **Data Aggregation:** Aggregated sales data by segment, product, and time period
2. **Log-Log Regression:** Built log-log regression models to calculate price elasticities
3. **Elasticity Analysis:** Calculated own-price and cross-price elasticities for top SKUs
4. **Substitution Effects:** Identified product substitution patterns within segments

The elasticity findings were insightful:

- **Segment Differences:** Champions were less price-sensitive than At-Risk customers
- **Category Variations:** Fresh seafood had lower price elasticity than processed foods
- **Substitution Effects:** Several clear substitution patterns emerged, especially in the retail segment
- **Volume Sensitivity:** Bulk buyers showed different elasticity patterns than small-volume customers

Our detailed analysis revealed that in the Supermarket segment, Raw products like SP01 Skinless Chicken Breast showed high price elasticity (-2.61), meaning customers were very sensitive to price changes. Meanwhile, Ready-to-Cook (RTC) products exhibited positive elasticity in some cases, indicating potential for price optimization without significantly reducing demand.

Cross-price elasticity analysis identified valuable product relationships:
- Complementary products (-0.85 to -5.41 elasticity) that should be bundled together for promotions
- Substitute products (0.28 to 31.73 elasticity) that compete for the same consumer need
- Price-insensitive premium products ideal for margin improvement

```python
# Log-log regression for elasticity calculation
def calculate_elasticity(segment_df, product):
    X = sm.add_constant(segment_df['log_price'])
    model = sm.OLS(segment_df['log_quantity'], X).fit()
    return model.params['log_price']  # This is the elasticity coefficient

# Cross-elasticity calculation
def calculate_cross_elasticity(df, sku_a, sku_b):
    merged_data = pd.merge(
        df[df['Inventory Code'] == sku_a], 
        df[df['Inventory Code'] == sku_b], 
        on='Transaction Date', suffixes=('_a', '_b')
    )
    X = sm.add_constant(merged_data['log_price_b'])
    model = sm.OLS(merged_data['log_qty_a'], X).fit()
    return model.params['log_price_b']  # Cross-price elasticity
```

### Phase 3: Revenue Optimization

Finally, we used the insights from segmentation and elasticity modeling to optimize pricing:

1. **Optimization Modeling:** Built a constrained optimization model using Gurobi
2. **Business Constraints:** Incorporated realistic constraints like maximum price changes
3. **Scenario Analysis:** Tested different constraint scenarios to understand tradeoffs
4. **Segment-Specific Strategies:** Developed tailored pricing strategies for each segment

The optimization model allowed us to:

- Identify optimal price points for each product in each segment
- Calculate the potential revenue impact of price changes
- Respect practical business constraints like maximum price changes
- Create a realistic implementation roadmap

Our optimization analysis revealed significant revenue improvement potential, particularly in the Supermarket segment where we identified over SGD 4M in potential annual revenue uplift. By treating each customer category separately and considering both own-price and cross-price elasticities, we developed tailored pricing strategies that balanced revenue growth with practical implementation constraints.

```python
# Revenue optimization with constraints using Gurobi
from gurobipy import Model, GRB, quicksum

def optimize_prices(elasticities, base_prices, base_quantities, constraints):
    m = Model("price_optimization")
    
    # Decision variables (price multipliers)
    price_vars = {}
    for product in elasticities.keys():
        price_vars[product] = m.addVar(
            lb=constraints['min_mult'], 
            ub=constraints['max_mult'], 
            name=f"price_mult_{product}"
        )
    
    # Objective: maximize revenue
    revenue = quicksum(
        base_quantities[p] * base_prices[p] * price_vars[p] * 
        (price_vars[p] ** elasticities[p]) 
        for p in elasticities.keys()
    )
    
    m.setObjective(revenue, GRB.MAXIMIZE)
    
    # Add constraints on average price increase
    avg_increase = quicksum(price_vars[p] for p in elasticities.keys()) / len(elasticities)
    m.addConstr(avg_increase <= constraints['max_avg_increase'])
    
    # Add constraint on max price change for premium products
    for p in constraints.get('premium_products', []):
        if p in price_vars:
            m.addConstr(price_vars[p] <= constraints['premium_max_mult'])
    
    # Solve and return results
    m.optimize()
    return {p: price_vars[p].X for p in elasticities.keys()}, m.ObjVal
```

## Results and Business Impact

Our data-driven approach delivered significant potential business value:

- **Revenue Potential:** Identified SGD ~4M/year in potential revenue uplift through optimized pricing
- **Segment Strategies:** Created tailored commercial strategies for each customer segment
- **Product Insights:** Uncovered valuable product substitution patterns to inform bundling strategies
- **Implementation Roadmap:** Developed a phased implementation plan with monitoring mechanisms
- **Organizational Capabilities:** Built data science capabilities within CS Tay's commercial team

## Key Takeaways

This project reinforced several important principles for successful data science applications in business:

1. **Start with Business Context:** Understanding the business challenge thoroughly before diving into technical solutions
2. **Combine Multiple Techniques:** Integrating segmentation, elasticity modeling, and optimization for a comprehensive solution
3. **Balance Sophistication with Practicality:** Creating models that were mathematically sound yet practically implementable
4. **Focus on Actionability:** Ensuring all insights could be translated into specific business actions
5. **Consider Implementation Realities:** Designing solutions with real-world implementation constraints in mind

Data science is most powerful when it bridges the gap between technical sophistication and business practicality. This project demonstrated how advanced analytics can transform commercial strategy when properly aligned with business goals and operational realities.

## Technologies Used

The technical implementation leveraged several powerful data science tools:

- **Python** - Core programming language for all analysis
- **Pandas & NumPy** - Data manipulation and numerical computation
- **Scikit-learn** - Machine learning and clustering algorithms
- **Yellowbrick** - Enhanced visualization for machine learning models
- **Squarify** - Advanced treemap visualizations for segment analysis
- **StatsModels** - Statistical modeling for elasticity calculations
- **Matplotlib & Seaborn** - Data visualization
- **Gurobi** - Mathematical optimization for revenue maximization

This combination of tools allowed us to move from raw transaction data to actionable commercial insights in a rigorous, reproducible way.
