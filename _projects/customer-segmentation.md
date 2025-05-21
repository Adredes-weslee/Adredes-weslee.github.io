---
layout: project
title: "Strategic Growth Engine: Data-Driven Customer Segmentation & Price Optimization for CS Tay"
date: 2024-08-15 # Retaining original date
categories: [data-science, business-analytics, pricing-strategy, commercial-strategy, capstone]
image: /assets/images/customer-segmentation.jpg # Or a new, more strategic image
technologies: [Python, Pandas, Scikit-Learn, Matplotlib, Seaborn, StatsModels, Gurobi, RFM Analysis, K-Means Clustering, Price Elasticity Modeling, Revenue Optimization]
github: https://github.com/Adredes-weslee/price-optimization
blog_post: /data-science/pricing-strategy/business-analytics/commercial-strategy/2024/08/15/customer-segmentation-price-optimization.html # Link to the new blog post
---

## Project Overview

As a capstone project for the BCG RISE 2.0 program, I led a data science initiative for CS Tay, a prominent frozen food distributor in Singapore. The project aimed to revitalize the company's commercial strategy by leveraging data analytics to drive revenue growth and improve profit margins in a competitive market. Our team delivered actionable insights through advanced customer segmentation, price elasticity modeling, and revenue optimization, identifying a potential SGD ~4M annual revenue uplift.

<div class="demo-link-container">
  <a href="https://adredes-weslee-price-optimization-streamlitapp-yxjoe3.streamlit.app/" class="demo-button" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-chart-line"></i> Try the Price Optimization Demo
  </a>
</div>

## The Business Challenge: Navigating a Mature Market

CS Tay faced several pressing commercial challenges:
* **Stagnating Revenue Growth:** Difficulty in expanding revenue despite a substantial customer base.
* **Margin Pressure:** Increasing operational costs and competitive pricing were eroding profitability.
* **Limited Customer Insights:** An abundance of transaction data was not being effectively utilized to understand customer behavior or tailor strategies.
* **Inefficient Pricing:** Pricing decisions were primarily cost-plus, not reflecting customer willingness-to-pay or market dynamics.
* **Homogeneous Strategy:** A one-size-fits-all approach was applied across a diverse customer portfolio, limiting effectiveness.

The core objective was to transition CS Tay towards a data-informed commercial decision-making framework.

## Our Strategic Three-Phase Solution

We adopted a structured, three-phase methodology to address these challenges:

![Three Phase Approach Diagram (Conceptual - User to provide or describe if specific image exists)](/assets/images/three-phase-approach-placeholder.png) **Phase 1: Customer Segmentation & In-depth Analysis**
* **Objective:** To identify distinct customer groups based on their purchasing behavior.
* **Methodology:**
    * Processed and cleaned over two years of transaction data (200+ customers, 800+ SKUs).
    * Conducted enhanced RFM (Recency, Frequency, Monetary) analysis, incorporating Total Quantity purchased.
    * Implemented a two-stage K-Means clustering algorithm, using `RobustScaler` for outlier handling and silhouette analysis to determine the optimal number of clusters at each stage.
* **Key Outcome:** Identified five key customer segments (Champions, Potential Loyalists, New Customers, Hibernating, Lost) with distinct profiles and purchasing patterns. For example, Supermarkets were disproportionately represented in the 'Champions' segment, while 'Cafes' constituted a large portion of 'New Customers'.

**Phase 2: Price Elasticity Modeling & Demand Analysis**
* **Objective:** To understand how different customer segments and products respond to price changes.
* **Methodology:**
    * Aggregated sales data by identified customer segments, products, and time periods.
    * Built log-log regression models to estimate own-price and cross-price elasticities for key SKUs within each segment.
* **Key Outcome:** Quantified price sensitivity across segments and product categories. For instance, 'Champions' were found to be less price-sensitive than 'Hibernating' customers. Fresh seafood showed lower elasticity than processed foods. Significant product substitution and complementarity effects were identified (e.g., Raw products like SP01 Skinless Chicken Breast showed high elasticity of -2.61 in Supermarkets).

**Phase 3: Revenue Optimization & Strategic Recommendations**
* **Objective:** To develop segment-specific pricing strategies that maximize revenue under realistic business constraints.
* **Methodology:**
    * Formulated a constrained optimization model using Gurobi, incorporating calculated elasticities.
    * Inputs included base prices, base quantities, elasticities, and business rules (e.g., maximum allowable price changes, average price increase limits).
    * Conducted scenario analysis to evaluate trade-offs.
* **Key Outcome:** Developed tailored pricing strategies for each customer segment and product category. The model projected a potential annual revenue uplift of approximately SGD 4 million, primarily from optimizing prices in the Supermarket segment. An actionable implementation roadmap was provided.

## Results and Tangible Business Impact

This data-driven approach yielded significant strategic insights and quantifiable potential benefits for CS Tay:

* **Identified Revenue Uplift:** Pinpointed an estimated **SGD ~4M per year** in potential revenue increase through optimized, segment-specific pricing.
* **Actionable Customer Segments:** Provided clear profiles for five distinct customer segments (Champions, Potential Loyalists, New Customers, Hibernating, Lost), enabling targeted marketing and sales strategies.
    * _Champions (0.3%):_ High-value, high-frequency.
    * _Potential Loyalists (0.1%):_ Recent, high-frequency, high-value.
    * _New Customers (80%):_ Largest group, varied behavior.
    * _Hibernating (11%):_ Declining activity.
    * _Lost (9%):_ Inactive.
* **Strategic Pricing Insights:** Revealed varying price sensitivities across segments and product categories, allowing for more nuanced pricing decisions beyond cost-plus.
* **Product Portfolio Optimization:** Uncovered product substitution and complementarity patterns, informing bundling, cross-selling, and promotional strategies.
* **Data-Driven Capabilities:** Enhanced CS Tay's internal capabilities for data analysis and strategic decision-making.
* **Implementation Roadmap:** Delivered a phased plan for rolling out the new pricing strategies, including key metrics for monitoring success.

## Technical Implementation Overview

The project leveraged a robust stack of data science tools and methodologies:

* **Data Processing & Analysis:** Python with Pandas and NumPy for data manipulation.
* **Customer Segmentation:**
    * RFM (Recency, Frequency, Monetary, Total Quantity) feature engineering.
    * Scikit-learn for K-Means clustering, `RobustScaler` for feature scaling, and `silhouette_score` for cluster validation.
    * Visualization with Matplotlib, Seaborn, and Squarify for treemaps.
* **Price Elasticity Modeling:**
    * StatsModels for log-log regression analysis to determine price elasticity coefficients.
* **Revenue Optimization:**
    * Gurobi for solving constrained mathematical optimization problems to maximize revenue.

```python
# Illustrative Snippet: RFM Calculation
# snapshot_date = df['Transaction Date'].max() + timedelta(days=1)
# rfm_df = df.groupby('Customer Code').agg(
#     Recency=('Transaction Date', lambda x: (snapshot_date - x.max()).days),
#     Frequency=('Sales Order No.', 'nunique'),
#     Monetary=('Total Base Amt', 'sum'),
#     TotalQuantity=('Qty', 'sum')
# ).reset_index()

# Illustrative Snippet: K-Means (Conceptual)
# from sklearn.cluster import KMeans
# from sklearn.preprocessing import RobustScaler
# scaler = RobustScaler()
# scaled_features = scaler.fit_transform(rfm_df[['Recency', 'Frequency', 'Monetary', 'TotalQuantity']])
# kmeans = KMeans(n_clusters=5, random_state=42, n_init=10) # Assuming k=5
# rfm_df['Segment'] = kmeans.fit_predict(scaled_features)

# Illustrative Snippet: Elasticity (Conceptual)
# import statsmodels.formula.api as smf
# model = smf.ols(formula='np.log(Qty) ~ np.log(Price)', data=product_segment_sales_data)
# results = model.fit()
# elasticity = results.params['np.log(Price)']

# Illustrative Snippet: Gurobi Optimization (Conceptual)
# from gurobipy import Model, GRB
# m = Model("RevenueMax")
# price_multipliers = m.addVars(products, lb=0.8, ub=1.2, name="price_mult")
# # ... define objective function based on revenue = new_price * new_quantity(elasticity) ...
# # ... add constraints ...
# m.optimize()
```

## Key Learnings & Strategic Value

This capstone project demonstrated the transformative power of applying data science to core commercial strategy. Key takeaways include:

* **Business Context is Crucial:** A deep understanding of the business problem and operational realities is essential for designing effective and implementable data science solutions.
* **Integrated Approach:** Combining multiple analytical techniques (segmentation, elasticity, optimization) yields a more holistic and robust strategy than any single method alone.
* **Actionability is Paramount:** Insights must be translated into clear, actionable recommendations that the business can execute.
* **Iterative Refinement:** Data science projects in business often require an iterative approach, refining models and analyses based on initial findings and stakeholder feedback.

By embedding data-driven insights into its commercial operations, CS Tay is positioned to achieve sustainable revenue growth and enhanced profitability.

---

*This project was completed as part of the BCG RISE 2.0 Program. For a detailed technical walkthrough of the methodologies, please refer to the [accompanying blog post](/data-science/pricing-strategy/business-analytics/commercial-strategy/2024/08/15/customer-segmentation-price-optimization.html). The full codebase is available on [GitHub](https://github.com/Adredes-weslee/price-optimization).*
