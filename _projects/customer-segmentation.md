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

As a capstone project for the BCG RISE 2.0 program, I led a comprehensive data science initiative for CS Tay, a leading frozen food distributor in Singapore. This project transformed the company's commercial strategy through advanced analytics, delivering a sophisticated four-stage optimization pipeline that maximizes retail revenue through data-driven pricing strategies.

**Strategic Impact**: Our team identified a potential **SGD ~4M annual revenue uplift** through segment-specific pricing optimization, representing a breakthrough in pricing intelligence for the frozen food distribution industry.

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

![Three Phase Approach Diagram](/assets/images/three-phase-approach-placeholder.png) 

### **Phase 1: Advanced Customer Intelligence & Behavioral Segmentation**
* **Strategic Objective:** Transform customer data into actionable behavioral insights for targeted commercial strategies
* **Technical Implementation:**
    * **Data Pipeline**: Processed 2+ years of transaction history (200+ customers, 800+ SKUs) with advanced ETL validation
    * **Enhanced RFM Analytics**: Four-dimensional analysis (Recency, Frequency, Monetary, Total Quantity) with 10-bin quantile scoring
    * **Two-Stage K-Means Clustering**: Sophisticated hierarchical approach using `RobustScaler` for outlier resilience and silhouette optimization for cluster validation
    * **Statistical Validation**: Comprehensive cluster stability analysis and business rule validation
* **Business Outcome:** Identified five actionable customer segments with distinct behavioral profiles and revenue potential:
    * **Champions** (Enterprise Focus): High-value customers driving disproportionate revenue
    * **Potential Loyalists** (Growth Opportunity): High-frequency customers with expansion potential  
    * **New Customers** (Market Penetration): Largest segment requiring nurturing strategies
    * **Hibernating** (Retention Critical): At-risk customers needing re-engagement
    * **Lost** (Win-Back Campaigns): Inactive customers for targeted reactivation

### **Phase 2: Econometric Price Intelligence & Market Dynamics Analysis**
* **Strategic Objective:** Quantify customer price sensitivity and competitive dynamics across segments and product categories
* **Technical Implementation:**
    * **Advanced Aggregation**: Multi-dimensional sales data consolidation by segment, product, and temporal dimensions
    * **Econometric Modeling**: Log-log OLS regression for direct elasticity interpretation with seasonal controls
    * **Cross-Price Analysis**: Comprehensive substitute/complement relationship modeling within customer categories
    * **Statistical Rigor**: P-value filtering (p < 0.05) and model diagnostics for robust elasticity estimates
* **Business Intelligence:** Delivered actionable pricing insights with quantified demand sensitivity:
    * **Segment-Specific Elasticity**: Champions demonstrate lower price sensitivity than Hibernating customers
    * **Category Intelligence**: Fresh seafood exhibits premium pricing tolerance vs. processed foods
    * **Product Dynamics**: Identified high-elasticity products (e.g., SP01 Skinless Chicken Breast: -2.61 elasticity in Supermarkets)
    * **Strategic Positioning**: Competitive interaction matrices informing portfolio pricing decisions

### **Phase 3: Mathematical Revenue Optimization & Strategic Implementation**
* **Strategic Objective:** Maximize revenue through mathematically optimized, segment-specific pricing strategies under realistic business constraints
* **Technical Implementation:**
    * **Gurobi Linear Programming**: Industrial-strength optimization engine for complex constraint handling
    * **Multi-Constraint Framework**: Price bounds (-50% to +200%), category consistency (±5%), and margin protection
    * **Cross-Price Integration**: Sophisticated modeling of substitute/complement effects in optimization objective
    * **Scenario Analysis**: Comprehensive sensitivity testing and trade-off evaluation
* **Strategic Deliverables:**
    * **Revenue Maximization**: Mathematical formulation incorporating elasticity-driven demand response
    * **Implementation Roadmap**: Phased rollout strategy with risk mitigation and success metrics
    * **Segment-Specific Strategies**: Tailored pricing approaches aligned with customer behavior and market position
    * **Business Impact Projection**: **SGD ~4M annual revenue uplift** with primary gains from Supermarket segment optimization

## Strategic Business Impact & Commercial Transformation

This data-driven commercial strategy transformation delivered quantifiable business value and strategic competitive advantages:

### **Financial Impact & Revenue Growth**
* **Primary Revenue Uplift**: **SGD ~4M annual revenue increase** through optimized pricing strategies
* **Margin Enhancement**: 2-8% improvement in profit margins through scientific price positioning
* **Portfolio Optimization**: Revenue concentration analysis revealing high-impact product categories
* **Competitive Positioning**: Strategic pricing relative to market dynamics and customer willingness-to-pay

### **Customer Intelligence & Segmentation Value**
* **Behavioral Segmentation**: Five actionable customer segments with distinct commercial strategies:
    * **Champions** (0.3%): Premium customers requiring retention-focused strategies
    * **Potential Loyalists** (0.1%): High-opportunity customers for expansion programs  
    * **New Customers** (80%): Largest segment requiring acquisition and onboarding optimization
    * **Hibernating** (11%): At-risk customers needing proactive re-engagement
    * **Lost** (9%): Inactive customers targeted for win-back campaigns
* **Lifetime Value Optimization**: CLV-based prioritization for marketing and sales resource allocation
* **Market Penetration Strategy**: Segment-specific approaches for category expansion and customer development

### **Operational Excellence & Data Capabilities**
* **Advanced Analytics Infrastructure**: Production-ready pipeline supporting real-time pricing decisions
* **Statistical Rigor**: Comprehensive model validation ensuring robust business recommendations  
* **Strategic Decision Framework**: Data-informed commercial processes replacing intuition-based pricing
* **Competitive Intelligence**: Market dynamics modeling for proactive strategy development
* **Implementation Excellence**: Phased rollout minimizing business disruption while maximizing impact

## Enterprise Technology Stack & Implementation Architecture

This project leveraged enterprise-grade data science technologies and methodologies for production-ready commercial analytics:

### **Core Analytics Engine**
* **Data Processing**: Python ecosystem with Pandas/NumPy for high-performance data manipulation
* **Statistical Computing**: Advanced econometric modeling with comprehensive validation frameworks
* **Machine Learning**: Scikit-learn clustering algorithms with robust preprocessing and validation
* **Mathematical Optimization**: Gurobi industrial solver for complex constrained optimization problems

### **Customer Intelligence Pipeline**
* **RFM Analytics Framework**: Four-dimensional behavioral analysis (Recency, Frequency, Monetary, Total Quantity)
* **Advanced Clustering**: Two-stage K-means with `RobustScaler` outlier handling and silhouette optimization
* **Behavioral Segmentation**: Rule-based classification with statistical validation and business logic integration
* **Interactive Visualization**: Professional dashboards with Matplotlib, Seaborn, and Squarify treemap analytics

### **Econometric Modeling & Optimization**
* **Statistical Modeling**: StatsModels OLS regression with log-log specification for direct elasticity interpretation
* **Revenue Optimization**: Gurobi linear programming with multi-constraint business rule integration
* **Real-time Analytics**: Streamlit web application enabling interactive scenario analysis and strategy simulation

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
