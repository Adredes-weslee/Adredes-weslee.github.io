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

This project developed a **production-ready machine learning application** to predict Housing & Development Board (HDB) flat resale prices in Singapore. The project evolved from exploratory data analysis to a complete Streamlit web application featuring interactive data exploration, real-time price predictions, and model performance insights. Using a comprehensive dataset of over 60,000 transactions, the system achieves an R² of ~0.90 with RMSE of ~45,000 SGD, providing valuable tools for homebuyers, policymakers, and urban planners.

**Key Features:**
- **Interactive Web Application**: Full-featured Streamlit dashboard with 4 main sections
- **Real-time Price Prediction**: User-friendly form accepting essential property details
- **Data Explorer**: Interactive visualizations with filtering and trend analysis  
- **Model Insights**: Performance comparison and feature importance analysis
- **Production Pipeline**: Sklearn-based preprocessing and model training pipelines

<div class="demo-link-container">
  <a href="https://adredes-weslee-making-predictions-on-hdb-resale-pric-app-iwznz4.streamlit.app/" class="demo-button" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-home"></i> Try the HDB Price Predictor
  </a>
</div>

## The Significance: Understanding Singapore's Housing Backbone

In Singapore, over 80% of the resident population lives in HDB flats, making public housing a cornerstone of the nation's social fabric and economy. The HDB resale market is dynamic and deeply impacts citizens' financial well-being, investment choices, and housing affordability. Accurate price prediction and a clear understanding of value drivers are crucial for:

-   **Informed Homebuyers:** Empowering individuals to make sound financial decisions when buying or selling flats.
-   **Effective Policymaking:** Providing data-driven insights for housing policies, subsidies, and urban development.
-   **Market Transparency:** Increasing clarity and reducing information asymmetry in the resale market.
-   **Urban Planning:** Guiding the development of amenities and infrastructure to support sustainable and equitable housing.

This project sought to answer critical questions such as: What are the most significant factors influencing HDB resale prices? How does the diminishing lease affect property valuation? What is the premium associated with location and amenities?

## Methodology: Production-Ready Data Science Application

The project demonstrates a complete evolution from research to production deployment:

### 1. **Application Architecture & User Experience**
- **4-Page Streamlit Application**:
  - **Home**: Project overview with dataset statistics and interactive demo callout
  - **Data Explorer**: Interactive visualizations with filtering by town, flat type, and time period
  - **Make Prediction**: User-friendly form for real-time price predictions
  - **Model Performance**: Detailed comparison of 3 regression models with feature importance analysis

### 2. **Data Processing & Feature Engineering**
- **Dual Pipeline Approach**:
  - **Exploratory pipeline**: For data visualization and analysis (`train_processed_exploratory.csv`)
  - **Production pipeline**: Streamlined for consistent training/inference (`train_pipeline_processed.csv`)
- **Core Features** (Simplified from 150+ to 12 essential user-providable features):
  - Location: `town`, `mrt_nearest_distance`, `Mall_Nearest_Distance`
  - Property: `flat_type`, `flat_model`, `floor_area_sqm`, `storey_range`
  - Infrastructure: `lease_commence_date`, `max_floor_lvl`, amenity distances
  - Derived: `hdb_age`, temporal features, remaining lease calculations

### 3. **Model Development & Deployment**
- **Three Production Models**:
  - **Linear Regression**: Baseline interpretable model
  - **Ridge Regression**: L2 regularization for stability (primary model)
  - **Lasso Regression**: L1 regularization with feature selection
- **Sklearn Pipeline Integration**: Complete preprocessing + model pipelines saved as `.pkl` files
- **Configuration-Driven**: YAML configs for hyperparameters and feature selection
- **Model Artifacts**: JSON files storing metrics, feature names, and model metadata

### 4. **User Interface & Interaction Design**
- **Smart Input Collection**: Only essential features users can realistically provide
- **Real-time Validation**: Input range checking and error handling
- **Model Selection**: Users can choose between Linear, Ridge, or Lasso predictions
- **Rich Feedback**: Prediction confidence, model accuracy metrics, and price per sqm calculations

## Key Findings & Production Application Insights

The deployed Streamlit application reveals critical factors influencing HDB resale prices through both data exploration and model analysis:

### 1. **Interactive Data Insights**
- **Advanced Filtering System**: Multi-dimensional filtering by town, flat type, and time period with real-time data updates
- **Price Distribution Analysis**: Interactive histograms with statistical overlays (mean, median, quartiles) and filtering capabilities
- **Temporal Trends**: Dynamic time series charts with monthly/yearly aggregation and trend line analysis
- **Geographic Variations**: Town-based price comparisons with statistical significance testing and location premium analysis
- **Feature Correlations**: Customizable correlation heatmaps with user-selectable features and statistical significance indicators
- **Raw Data Explorer**: Expandable view with sortable/filterable data tables for detailed transaction analysis

### 2. **Model-Driven Price Factors** (From Feature Importance Analysis)
- **Location Dominance**: Town categories consistently rank as top predictors, with central areas commanding 20-30% premiums
- **Physical Attributes**: Floor area and storey level show strong linear relationships with price
- **Lease Impact**: HDB age and remaining lease duration demonstrate significant non-linear price effects
- **Accessibility Premium**: MRT and mall distances show measurable impact on resale values

### 3. **Production Model Performance**
- **Ridge Regression** (Primary Model):
  - Training R²: 0.898, Testing R²: 0.899 (excellent generalization)
  - RMSE: ~45,400 SGD (±6.8% typical error margin)
- **Model Comparison Dashboard**: Side-by-side performance metrics with interactive visualizations
- **Feature Selection Analysis**: Lasso model identifies ~60% feature sparsity, highlighting most critical predictors
- **Cross-Validation Robustness**: Consistent performance across different data splits ensuring model reliability

### 4. **User Experience & Application Features**
- **Simplified Input Requirements**: Reduced from 150+ research features to 12 essential user-providable attributes
- **Real-time Prediction Engine**: Sub-second response times with immediate price estimates and confidence intervals
- **Multi-Model Comparison**: Users can compare predictions across Linear, Ridge, and Lasso algorithms with explanation of differences
- **Smart Input Validation**: Range checking, error handling, and user-friendly feedback for invalid inputs
- **Model Transparency**: Feature importance visualization and coefficient analysis helping users understand prediction factors
- **Production-Grade Caching**: Optimized model loading and data processing with Streamlit caching for improved performance
- **Practical Validation**: Price per square meter calculations provide intuitive sanity checks and market context

## Technical Implementation & Production Architecture

The application demonstrates professional software engineering practices suitable for real-world deployment:

### **Frontend & User Experience**
- **Streamlit Framework**: Multi-page application with responsive design, custom CSS styling, and optimized caching
- **Interactive Components**: Dynamic sliders, multi-select boxes, date pickers, and forms with real-time validation and feedback
- **Advanced Visualization Suite**: Plotly-based interactive charts including:
  - Histograms with statistical overlays and distribution fitting
  - Time series plots with trend analysis and seasonal decomposition
  - Box plots and violin plots for categorical feature analysis
  - Correlation heatmaps with customizable feature selection
  - Feature importance charts with coefficient analysis
- **Error Handling & Fallbacks**: Graceful degradation for missing data, model loading failures, and invalid user inputs
- **Performance Optimization**: Strategic caching of models, data loading, and visualization rendering

### **Backend & Data Processing**
- **Modular Architecture**: Clear separation between `app/`, `src/`, `models/`, and `configs/` directories
- **Pipeline-First Design**: Sklearn pipelines ensuring preprocessing consistency between training and prediction
- **Configuration Management**: YAML-based configs for model hyperparameters and application settings
- **Caching Strategy**: Strategic use of Streamlit caching for model loading and data processing

### **Model Management & Deployment**
```python
# Production pipeline structure
Pipeline([
    ('preprocessor', ColumnTransformer([
        ('numerical', StandardScaler(), numerical_features),
        ('categorical', OneHotEncoder(drop='first'), categorical_features)
    ])),
    ('model', Ridge(alpha=1.0))
])
```

### **Core Application Features**
1. **Data Explorer**: 
   - **Advanced Filtering**: Multi-dimensional filtering by town (5+ defaults), flat type, and date ranges
   - **Statistical Analysis**: Real-time calculation of price statistics (mean, median, min, max, std dev) with dynamic updates
   - **Interactive Visualizations**: 4-tab interface covering distribution analysis, temporal trends, categorical comparisons, and correlation analysis
   - **Raw Data Access**: Expandable view with first 100 transactions for detailed examination

2. **Price Prediction Interface**:
   - **Intelligent Input Collection**: User-friendly forms collecting only essential property details with smart defaults
   - **Multi-Model Prediction**: Real-time comparison across Linear, Ridge, and Lasso regression models
   - **Confidence & Context**: Prediction intervals, model accuracy metrics, and price per square meter calculations
   - **Input Validation**: Range checking for realistic values with helpful error messages and suggestions

3. **Model Performance Dashboard**:
   - **Comprehensive Metrics**: Side-by-side comparison of R², RMSE, MAE, and other performance indicators
   - **Feature Importance Analysis**: Interactive visualization of feature coefficients and their impact on predictions
   - **Model Methodology**: Detailed explanations of algorithm differences and validation approaches
   - **Production Insights**: Real-world performance metrics and lessons learned from deployment

## Policy & Societal Implications

The insights derived from this HDB resale price prediction model have several important implications:

1.  **Enhanced Market Transparency:** Provides homebuyers and sellers with a data-driven tool for assessing fair market value.
2.  **Informed Policymaking on Lease Decay:** Offers quantitative evidence on the financial impact of diminishing leases, informing discussions on policies like VERS (Voluntary Early Redevelopment Scheme) and lease buyback options.
3.  **Targeted Housing Subsidies:** Allows for more nuanced design of housing grants and subsidies, potentially differentiating based on lease status or location-specific price pressures.
4.  **Data-Driven Urban Planning:** Insights into amenity valuation can guide future development plans, ensuring equitable distribution of facilities and infrastructure.
5.  **Financial Planning for Citizens:** Helps Singaporeans better understand the long-term value trajectory of their primary asset, aiding in retirement and financial planning.
6.  **Market Stability Monitoring:** Predictive models can serve as an early warning system for potential market anomalies or unsustainable price trends.

## Future Work & Potential Enhancements

Building on this production-ready foundation, several opportunities exist for further development:

**Data & Features:**
-   **Enhanced Geospatial Analysis:** Integrate precise coordinates and GIS data for micro-location analysis and distance calculations to multiple amenities
-   **Real-time Market Data:** Connect to live property listings and recent transactions for dynamic market trend analysis
-   **Additional Property Attributes:** Include renovation status, unit facing direction, specific floor level, and interior condition assessments

**Model Sophistication:**
-   **Time Series Forecasting:** Implement LSTM or Prophet models for explicit temporal prediction and market trend forecasting
-   **Ensemble Methods:** Combine multiple algorithms (Random Forest, XGBoost, Neural Networks) for improved accuracy
-   **Neighborhood-Specific Models:** Develop localized models capturing unique micro-market dynamics within HDB towns

**Application Features:**
-   **Market Analysis Dashboard:** Add comparative market analysis tools and investment return calculators
-   **User Personalization:** Save favorite properties, custom alerts, and personalized recommendations
-   **Mobile Optimization:** Responsive design improvements and potential native mobile app development
-   **API Integration:** Connect with property portals, government databases, and financial institutions for expanded functionality

**Advanced Analytics:**
-   **Macroeconomic Integration:** Include interest rates, GDP growth, and policy changes as model features
-   **Market Prediction:** Develop early warning systems for market bubbles or correction phases
-   **Portfolio Analysis:** Multi-property analysis tools for investors and property developers

## Conclusion

This HDB resale price prediction project successfully demonstrates the complete journey from data science research to production-ready application deployment. By combining sophisticated machine learning techniques with user-centered design principles, the resulting Streamlit application provides a powerful, accessible tool for Singapore's housing market stakeholders.

**Technical Achievement:** The production system achieves excellent predictive performance (R² ~0.90, RMSE ~45,400 SGD) while maintaining model interpretability and user accessibility. The multi-model approach provides transparency in prediction methodology and allows users to understand the trade-offs between different algorithmic approaches.

**Practical Impact:** Beyond technical metrics, this project delivers genuine value through its comprehensive feature set - from interactive data exploration that reveals market insights to real-time predictions that assist in property valuation decisions. The application successfully bridges the gap between complex machine learning models and practical user needs.

**Production Excellence:** The implementation demonstrates professional software engineering practices including modular architecture, comprehensive error handling, performance optimization through caching, and user experience design. The system is designed for real-world deployment with proper model management, configuration-driven setup, and scalable infrastructure patterns.

**Strategic Value:** The insights into price drivers—particularly the quantified impact of location, lease decay, and property attributes—provide significant value for individual decision-making, public policy formulation, and the pursuit of a more transparent and efficient housing market in Singapore. The application serves as both a practical tool and a demonstration of how data science can address real societal challenges.

This project establishes a strong foundation for continued development and showcases the potential for data-driven solutions in Singapore's dynamic property market.

---

*For a detailed technical walkthrough of the feature engineering and modeling process, please refer to the [accompanying blog post](/data-science/machine-learning/real-estate/2023/06/18/predicting-hdb-resale-prices.html). The full codebase is available on [GitHub](https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price).*
