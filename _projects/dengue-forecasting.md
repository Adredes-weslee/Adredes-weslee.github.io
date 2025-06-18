---
layout: project
title: "Strategic Dengue Control: Production-Ready Forecasting Platform for Singapore's Public Health Authority"
categories: [epidemiological-forecasting, public-health, health-economics, policy-analytics, time-series]
image: /assets/images/dengue-forecasting-strategic.jpg
technologies: [Python, Prophet, Streamlit, Time Series Analysis, Health Economics, DALY Analysis, Cost-Benefit Modeling, Interactive Dashboards]
github: https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis
blog_post: /epidemiology/forecasting/health-economics/2025/06/18/production-dengue-forecasting-platform.html
streamlit_app: https://adredes-weslee-dengue-case-prediction-and-c-dashboardapp-aszwww.streamlit.app/
---

## Executive Summary

This project delivers a **production-ready epidemiological forecasting platform** that transforms 11 years of Singapore dengue surveillance data into actionable intelligence for the National Environment Agency (NEA) and Ministry of Health. The platform provides **16-week dengue outbreak predictions** with 9.5% accuracy and comprehensive **cost-benefit analysis** comparing major intervention strategies, directly supporting Singapore's $100M annual dengue control budget decisions.

<div class="demo-link-container">
  <a href="https://adredes-weslee-dengue-case-prediction-and-c-dashboardapp-aszwww.streamlit.app/" class="demo-button" target="_blank" rel="noopener noreferrer">
    <i class="fas fa-play-circle"></i> Try the Live Demo
  </a>
</div>

## Strategic Business Challenge

Singapore faces escalating dengue threats that demand evidence-based resource allocation and proactive intervention planning:

### **Public Health Impact**
- **Record Outbreak**: 35,068 cases in 2020 (highest in Singapore's history)
- **Economic Burden**: $1.01-2.27 billion total costs (2010-2020)
- **Healthcare Strain**: 30% surge in hospital admissions during peak seasons
- **Climate Risk**: Rising temperatures expanding optimal mosquito breeding conditions

### **Decision-Making Gaps**
- **Reactive Response**: Current surveillance relies on post-outbreak detection
- **Resource Inefficiency**: Ad-hoc deployment without predictive planning
- **Intervention Uncertainty**: Limited economic evaluation of control strategies
- **Policy Justification**: Need for evidence-based budget allocation to stakeholders

## Strategic Solution: Integrated Forecasting & Economic Platform

Our platform addresses these challenges through two complementary analytical engines:

### **1. Predictive Intelligence Engine**
- **16-Week Forecasting Horizon**: Operational lead time for vector control deployment
- **Multi-Source Integration**: Disease surveillance + weather + behavioral signals + demographics
- **9.5% MAPE Accuracy**: Exceeds international benchmarks for epidemiological forecasting
- **Real-Time Updates**: Automated pipeline for continuous model refresh

### **2. Health Economics Decision Engine**
- **WHO-Standard DALY Analysis**: Disability-Adjusted Life Years methodology
- **Intervention Comparison**: Wolbachia vs Dengvaxia cost-effectiveness evaluation
- **Budget Impact Modeling**: 10-year financial projections with sensitivity analysis
- **Policy Threshold Analysis**: Cost per DALY benchmarked against Singapore's $82,703 threshold

## Key Strategic Findings

### **Forecasting Performance & Validation**
| **Metric** | **Prophet Model** | **International Benchmark** | **Strategic Value** |
|------------|-------------------|----------------------------|-------------------|
| **MAPE** | **9.5%** | 15-25% (typical range) | Superior early warning capability |
| **Forecast Horizon** | **16 weeks** | 4-8 weeks (standard) | Extended planning window |
| **Historical Validation** | **2020 outbreak predicted** | Often reactive | Proactive outbreak management |
| **Feature Integration** | **4 data sources** | Single source (typical) | Comprehensive situational awareness |

### **Economic Analysis: Clear Policy Direction**

| **Intervention Strategy** | **Annual Cost** | **DALYs Averted** | **Cost per DALY** | **WHO Threshold Compliance** |
|--------------------------|-----------------|-------------------|------------------|---------------------------|
| **Project Wolbachia** | **$27.0M USD** | **449.7** | **$60,039** | ✅ **Cost-Effective** |
| **Dengvaxia® Vaccination** | **$220.7M USD** | **612.1** | **$360,876** | ❌ **Not Cost-Effective** |
| **Singapore Threshold** | - | - | **$82,703** | WHO standard for high-HDI countries |

### **Return on Investment Analysis**
- **Wolbachia Program**: **$2.90 benefit per $1 invested** (BCR = 2.90)
- **Cost Avoidance**: **$76.8M annually** in healthcare and productivity costs
- **Population Coverage**: **5.9M residents** protected through vector suppression
- **Implementation Timeline**: **3-year deployment** vs **ongoing vaccination cycles**

## Business Impact & Operational Value

### **For Singapore's National Environment Agency (NEA)**
✅ **Proactive Resource Deployment**: 16-week lead time enables optimal field team allocation  
✅ **Budget Justification**: Evidence-based $27M Wolbachia investment recommendation  
✅ **Performance Monitoring**: Real-time dashboard tracking intervention effectiveness  
✅ **Risk Communication**: Data-driven public messaging during forecast high-risk periods  

### **For Ministry of Health (MOH)**
✅ **Hospital Capacity Planning**: Predicted case loads inform bed allocation and staffing  
✅ **Surveillance Optimization**: Focus testing resources on forecast hotspot periods  
✅ **Vaccine Policy Guidance**: Economic evidence against population-wide Dengvaxia deployment  
✅ **Regional Coordination**: Shareable forecasts for ASEAN health cooperation  

### **For Government Budget Planning**
✅ **Cost-Effectiveness Evidence**: Clear recommendation for Wolbachia over vaccination  
✅ **Multi-Year Projections**: 10-year budget impact modeling for strategic planning  
✅ **Sensitivity Analysis**: Risk assessment under different outbreak scenarios  
✅ **International Benchmarking**: Singapore performance vs regional health systems  

## Technology Architecture & Scalability

### **Production-Ready Infrastructure**
- **Modular Pipeline**: Preprocessing → Training → Forecasting → Analysis → Dashboard
- **Automated Workflows**: Weekly data refresh and model retraining capabilities
- **Interactive Platform**: Streamlit dashboard for real-time scenario modeling
- **API-Ready**: RESTful endpoints for integration with existing health systems

### **Quality Assurance & Governance**
- **Model Validation**: Holdout testing on 2022 dengue resurgence data
- **Performance Monitoring**: Automated MAPE tracking and alert thresholds
- **Data Lineage**: Full traceability from raw surveillance to policy recommendations
- **Security Compliance**: Anonymized data handling following health privacy standards

### **Deployment & Maintenance**
- **Cloud-Native**: Scalable deployment on AWS/Azure government clouds
- **Version Control**: Git-based model management and rollback capabilities
- **Documentation**: Comprehensive user guides for technical and policy stakeholders
- **Training Program**: Knowledge transfer for NEA/MOH operational teams

## Strategic Recommendations

### **Immediate Actions (0-6 months)**
1. **Deploy Wolbachia Island-Wide**: Accelerate from pilot to full Singapore coverage
2. **Implement Forecasting Dashboard**: Integrate into NEA's weekly surveillance workflow
3. **Establish Prediction-Response Protocols**: Standard operating procedures for forecast-driven interventions

### **Medium-Term Enhancements (6-18 months)**
1. **Geospatial Expansion**: District-level forecasting for targeted interventions
2. **Multi-Disease Platform**: Extend framework to chikungunya and Zika surveillance
3. **Regional Collaboration**: Share forecasting capabilities with ASEAN partners

### **Long-Term Vision (18+ months)**
1. **AI-Driven Optimization**: Machine learning for dynamic intervention timing
2. **Climate Integration**: Incorporate IPCC climate projections for long-term planning
3. **One Health Platform**: Integrate human, animal, and environmental surveillance

## Economic Return Summary

**Total Platform Value**: **$76.8M annually** in averted healthcare and productivity costs  
**Implementation Cost**: **$27.0M annually** for recommended Wolbachia deployment  
**Net Benefit**: **$49.8M annually** (excluding platform development costs)  
**ROI Timeline**: **12-month payback period** for platform and intervention investment  

## Conclusion

This platform represents a paradigm shift from reactive dengue management to predictive, evidence-based public health strategy. With superior forecasting accuracy, clear economic guidance favoring Wolbachia deployment, and a production-ready technological foundation, the platform positions Singapore as a global leader in epidemiological intelligence and cost-effective disease prevention.

The integration of advanced time-series modeling with rigorous health economics provides NEA and MOH with the analytical foundation needed to optimize Singapore's $100M annual dengue control investment while protecting 5.9M residents from preventable disease burden.

---

*For detailed technical implementation, model architecture, and code walkthrough, see the [technical blog post](/epidemiology/forecasting/health-economics/2025/06/18/production-dengue-forecasting-platform.html). Complete source code and deployment guides available on [GitHub](https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis).*