---
layout: project
title: "Retail Price Optimization for CS Tay"
description: "Segmentation, elasticity modeling, and constrained optimization for retail pricing decisions."
date: 2024-08-15
categories: [data-science, business-analytics, pricing-strategy, commercial-strategy, capstone]
image: /assets/images/project-covers/customer-segmentation.jpg
technologies: [Python, Streamlit, scikit-learn, StatsModels, Gurobi, RFM analysis, K-means clustering, Price elasticity modeling]
github: https://github.com/Adredes-weslee/price-optimization
blog_post: /data-science/pricing-strategy/business-analytics/commercial-strategy/2024/08/15/customer-segmentation-price-optimization.html
streamlit_app: https://adredes-weslee-price-optimization-streamlitapp-yxjoe3.streamlit.app/
---

## Business context

CS Tay needed a pricing strategy that went beyond cost-plus rules. The problem was not just finding better prices. It was connecting customer behavior, price sensitivity, and commercial guardrails into one decision flow that a business team could actually use.

## Outcome

- Built a four-stage analytics pipeline from transaction cleanup to pricing recommendations.
- Combined RFM segmentation, elasticity modeling, and Gurobi optimization inside one workflow.
- Modeled a revenue-upside scenario of roughly SGD 4M annually from segment-specific pricing moves.
- Shipped the work as an interactive demo instead of leaving it as a slide deck or notebook.

## Key decisions

- Separated segmentation, elasticity estimation, and optimization so each layer could be explained and validated on its own.
- Used RFM plus clustering to capture customer heterogeneity before pricing.
- Modeled own-price and cross-price elasticity before optimization, so recommendations reflected demand response instead of static margin logic.
- Added practical constraints like price bounds and category movement limits to keep outputs realistic.

## System design

Transaction data is cleaned and aggregated, passed into customer segmentation, then into econometric price modeling. The resulting elasticity signals feed a constrained optimization layer that recommends price moves and exposes scenario analysis through the dashboard.

## Stack

- Python, pandas, scikit-learn, StatsModels, and Gurobi
- Streamlit for the decision-support interface
- RFM analysis, clustering, elasticity modeling, and constrained optimization
