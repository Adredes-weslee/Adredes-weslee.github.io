---
layout: project
title: "HDB Resale Price Predictor"
description: "A Singapore housing estimator built by narrowing a research model into practical user inputs."
date: 2023-06-18
categories: [machine-learning, regression, real-estate, public-policy, urban-planning]
image: /assets/images/project-covers/hdb-resale-prices.jpg
technologies: [Python, Streamlit, pandas, scikit-learn, Regression modeling, Feature engineering, Model artifacts]
github: https://github.com/Adredes-weslee/Making-Predictions-on-HDB-Resale-Price
blog_post: /data-science/machine-learning/real-estate/2023/06/18/predicting-hdb-resale-prices.html
streamlit_app: https://adredes-weslee-making-predictions-on-hdb-resale-pric-app-iwznz4.streamlit.app/
---

## Business context

Housing-price models are easy to over-engineer for accuracy and hard to turn into a usable product. This project focused on Singapore HDB resale pricing, with the main challenge being how to keep predictive quality high while limiting inputs to factors a real user can provide.

## Outcome

- Production model reports roughly 0.9261 R^2 with RMSE around 39,180 SGD.
- Refactored a research notebook with 150+ engineered features into a deployable Streamlit application.
- Built a consistent training and inference pipeline around a reduced, user-facing feature set.
- Framed the product around Singapore-specific housing decisions rather than generic regression output.

## Key decisions

- Explicitly prioritized usability over maximum predictive accuracy.
- Reduced the feature space to variables normal users can actually enter.
- Kept preprocessing shared between training and inference so the deployed app matched the modeling pipeline.
- Chose a simpler, more explainable production path over a larger academic feature set.

## System design

Historical HDB resale records are cleaned, transformed, and passed through a modular preprocessing and regression pipeline. The trained artifacts are then reused inside the Streamlit app so feature handling and predictions remain consistent between model development and live inference.

## Stack

- Python, pandas, scikit-learn, and joblib artifacts
- Streamlit for the user-facing estimator
- Regression modeling, feature engineering, and Singapore housing domain logic
