---
layout: project
title: "Dengue Forecasting and Intervention Economics"
description: "Forecasting dengue risk and comparing intervention economics in one policy-facing decision-support system."
date: 2025-06-18
categories: [epidemiological-forecasting, public-health, health-economics, policy-analytics, time-series]
image: /assets/images/project-covers/dengue-forecasting.jpg
technologies: [Python, Prophet, Streamlit, Time series analysis, Health economics, DALY modeling, Decision dashboards]
github: https://github.com/Adredes-weslee/Dengue-Case-Prediction-and-Cost-Benefits-Analysis
blog_post: /epidemiology/forecasting/health-economics/2025/06/18/forecasting-dengue-cases-and-cost-benefit-analysis.html
streamlit_app: https://adredes-weslee-dengue-case-prediction-and-c-dashboardapp-aszwww.streamlit.app/
---

## Business context

Public-health teams do not just need a forecast. They need to know how far ahead risk can be seen, which interventions are worth funding, and how those choices trade off under real budget pressure. This project was built to connect those questions in one decision surface.

## Outcome

- Built a 16-week forecasting workflow designed around operational planning windows.
- Reported 9.5% MAPE with the selected Prophet production model.
- Estimated Wolbachia at about $60,039 per DALY versus roughly $360,876 per DALY for vaccination in the modeled scenarios.
- Combined epidemiological forecasting and cost-benefit analysis in a Streamlit dashboard rather than splitting them across separate analyses.

## Key decisions

- Chose a 16-week forecast horizon because it is long enough to influence intervention planning.
- Combined surveillance, weather, search-trend, and demographic signals instead of treating dengue as a single-series problem.
- Evaluated multiple time-series baselines but foregrounded Prophet as the most practical production path.
- Integrated health economics into the same product so decision-makers could compare forecasted need and intervention value together.

## System design

Multiple data sources are cleaned into a master time series, passed through a forecasting pipeline, and paired with a separate economic analysis module. The dashboard presents forecast views, intervention comparisons, and policy-facing summaries in one interface.

## Stack

- Python, Prophet, and supporting time-series tooling
- Streamlit for the decision-support interface
- Custom ETL, forecasting, DALY modeling, and intervention comparison modules
