# Dengue Case Prediction and Cost-Benefits Analysis Platform

## Project Thesis

A public-health decision support system that pairs outbreak forecasting with intervention economics for Singapore dengue planning.

## Business Problem

Health agencies do not just need forecasts; they need to know which intervention options are worth funding and deploying. This project addresses both operational prediction and policy-level cost-effectiveness.

## Outcome and Evidence

- Prophet model reports 9.5% MAPE on 16-week forecasts.
- Cost-benefit analysis estimates Wolbachia at $60,039 per DALY versus $360,876 per DALY for vaccination.
- Connects epidemiological forecasting with economic impact estimates across a 10-year intervention horizon.
- Builds a usable Streamlit interface rather than leaving the work in notebook form.

## Key Decision Choices

- Chose a 16-week forecast horizon because it is operationally meaningful for intervention planning.
- Combined surveillance, weather, search trend, and demographic signals.
- Evaluated multiple time-series baselines but foregrounded Prophet as the chosen production model.
- Integrated economic analysis into the same product surface so decision-makers can compare forecasted need and intervention value together.

## Tech Stack

- Python, Prophet, scikit-time ecosystem
- Streamlit dashboard
- Custom ETL and modeling modules
- Cost-benefit analysis and DALY modeling

## Architecture Snapshot

Data sources are cleaned and merged into a master time series, then passed into a forecasting pipeline and a separate economic analysis module. Outputs are served through dashboard pages for prediction and intervention comparison.

## Portfolio Content Angle

This should read as a policy intelligence system, not just a forecasting model. The strongest story is the combination of predictive and economic decision support.

## Evidence Gaps / Refresh Notes

- Later content pass should verify whether the portfolio should foreground forecast accuracy, cost-effectiveness, or both equally.
- Strong candidate for an outcome-led portfolio section because the repo already has hard numbers.

