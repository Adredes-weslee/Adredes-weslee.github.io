# Singapore Wet Bulb Temperature Analysis Platform

## Project Thesis

A climate analysis platform that studies Singapore heat stress using wet-bulb temperature as the core livability metric, linking local weather and global climate signals.

## Business Problem

Air temperature alone does not capture how dangerous heat becomes in humid tropical environments. For climate researchers and public policy stakeholders, wet-bulb temperature is a more decision-relevant signal for human heat risk.

## Outcome and Evidence

- Merges seven climate and emissions datasets into a unified analysis dataset.
- Converts an academic notebook into a modular Streamlit platform with ETL, modeling, and visualization layers.
- Covers more than 40 years of climate and greenhouse gas context for Singapore-focused analysis.
- Produces publication-style visualizations and regression surfaces for exploration.

## Key Decision Choices

- Framed the project around wet-bulb temperature instead of generic temperature trends.
- Combined local meteorological variables with global greenhouse gas indicators.
- Broke the notebook into reusable modules for data loading, feature engineering, modeling, and visualization.
- Treated the project as both scientific communication and analysis, not just prediction.

## Tech Stack

- Python, pandas, scikit-learn, matplotlib
- Streamlit dashboard
- Multi-source ETL and feature engineering modules
- Statistical utilities and regression models

## Architecture Snapshot

Raw climate and emissions data feed a preprocessing pipeline that produces an analysis-ready dataset. Feature engineering and regression modules support dashboard pages for time series, correlation, and model exploration.

## Portfolio Content Angle

Lead with “human heat risk in tropical cities” rather than “climate dashboard.” The value is in choosing the right metric and making the research explorable.

## Evidence Gaps / Refresh Notes

- Later content pass should pull out the strongest substantive finding from the regression and correlation outputs.
- Visual assets likely need refresh because the story is stronger than the current imagery.

