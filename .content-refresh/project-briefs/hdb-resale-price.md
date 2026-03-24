# HDB Resale Price Predictor

## Project Thesis

A Singapore housing price prediction app that deliberately trades a small amount of academic complexity for user-facing practicality.

## Business Problem

Homebuyers need a prediction tool they can actually use, not a research notebook that depends on inputs they would never know. The real challenge is choosing the feature set that preserves enough signal while remaining realistic for end users.

## Outcome and Evidence

- Production model reports roughly 0.9261 R² and RMSE around 39,180 SGD.
- Refactors a research notebook with 150+ features into a deployable Streamlit application.
- Builds a consistent training and inference pipeline around the reduced feature set.
- Centers the user experience around Singapore-specific real-estate decision factors.

## Key Decision Choices

- Explicitly prioritized usability over maximum predictive accuracy.
- Reduced the feature space to variables normal users can provide.
- Kept the pipeline modular so training and inference share preprocessing logic.
- Chose a simpler production model and interpretation path over a more academic feature set.

## Tech Stack

- Python
- Streamlit
- pandas
- scikit-learn
- joblib model artifacts
- YAML / JSON config surfaces

## Architecture Snapshot

The app layer handles page views and reusable UI components, while the `src` layer manages data loading, feature engineering, model prediction, utilities, and visualization. Trained models are stored as reusable artifacts for app inference.

## Portfolio Content Angle

Lead with the product decision: “usability over perfect accuracy.” That makes the project more mature than a standard predictive-model demo.

## Evidence Gaps / Refresh Notes

- Later content pass should highlight the specific user-input tradeoffs you made.
- This project is well suited to a portfolio narrative about turning research into a usable product.

