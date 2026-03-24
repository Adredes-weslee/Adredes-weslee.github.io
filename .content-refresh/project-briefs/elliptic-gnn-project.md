# Elliptic GNN Project

## Project Thesis

A graph machine learning system for illicit cryptocurrency transaction detection, built with leakage-safe temporal evaluation and operations-oriented metrics.

## Business Problem

Fraud and compliance teams need to identify risky transaction flows early, but graph-based financial datasets are easy to evaluate incorrectly due to time leakage and unrealistic thresholds. This project aims to build a defensible detection workflow, not just a high offline score.

## Outcome and Evidence

- Compares feature-only baselines against GCN, GraphSAGE, and GAT models.
- Uses temporal splits, calibration, robustness checks, Precision@K, PR-AUC by timestep, and workload curves.
- Documents recommended SAGE-ResBN configurations as the best-performing settings in experiments.
- Includes interpretability and ensemble analysis surfaces.

## Key Decision Choices

- Prioritized leakage-safe temporal splits before model tuning.
- Benchmarked simpler baselines before claiming graph-model gains.
- Used precision-at-investigation-budget style metrics instead of relying only on ROC-AUC.
- Added calibration, robustness, and hub-ablation checks to test operational stability.
- Built explainability paths for both XGBoost and GNN models.

## Tech Stack

- Python
- PyTorch Geometric
- XGBoost
- Logistic Regression baselines
- TensorBoard
- Streamlit app for presentation

## Architecture Snapshot

Raw Elliptic CSVs are converted into a processed graph artifact, then passed through baseline and GNN training pipelines. Analysis modules handle by-time drift, calibration, workload, bootstrap comparison, robustness, explanation, and ensembling.

## Portfolio Content Angle

Frame this as a trustworthy graph ML project for financial crime detection, with special attention to evaluation rigor and deployment-relevant metrics.

## Evidence Gaps / Refresh Notes

- Later content pass should extract exact headline metrics from run artifacts in `outputs/`.
- Strong candidate for a “what makes model evaluation trustworthy” angle on the site.

