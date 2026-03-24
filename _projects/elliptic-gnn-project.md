---
layout: project
title: "Elliptic Graph ML for Illicit Transaction Detection"
description: "Leakage-safe graph ML for illicit transaction detection with operational metrics and robustness checks."
date: 2026-03-24
categories: [graph-ml, gnn, fraud-detection, fintech, model-evaluation]
image: /assets/images/project-covers/elliptic-gnn-project.jpg
technologies: [Python, PyTorch Geometric, XGBoost, Logistic regression, TensorBoard, Streamlit, Calibration analysis]
repo_private: true
repo_note: "Source code is private. The public case study and article focus on the methodology, evaluation design, and operational tradeoffs."
blog_post: /graph-ml/fintech/fraud-detection/2026/03/22/building-leakage-safe-graph-ml-for-illicit-transaction-detection.html
---

## Business context

Fraud and compliance teams need to identify risky transaction flows early, but graph-based financial datasets are easy to evaluate incorrectly because of time leakage and unrealistic thresholds. This project focused on building a defensible detection workflow, not just chasing a strong offline score.

## Outcome

- Compares feature-only baselines against GCN, GraphSAGE, and GAT models.
- Uses temporal splits, calibration, robustness checks, Precision@K, PR-AUC by timestep, and workload curves.
- Documents SAGE-ResBN configurations as the strongest-performing setup in the repo's experiments.
- Includes interpretability and ensemble-analysis paths for both baseline and graph models.

## Key decisions

- Prioritized leakage-safe temporal splits before model tuning.
- Benchmarked simpler baselines before claiming graph-model gains.
- Used precision-at-investigation-budget style metrics instead of relying only on ROC-AUC.
- Added calibration, robustness, and hub-ablation checks to test operational stability.

## System design

Raw Elliptic CSVs are transformed into a processed graph artifact, then passed through baseline and GNN training pipelines. Analysis modules handle by-time drift, calibration, workload curves, bootstrap comparison, robustness checks, explanations, and optional ensembling.

## Stack

- Python, PyTorch Geometric, XGBoost, and logistic-regression baselines
- TensorBoard and Streamlit for experiment presentation
- Graph preprocessing, temporal evaluation, calibration, and explainability tooling
