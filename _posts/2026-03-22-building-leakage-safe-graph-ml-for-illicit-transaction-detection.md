---
layout: post
title: "Building Leakage-Safe Graph ML for Illicit Transaction Detection"
date: 2026-03-22 09:00:00 +0800
categories: [graph-ml, fintech, fraud-detection]
tags: [pytorch-geometric, xgboost, logistic-regression, graphsage, gat, calibration, precision-at-k, temporal-splits, robustness]
author: Wes Lee
feature_image: /assets/images/article-heroes/2026-03-22-building-leakage-safe-graph-ml-for-illicit-transaction-detection.jpg
---

## Introduction: Fraud modeling gets overstated when evaluation is weak

Graph neural networks are attractive in financial-crime detection because illicit behavior often spreads through relationships rather than isolated records. But graph projects are also unusually easy to oversell. A model can look strong offline while leaking future information, optimizing the wrong metric, or performing badly once investigators have limited review capacity.

This repo is interesting because it is built around those failure modes. It compares feature-only baselines and GNNs on the Elliptic dataset, but the real point is not simply "GNNs can do better." The point is to build a detection workflow that stays credible under temporal constraints, operating thresholds, calibration, and robustness checks.

> Related: For the shorter case-study summary, see the [Elliptic Graph ML project page](/projects/elliptic-gnn-project/).

## The first decision was methodological: stop leakage before tuning models

The repo starts with temporal splits and explicit leakage checks rather than jumping straight into graph architecture. That is the right priority for transaction data.

In this setting, a random split can dramatically overstate quality because neighboring nodes and adjacent timesteps leak structure across train and test. The project therefore:

- builds a processed graph from the raw Elliptic CSVs
- keeps temporal ordering explicit
- checks for cross-timestep edge leakage
- evaluates drift by time instead of reporting one undifferentiated score

That makes the later model comparisons much more believable. A fancy graph model only matters if the evaluation regime resembles the chronology of actual detection work.

## Baselines came first for a reason

This repo does not assume graph models deserve the win by default. Logistic regression and XGBoost are trained first, then compared against GCN, GraphSAGE, and GAT variants.

That is good practice for two reasons:

- it forces the project to prove graph value instead of assuming it
- it keeps simpler models in the conversation as operational baselines

In many detection contexts, a well-tuned tree model plus sensible thresholding can already be useful. The repo treats that seriously, then asks whether graph structure improves enough to justify additional complexity.

The strongest reported direction in the experiment set is a SAGE-ResBN configuration with small sinusoidal time embeddings and limited train windows. Even there, the repo does not stop at a headline score. It keeps comparing models through calibration, workload curves, and robustness checks.

## The metric choice is more important than the model choice

One of the best decisions in this project is that it does not rely on ROC-AUC alone. The analysis pipeline emphasizes metrics that match how investigation teams actually work:

- PR-AUC for the illicit class
- Precision@K for limited investigation budgets
- Recall at a chosen precision target
- expected calibration error
- PR-AUC by timestep to see whether quality drifts across time

That is the right way to frame an ops-facing detection system. If investigators can only review the top 100 or top 500 alerts, then workload curves and Precision@K are more useful than abstract threshold-free metrics. If scores are going to drive escalation, calibration matters because overconfident probabilities create bad downstream decisions even when ranking is decent.

This is what makes the repo feel more operational than many graph-ML projects. It is not just asking which model is strongest in aggregate. It is asking what model behavior looks like under actual review constraints.

## The analysis layer is where the repo becomes most credible

The project includes dedicated scripts for:

- by-time evaluation
- calibration plots
- workload curves
- paired bootstrap comparisons
- hub-removal ablation
- robustness under edge drop and feature noise
- explainability for both XGBoost and GNN runs
- optional ensembling across recommended configurations

That analysis surface is more important than it looks. Fraud and compliance models often fail when the graph shifts, when highly connected hubs dominate behavior, or when the model looks stable only at one operating threshold. By adding these checks directly into the repo, the project turns model evaluation into a repeatable pipeline rather than an afterthought.

The hub-ablation and robustness pieces are especially useful because they ask whether the model is learning durable patterns or just leaning too hard on a fragile part of the graph. That is the kind of question that matters if a detection model is going to face adversarial or evolving behavior.

## The dashboard is there to inspect experiments, not to hide them

The Streamlit app does not try to disguise the project as a productized fraud console. Instead, it serves as an experiment-review surface:

- overview metrics
- by-time drift
- calibration curves
- workload curves
- comparison artifacts
- downloadable configs and plots

That is the right fit for the repo. The dashboard is not pretending to be the final fraud-operations interface. It is there to help a reviewer understand how the experiment set behaves, which model is stronger under which criteria, and where the tradeoffs sit.

This also matches the broader lesson from the project: a detection system becomes more trustworthy when its evaluation artifacts are inspectable, not just summarized.

## The main lesson: defensible graph ML is mostly an evaluation discipline

The project page summarizes this as leakage-safe graph ML with robustness checks. The deeper takeaway is that the repo treats graph modeling as only one layer of the problem.

What makes the work strong is the combination of:

- temporal discipline before modeling
- baseline comparisons before GNN claims
- operating metrics that reflect investigation budgets
- calibration and workload analysis, not just aggregate ranking
- robustness and ablation checks to test whether the model is leaning on brittle structure

That is the posture worth keeping from this project. In financial-risk settings, the best graph model is not automatically the one with the flashiest architecture. It is the one that still looks credible after the evaluation protocol gets stricter.
