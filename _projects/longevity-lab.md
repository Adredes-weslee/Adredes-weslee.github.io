---
layout: project
title: "Longevity Lab Health Scenario Platform"
description: "A public-health risk communication platform with artifact-backed scoring, model cards, provenance surfaces, and separate causal-analysis workflows."
date: 2026-05-02
categories: [health-ai, public-health, decision-support, model-evaluation, risk-communication]
image: /assets/images/project-covers/longevity-lab.jpg
technologies: [FastAPI, React, TypeScript, DuckDB, scikit-learn, XGBoost, Playwright, Render, Vercel]
github: https://github.com/Adredes-weslee/longevity-lab
blog_post: /health-ai/public-health/model-evaluation/2026/05/02/building-longevity-lab-for-health-risk-scenario-modeling.html
---

## Business context

Health-risk interfaces can become misleading when they compress data provenance, predictive scoring, model quality, and causal interpretation into one confident-looking number. Longevity Lab addresses that by treating risk communication as an evidence product: users should see the scenario, the active runtime mode, the data sources, the model-card context, and the caveats around prediction versus causation.

## Outcome

- Built a local-first FastAPI backend and React frontend with Explorer, Data Evidence, Model Cards, and Scenario Lab surfaces.
- Added public-data pipelines for BRFSS, EPA AirData, ACS, SVI, and CDC PLACES-oriented context, with schema and provenance documentation.
- Implemented artifact-backed scoring paths, benchmark harnesses, subgroup metrics, model-card manifests, typed explanations, and uncertainty metadata.
- Kept causal analysis in a separate workbench so predictive risk scores are not presented as causal or diagnostic output.

## Key decisions

- Separated scenario comparison, data evidence, model-card review, and causal analysis into distinct user-facing surfaces.
- Kept the repo local-first and artifact-aware so large public datasets and trained bundles are not committed into source control.
- Used model-card and provenance surfaces to make model behavior inspectable rather than hiding it behind a polished dashboard.
- Added deployment guidance for demo/sample-artifact modes without implying that the public surface is a clinical decision system.

## System design

Source downloaders and feature builders create public-health tables and derived context features. Training and benchmark scripts produce calibrated model artifacts, metrics, subgroup slices, explanation metadata, and model-card manifests. The FastAPI API exposes health, metadata, scenario, pipeline, and model-card contracts, while the React frontend turns those contracts into scenario comparison, data evidence, model-card, and scenario-lab views.

## Stack

- FastAPI, Pydantic, DuckDB, pandas, scikit-learn, optional XGBoost, Hydra, and Optuna
- React, Vite, TypeScript, D3 utilities, and Playwright
- Public health data pipelines for BRFSS, EPA AirData, ACS, SVI, and CDC PLACES context
- Render/Vercel deployment profile with release-artifact checksum verification
