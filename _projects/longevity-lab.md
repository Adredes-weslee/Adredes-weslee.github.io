---
layout: project
title: "Longevity Lab Health Scenario Platform"
description: "A public-health risk communication platform with artifact-backed scoring, evidence-status surfaces, model cards, and separate causal-analysis workflows."
date: 2026-05-02
categories: [health-ai, public-health, decision-support, model-evaluation, risk-communication]
image: /assets/images/project-covers/longevity-lab.jpg
technologies: [FastAPI, React, TypeScript, DuckDB, scikit-learn, XGBoost, Playwright, Render, Vercel]
github: https://github.com/Adredes-weslee/longevity-lab
blog_post: /health-ai/public-health/model-evaluation/2026/05/02/building-longevity-lab-for-health-risk-scenario-modeling.html
streamlit_app: https://longevity-lab-frontend.onrender.com/
---

## Business context

Health-risk interfaces can become misleading when they compress data provenance, predictive scoring, model quality, and causal interpretation into one confident-looking number. Longevity Lab addresses that by treating risk communication as an evidence product: users should see the scenario, the active runtime mode, the data sources, the model-card context, and the caveats around prediction versus causation.

## Outcome

- Built a local-first FastAPI backend and React frontend with Explorer, Data Evidence, Model Cards, and Scenario Lab surfaces.
- Published a public Render demo for the React interface; first load can take about a minute while the service wakes and fetches sample data.
- Added public-data pipelines for BRFSS, EPA AirData, ACS, SVI, and CDC PLACES-oriented context, with schema and provenance documentation.
- Added an evidence-status API and Data Evidence UI that separate active scoring inputs from local assets, validation reports, production artifact readiness, and inactive geography context.
- Implemented artifact-backed scoring paths for eight BRFSS-derived conditions, benchmark harnesses, subgroup metrics, model-card manifests, typed explanations, and uncertainty metadata.
- Kept causal analysis in a separate workbench so predictive risk scores are not presented as causal or diagnostic output.

## Key decisions

- Separated scenario comparison, data evidence, model-card review, and causal analysis into distinct user-facing surfaces.
- Kept the repo local-first and artifact-aware so large public datasets and trained bundles are not committed into source control.
- Used model-card and provenance surfaces to make model behavior inspectable rather than hiding it behind a polished dashboard.
- Added deployment guidance for demo/sample-artifact modes without implying that the public surface is a clinical decision system.

## System design

Source downloaders and feature builders create public-health tables and derived context features. Training and benchmark scripts produce calibrated model artifacts, metrics, subgroup slices, explanation metadata, pollutant ablations, and model-card manifests. The FastAPI API exposes health, metadata, scenario, evidence, pipeline, and model-card contracts, while the React frontend turns those contracts into scenario comparison, data evidence, model-card, and scenario-lab views.

## Stack

- FastAPI, Pydantic, DuckDB, pandas, scikit-learn, optional XGBoost, Hydra, and Optuna
- React, Vite, TypeScript, D3 utilities, and Playwright
- Public health data pipelines for BRFSS, EPA AirData, ACS, SVI, and CDC PLACES context
- Render/Vercel deployment profile with `real-20260504-full` release-artifact checksum verification
