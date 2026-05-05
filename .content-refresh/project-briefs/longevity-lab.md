# Longevity Lab

## Project Thesis

Longevity Lab is a public-health risk communication and scenario modeling platform that turns public datasets, calibrated model artifacts, and evidence-status surfaces into an inspectable React + FastAPI product prototype.

## Business Problem

Health-risk tools often collapse several different jobs into one opaque score: public-data provenance, individual scenario comparison, predictive modeling, model limitations, and causal interpretation. Longevity Lab separates those surfaces so a user can inspect active scoring inputs, local data assets, validation reports, artifact readiness, scenario deltas, and outputs that should not be treated as diagnosis or causal advice.

## Outcome and Evidence

- Ships a local-first FastAPI backend and React/Vite frontend with Explorer, Data Evidence, Model Cards, and Scenario Lab pages.
- Uses public BRFSS, EPA AirData, ACS, SVI, and CDC PLACES oriented pipeline code with provenance and feature-contract documentation.
- Adds an evidence-status endpoint and React Data Evidence UI that distinguish active scoring inputs from validation-only reports and inactive geography context.
- Includes artifact-backed scoring paths for eight BRFSS-derived conditions, model-card endpoints, calibrated tree and gradient-boosting benchmark harnesses, pollutant ablations, subgroup metrics, explanation records, and uncertainty metadata.
- Keeps causal analysis in a separate workbench with explicit question contracts and sensitivity framing rather than blending causal claims into predictive risk scores.
- Provides CI for backend, frontend, and Playwright E2E checks plus Render/Vercel deployment guidance and release-artifact download verification.

## Key Decision Choices

- Kept prediction, provenance, model-card review, and causal workbench surfaces separate.
- Chose a local-first artifact strategy so large public datasets and trained bundles are not committed into the repo.
- Built the frontend around scenario comparison and evidence transparency rather than a single dashboard score.
- Used model cards, subgroup metrics, calibration, and source manifests as first-class product evidence.

## Tech Stack

- FastAPI, Pydantic, DuckDB, pandas, scikit-learn, optional XGBoost, Hydra/Optuna
- React, Vite, TypeScript, D3 utilities, Playwright
- Public health data pipelines for BRFSS, EPA AirData, ACS, SVI, and PLACES
- Render/Vercel deployment profile with artifact download verification

## Architecture Snapshot

The repo separates source downloaders, feature builders, benchmark/evaluation pipelines, artifact loading, API contracts, and frontend pages. The API can serve demo or artifact-backed scores, while the frontend makes runtime mode, data readiness, evidence status, model-card metadata, and scenario deltas visible to the user.

## Portfolio Content Angle

Position this as a decision-support and evidence-transparency system, not as a medical tool. The strongest angle is the separation between predictive risk communication, model evidence, data provenance, and causal-analysis boundaries.

## Evidence Gaps / Refresh Notes

- Public demo is registered at https://longevity-lab-frontend.onrender.com/ and may need a cold-start wait on first load.
- If visuals are refreshed further, prioritize screenshots of the Explorer body map, Model Cards, Data Evidence, and Scenario Lab pages over the abstract hero asset.
