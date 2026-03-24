# Advanced Retail Price Optimization System

## Project Thesis

A revenue optimization platform that combines segmentation, elasticity modeling, and constrained mathematical optimization to produce pricing recommendations.

## Business Problem

Retail pricing is not a one-model problem. Teams need to understand customer segments, estimate price sensitivity, and optimize under business constraints such as price bounds and category consistency.

## Outcome and Evidence

- Implements a four-stage analytics pipeline from raw transactions to optimization outputs.
- Uses RFM segmentation, elasticity modeling, and Gurobi optimization.
- Exposes results through a Streamlit dashboard and a price strategy simulator.
- Builds in constraints to keep recommended price changes realistic.

## Key Decision Choices

- Separated segmentation, elasticity estimation, and optimization into distinct modules.
- Used RFM plus clustering to capture customer heterogeneity.
- Modeled own-price and cross-price elasticity before optimization.
- Added practical optimization constraints like price bounds and category movement limits.

## Tech Stack

- Python 3.11
- Streamlit
- scikit-learn
- Gurobi
- OLS econometric modeling
- Data preprocessing and analytics utilities

## Architecture Snapshot

Transaction data is cleaned and engineered, passed into customer segmentation, then into elasticity modeling. The resulting coefficients feed a Gurobi-based optimization layer, and the outputs are presented in dashboard pages for strategy analysis and simulation.

## Portfolio Content Angle

Present this as decision optimization, not just descriptive analytics. The strongest public-facing story is the handoff from customer behavior to price elasticity to recommended action.

## Evidence Gaps / Refresh Notes

- Later content pass should surface one example of an optimized pricing outcome or revenue lift scenario.
- This project is suited to a case-study format with “from raw transactions to constrained pricing action.”

