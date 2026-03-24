# Agentive Inventory Management System

## Project Thesis

An agentic inventory planning system that turns historical demand data into forecast-driven procurement recommendations while keeping a human approver in the loop.

## Business Problem

Inventory teams need reorder guidance that balances service level, stockout risk, cash use, and category economics. Fully automated recommendations can be hard to trust, so the system is designed to support approval workflows rather than opaque automation.

## Outcome and Evidence

- Ships a working FastAPI backend, Streamlit collaboration UI, and n8n orchestration layer.
- Supports EOQ/ROP procurement logic, GMROI guardrails, approvals, audit logging, and batch recommendation flows.
- Includes production-oriented features such as auth, rate limiting, structured logs, `/metrics`, Docker Compose, CI, and Render deployment config.

## Key Decision Choices

- Chose a human-in-the-loop workflow instead of direct auto-ordering.
- Used the M5 Walmart demand dataset as a realistic canonical planning dataset.
- Combined forecast outputs with procurement guardrails instead of relying on forecasts alone.
- Made LLM rationales optional so recommendations still work without Gemini.
- Added audit and observability surfaces to make approval decisions reviewable.

## Tech Stack

- Backend: FastAPI, Pydantic, pytest, mypy, ruff
- Frontend: Streamlit
- Forecasting and planning: SMA, Prophet, XGBoost, EOQ/ROP logic
- Orchestration: n8n
- Ops: Docker Compose, Prometheus-style metrics, Render deployment
- Data: M5 Forecasting dataset

## Architecture Snapshot

Demand history feeds forecasting and inventory services, which expose recommendation APIs. The Streamlit UI handles review, approval, and settings management. n8n automates recurring runs, while config files define thresholds and business context.

## Portfolio Content Angle

Frame this as an AI operations system, not just a forecasting app: agentic planning, decision support, approvals, and governance.

## Evidence Gaps / Refresh Notes

- Later content pass should verify which forecast model performs best in practice.
- Good candidate for a portfolio angle around “trustworthy AI recommendations for operations teams.”

