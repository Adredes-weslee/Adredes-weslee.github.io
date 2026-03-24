---
layout: project
title: "Agentive Inventory Management System"
description: "Human-in-the-loop inventory planning with forecasts, procurement guardrails, and auditable approvals."
date: 2026-03-22
categories: [ai-ops, forecasting, inventory-optimization, decision-support, operations]
image: /assets/images/project-covers/agentive-inventory.jpg
technologies: [FastAPI, Streamlit, Python, Prophet, XGBoost, EOQ and ROP logic, n8n, Docker Compose, Observability]
github: https://github.com/Adredes-weslee/agentive-inventory
blog_post: /ai-ops/forecasting/operations/2026/03/23/human-in-the-loop-inventory-planning.html
---

## Business context

Inventory teams need reorder guidance that balances service level, stockout risk, cash use, and category economics. The harder problem is trust: recommendations need to be reviewable, not just automated, especially when real purchasing decisions are involved.

## Outcome

- Ships a working FastAPI backend, Streamlit collaboration UI, and n8n orchestration layer.
- Supports EOQ and reorder-point logic, GMROI guardrails, approvals, audit logging, and batch recommendation flows.
- Includes production-oriented features such as auth, rate limiting, structured logs, `/metrics`, Docker Compose, CI, and Render deployment config.
- Uses the M5 dataset as a realistic planning backbone for forecast-driven inventory decisions.

## Key decisions

- Chose a human-in-the-loop workflow instead of direct auto-ordering.
- Combined forecast outputs with procurement guardrails so recommendations remain operationally usable.
- Made LLM rationales optional so the system still functions without Gemini or another hosted model.
- Added audit and observability surfaces so approval decisions remain reviewable.

## System design

Demand history feeds forecasting and inventory services, which expose recommendation APIs for reorder planning. The Streamlit UI handles review, approvals, and settings management, while n8n orchestrates recurring runs and configuration-driven planning workflows.

## Stack

- FastAPI, Pydantic, pytest, mypy, and ruff on the backend
- Streamlit for the collaboration interface
- Prophet, XGBoost, EOQ and ROP logic, n8n orchestration, and Docker-based deployment
