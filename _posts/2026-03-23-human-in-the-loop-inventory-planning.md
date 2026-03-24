---
layout: post
title: "Designing Human-in-the-Loop Inventory Planning"
date: 2026-03-23 09:00:00 +0800
categories: [ai-ops, forecasting, operations]
tags: [fastapi, streamlit, inventory-planning, prophet, xgboost, gmroi, approvals, n8n, docker, observability]
author: Wes Lee
feature_image: /assets/images/article-heroes/2026-03-23-human-in-the-loop-inventory-planning.jpg
---

## Introduction: Inventory systems fail when they optimize for automation instead of trust

It is easy to describe an inventory-planning product as "forecast demand, recommend an order quantity, then automate purchasing." It is much harder to build something that a real operator would actually trust.

This project takes the second problem seriously. The system combines forecasting, EOQ and reorder-point logic, GMROI guardrails, approval flows, and audit logging inside a workflow that is explicitly human-in-the-loop. That choice shapes the entire architecture. The interesting part is not that it can emit a recommendation. The interesting part is that it treats recommendation quality, operational constraints, and reviewability as one product problem.

> Related: For the concise case-study version, see the [Agentive Inventory Management System project page](/projects/agentive-inventory/).

## Why the recommendation engine is only one layer of the system

The backend does forecast generation and procurement logic, but the product is broader than that. The repo is split into:

- a FastAPI backend for forecasts, backtests, configs, procurement logic, approvals, and health surfaces
- a Streamlit UI that acts as the operator-facing collaboration hub
- config files for service levels, lead times, costs, and approval thresholds
- n8n workflows for recurring planning runs and approval loops

That structure is the right one for an operations system. Inventory planning is not a single inference call. It is a repeated cycle:

1. read demand history
2. forecast likely near-term demand
3. translate forecast into reorder guidance
4. check guardrails such as service level and GMROI
5. send the result to a human who can approve or reject it
6. log that decision so the system remains auditable

The repo reflects that lifecycle instead of pretending the product is just "AI orders inventory."

## Forecasting is useful here only because it feeds a policy layer

The forecasting service supports multiple strategies, including simpler baselines and heavier models such as Prophet and XGBoost, but the output is not meant to stand alone. Forecasts are passed into procurement logic that computes reorder points, order quantities, confidence, and whether the recommendation should require approval.

That is the right design choice. Inventory teams do not buy forecasts. They buy decisions under constraints.

The project therefore pairs model output with business rules from configuration:

- service level targets
- lead times
- carrying cost assumptions
- setup cost assumptions
- auto-approval limits
- minimum service level
- GMROI thresholds

Those are not implementation footnotes. They are what turn a forecast into a planning product. They also make the system much easier to explain to a non-ML stakeholder because the recommendation can be traced back to parameters the business actually recognizes.

## Human review is not a fallback. It is the point

One of the most important design choices in this repo is that approvals and audit logging are first-class surfaces. There are dedicated endpoints and UI pages for recommendations, approvals, and the audit log because the system is explicitly designed around review, not silent automation.

That becomes even clearer in the explanation pathway. Gemini-based rationales are optional, and the explain endpoint is allowed to return `404` when LLM explanations are disabled by configuration. That is a subtle but important signal. The core planning workflow still functions without an external model. The LLM layer is there to help humans interpret a recommendation, not to make the recommendation engine inseparable from a hosted model vendor.

This design buys a few practical benefits:

- the core decision system still works when an LLM is unavailable
- explanations can be added without making them the source of truth
- approval history stays anchored in structured policy and logged actions
- operators can review batch recommendations under optional cash-budget constraints

That is much closer to how enterprise decision-support systems usually need to behave.

## Operational reality shaped the repo as much as the planning logic

The README is unusually explicit about deployment constraints, especially memory pressure on Render's free tier. That is useful because it shows the repo is grounded in actual execution constraints rather than a best-case local machine.

The system includes:

- CI for ruff, mypy, pytest, and coverage
- optional API auth and rate limiting
- structured JSON logs
- Prometheus-style `/metrics`
- Docker Compose for local orchestration
- Render deployment configuration
- persistent-disk guidance for model cache and audit artifacts

That is the right level of seriousness for an ops-facing AI system. If a planning tool cannot be tested, rate-limited, observed, and deployed repeatably, the fancy recommendation logic will not matter for long.

The same realism shows up in the UI design. The frontend is not just a dashboard. It has specific pages for forecasts, recommendations, settings, backtests, and audit logs. That means the product is trying to support the operator's workflow, not just visualize model output.

## n8n makes the system feel more like a workflow product than a model demo

The orchestration layer is a meaningful part of the architecture here. Daily scheduling and approval-loop flows move the system from "tool an analyst clicks when they remember" toward "workflow that can participate in regular operations."

This matters because many AI demos stall at the moment they need recurrence or handoff. By including n8n workflow examples directly in the repo, the project acknowledges that inventory planning is a repeated operational process with external notifications and approval steps, not just an interactive UI.

That orchestration layer also complements the human-in-the-loop design. Recommendations can be generated on a schedule, surfaced for review, and then recorded through approval endpoints instead of being treated as one-off interactions.

## The main lesson: decision support beats forced autonomy

The best part of this repo is that it refuses to confuse automation with quality. The architecture is deliberately shaped around review, guardrails, and auditability:

- forecasting feeds policy rather than replacing it
- LLM explanations are optional, not foundational
- approvals are explicit and logged
- orchestration exists to support repeated planning cycles
- deployability and observability are treated as part of the product

That is what makes this a better portfolio piece than a generic demand-forecasting demo. The project does not just ask "can a model predict demand?" It asks "how should a forecasting system behave when someone has to defend the resulting purchasing decision?"
