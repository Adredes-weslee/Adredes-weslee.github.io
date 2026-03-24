# NTUC Workforce Intelligence Prototype

## Project Thesis

A workforce-risk intelligence platform that turns noisy public signals into incident-centric company monitoring, retrenchment forecasts, and operator-facing dashboards.

## Business Problem

Workforce and policy teams need early warning on layoffs, retrenchment risk, and labor-market instability, but public signals are fragmented across news, forums, company pages, finance feeds, and official indicators. A useful system needs durable ingestion, evidence normalization, scoring, alerting, and human-review workflows rather than one-off scraping or summarization.

## Outcome and Evidence

- Delivers a full-stack prototype with a Next.js frontend and FastAPI backend.
- Implements a DB-backed incident pipeline: ingest, normalize, screen, deduplicate, cluster, score, and alert.
- Adds retrenchment forecasting and early-warning workflows, including industry dashboards, company prediction endpoints, and replay/debug artifacts.
- Supports Langfuse observability, experiment harnesses, and deterministic replay/evaluation paths.
- Includes runtime hardening such as auth gates, bounded concurrency, admin-key controls, timeout handling, SSRF guardrails, and 5xx sanitization.

## Key Decision Choices

- Shifted the unit of analysis from raw scraped rows to documents, incidents, and company risk snapshots.
- Kept LLM usage focused on grounded summaries and hypothesis generation rather than row-level filtering.
- Used a same-origin Next.js BFF boundary so the frontend never calls the backend directly.
- Added deterministic scoring, anomaly detection, and replay tooling before attempting more ambitious supervised modeling.
- Chose a VM-friendly architecture with Docker Compose, Postgres, raw artifact storage, and controlled source expansion.

## Tech Stack

- Next.js App Router, React, TypeScript, Tailwind, Recharts
- FastAPI, Python, SQLAlchemy, Alembic, Postgres
- Playwright scraping, Google News RSS, Reddit, Yahoo Finance, browser scraping
- Langfuse observability and experiment tooling
- Docker Compose and GCE-oriented deployment docs

## Architecture Snapshot

The frontend operates behind a login-gated Next.js shell and uses same-origin API proxy routes. The FastAPI backend handles scraping, hypothesis generation, incident ingestion, retrenchment pipelines, summaries, and dashboards. Documents are normalized into Postgres-backed entities, clustered into incidents, scored into risk snapshots, and surfaced through company, alert, and retrenchment APIs. Langfuse traces and replay artifacts make both user-facing and pipeline behavior inspectable.

## Portfolio Content Angle

This should be presented as an operator-facing AI intelligence system: multi-source ingestion, incident modeling, forecast-driven dashboards, and evaluation-aware product engineering for workforce monitoring.

## Evidence Gaps / Refresh Notes

- Repo is private, so the public writeup should emphasize architecture, evaluation discipline, and product framing rather than code-level implementation specifics.
- Good candidate for a case-study angle around evidence-grounded risk intelligence and public-signal governance.
- If visual assets are refreshed later, prioritize dashboard screenshots, incident pipeline diagrams, and retrenchment forecast surfaces.
