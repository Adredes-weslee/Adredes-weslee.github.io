---
layout: project
title: "Workforce Risk Intelligence Prototype"
description: "Incident-centric workforce intelligence built from governed public signals, retrenchment forecasting, and operator dashboards."
date: 2026-03-24
categories: [ai-platforms, workforce-intelligence, decision-support, forecasting, risk-intelligence]
image: /assets/images/project-covers/workforce-risk-intelligence.jpg
technologies: [Next.js, FastAPI, PostgreSQL, Langfuse, Playwright, Docker Compose, Recharts]
repo_private: true
repo_note: "Source code is private. The public case study and article focus on architecture, evidence handling, and operational design choices."
blog_post: /ai-ops/workforce-intelligence/public-signals/2026/03/24/building-operational-workforce-risk-intelligence-from-public-signals.html
---

## Business context

Workforce teams need early warning on layoffs, retrenchment risk, and labor-market instability, but the signal arrives through fragmented public sources with different reliability levels and update cadences. The challenge is not collecting more raw data. It is turning noisy evidence into a reviewable incident pipeline and a usable decision surface.

## Outcome

- Built a full-stack prototype with operator dashboards, incident views, and company-level monitoring.
- Implemented a DB-backed pipeline that ingests, normalizes, screens, clusters, scores, and alerts on workforce-risk signals.
- Added retrenchment forecasting and early-warning views for industry and company monitoring.
- Integrated Langfuse tracing, replay artifacts, and deterministic evaluation paths for pipeline review.

## Key decisions

- Shifted the system from raw scraped rows to documents, incidents, and company risk snapshots.
- Used same-origin Next.js API proxies so the frontend stays behind a controlled BFF boundary.
- Kept LLMs focused on grounded summaries and hypothesis support instead of row-level classification.
- Added auth gates, bounded concurrency, SSRF guardrails, and timeout controls before treating the prototype as operator-ready.

## System design

Public-source connectors feed normalized documents into a Postgres-backed ingestion and scoring pipeline. Those documents are screened for workforce relevance, deduplicated into incidents, aggregated into company and industry snapshots, and surfaced through dashboards, alert routes, and retrenchment forecast views. Observability and replay artifacts make both user-facing requests and pipeline runs inspectable.

## Stack

- Next.js App Router, React, TypeScript, Recharts
- FastAPI, SQLAlchemy, Alembic, PostgreSQL
- Playwright scraping, RSS, Reddit, Yahoo Finance, governed browser sources
- Langfuse observability and Docker Compose deployment
