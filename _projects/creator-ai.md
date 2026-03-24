---
layout: project
title: "Creator AI Platform"
description: "Backend-first AI content generation platform with staged orchestration, deterministic quality gates, and human review."
date: 2026-03-24
categories: [ai-platforms, edtech, orchestration, evaluation-systems, workflow-governance]
image: /assets/images/project-covers/creator-ai.jpg
technologies: [Python, FastAPI, PostgreSQL, Redis, Milvus, OpenTelemetry, Azure, TypeScript]
repo_private: true
repo_note: "Source code is private. The public case study and article focus on workflow design, platform architecture, and evaluation/governance choices."
blog_post: /ai-platforms/edtech/orchestration/2026/03/24/designing-creator-ai-as-a-backend-first-content-generation-platform.html
---

## Business context

Generating courses, labs, quizzes, and capstones with LLMs is easy to prototype and hard to operationalize. To be useful in a real education product, the workflow needs structured intake, retrieval and standards grounding, prompt governance, validation, review queues, export paths, and strong observability around every stage.

## Outcome

- Built a multi-service platform that covers gate, discovery, strategy, retrieval, context, generation, validation, build, feedback, and export stages.
- Added contract-first APIs, OpenAPI snapshots, schema export, and a large deterministic verification matrix.
- Implemented explicit run and task state machines, durable artifacts, replay tooling, and review workflows.
- Supported simulation-mode testing so the system can be verified without depending on live model calls.

## Key decisions

- Made the product backend-first and API-first instead of building a frontend-led copilot demo.
- Treated Discovery lock as an integrity boundary before downstream stages can run.
- Used shared contracts and OpenAPI snapshots to keep services, frontend, and verification aligned.
- Built evaluation, review, and governance into the platform rather than treating them as post-processing.

## System design

Requests enter through a BFF facade, then move through gate and discovery before optional persona and strategy stages. Standards and retrieval feed a deterministic context bundle, which drives generation, validation, build, and export workflows. The orchestrator is the system of record for runs, tasks, events, and artifacts, while the frontend stays thin and consumes platform state rather than owning workflow logic.

## Stack

- Python 3.11, FastAPI, SQLAlchemy, Alembic, Redis, PostgreSQL
- Milvus, MinIO, JSON Schema, OpenAPI, Spectral
- OpenTelemetry, Azure Monitor, LangSmith, Azure Key Vault
- TypeScript contract packages and a BFF/web layer
