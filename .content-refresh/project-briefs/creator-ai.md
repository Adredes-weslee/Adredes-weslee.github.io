# Creator AI

## Project Thesis

A backend-first AI content-generation platform for learning assets, built around staged orchestration, deterministic quality gates, human review, and production-grade governance.

## Business Problem

Generating courses, labs, quizzes, and capstones with LLMs is easy to demo but hard to operationalize. Real deployment requires reproducible prompts, locked workflow stages, retrieval and standards grounding, validation, review queues, export paths, observability, and cost/governance controls. The product challenge is not just generation quality, but how to build a trustworthy system around it.

## Outcome and Evidence

- Implements a monorepo platform with contract-first APIs, shared schemas, OpenAPI snapshots, and service-level development commands.
- Covers a staged workflow across gate, discovery, strategy, standards, retrieval, context, generation, validation, build, feedback, skills, and export services.
- Uses deterministic verification gates, prompt-to-proof matrices, dataset builders, eval harnesses, and replay tooling to make behavior auditable.
- Includes production-shaped infrastructure concerns such as state machines, outbox patterns, auth, observability, Azure deployment notes, retention, and tenant isolation.
- Provides a thin frontend/BFF layer while keeping orchestration, persistence, and policy enforcement in the backend.

## Key Decision Choices

- Chose an API-first, backend-first model rather than a frontend-led copilot prototype.
- Made Discovery lock a formal integrity boundary before downstream generation can proceed.
- Used contract-first schemas and OpenAPI snapshots to keep frontend, services, and verification aligned.
- Built deterministic simulation and verifier paths so the platform can be tested without relying on live model calls.
- Treated retrieval corpora, standards grounding, prompt registry, evaluation, and review as first-class system concerns rather than add-ons.

## Tech Stack

- Python 3.11, FastAPI, SQLAlchemy, Alembic, Redis, Postgres
- OpenTelemetry, Azure Monitor, LangSmith, Azure Key Vault, Azure identity tooling
- Milvus, MinIO, JSON Schema, OpenAPI, Spectral
- TypeScript packages for contracts and web integration
- Azure-focused infrastructure and deployment documentation

## Architecture Snapshot

Creator AI is a monorepo of coordinated services with the orchestrator as the system of record. Requests enter through a BFF facade, pass through gate and discovery, then move into optional persona and strategy stages before standards grounding, retrieval, context assembly, generation, validation, build, feedback, and export. Runs and tasks follow explicit state machines, and artifacts, events, prompts, and eval outputs are versioned and replayable. The frontend stays thin while the backend owns workflow integrity, persistence, and governance.

## Portfolio Content Angle

This should be framed as platform engineering for AI-native learning systems: orchestration, workflow governance, evaluation infrastructure, and production controls for educational content generation.

## Evidence Gaps / Refresh Notes

- Repo is private and large, so the public writeup should focus on architecture and product systems rather than exhaustive feature inventory.
- Strong candidate for a flagship private-case-study slot because it aligns directly with current work and backend-first AI platform positioning.
- Later content pass could separate “what the platform does” from “what engineering disciplines make it safe and evaluable.”
