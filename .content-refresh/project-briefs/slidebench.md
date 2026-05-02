# ArtifactBench

## Project Thesis

An artifact evaluation system for AI-generated decks, curricula, project documents, PDFs, and code bundles that combines deterministic metrics, benchmark retrieval, provenance checks, and LLM-based judging behind a durable evaluation workflow.

## Business Problem

Teams increasingly generate learning assets, slide decks, project artifacts, and supporting documents with AI, but most review remains subjective and hard to reproduce. A usable benchmarking product needs structured extraction, repeatable metrics, benchmark comparison, evaluation reports, provenance guardrails, and resumable background jobs rather than a one-shot judge prompt.

## Outcome and Evidence

- Delivers a FastAPI backend plus Vite/React frontend for uploading, indexing, evaluating, and reviewing artifacts.
- Supports multiple artifact types: PPTX, curriculum markdown/JSON, project PDFs, and optional project-code zip inputs.
- Runs a modular evaluation pipeline: extract, deterministic metrics, retrieval, and LLM judge.
- Adds provenance guardrails, evaluation focus profiles, evaluator guidance, model budgets, and durable checkpoints for interrupted runs.
- Includes retrieval with FAISS, offline fake modes for testing, CI drift gates without live model dependencies, and refreshed GPT-5.5 provider defaults.

## Key Decision Choices

- Combined deterministic metrics with LLM judging instead of relying on either alone.
- Built retrieval against benchmark sets so evaluations can cite comparative evidence rather than only judge impressions.
- Added durable async job semantics with retries, backoff, and resumable checkpoints for long evaluation runs.
- Kept all provider calls backend-only and introduced provider-budget controls to avoid runaway evaluation cost.
- Treated evidence provenance as a product requirement, separating candidate evidence from benchmark evidence and surfacing remediation warnings.

## Tech Stack

- FastAPI, Python, SQLite, SQLAlchemy-style persistence patterns
- React, Vite, TypeScript, Playwright
- python-pptx, pypdf, FAISS, Pydantic
- Optional OpenAI, Gemini, or MLAPI providers for embeddings, judging, and captioning

## Architecture Snapshot

ArtifactBench separates evaluation into upload/extraction, benchmark-set management, retrieval indexing, and report generation. The backend stores candidate artifacts, builds benchmark corpora, runs deterministic analytics and retrieval, then invokes an LLM judge when configured. The frontend provides benchmark and candidate management, progress tracking, and master-detail report review. Long-running work is backed by durable job state and run artifacts under local data directories.

## Portfolio Content Angle

This should be positioned as an evaluation product, not a slide-only tool: benchmarking AI-generated artifacts, combining deterministic analytics with LLM review, and enforcing provenance-aware quality signals.

## Evidence Gaps / Refresh Notes

- Repo is private, so the public writeup should focus on evaluation design, product workflow, and governance choices.
- Strong candidate for a writeup that contrasts "AI generation" with "AI evaluation," which fits the portfolio's evaluation-system narrative.
- If later visuals are refreshed, prioritize the report viewer, benchmark-set workflow, and evaluation pipeline diagram.
