# SlideBench

## Project Thesis

A benchmarking system for AI-generated slide decks and learning artifacts that combines deterministic metrics, benchmark retrieval, provenance checks, and LLM-based judging behind a durable evaluation workflow.

## Business Problem

Teams increasingly generate slides, curricula, and project artifacts with AI, but most review remains subjective and hard to reproduce. A usable benchmarking product needs structured extraction, repeatable metrics, benchmark comparison, evaluation reports, provenance guardrails, and resumable background jobs rather than a one-shot judge prompt.

## Outcome and Evidence

- Delivers a FastAPI backend plus Vite/React frontend for uploading, indexing, evaluating, and reviewing decks.
- Supports multiple artifact types: PPTX, curriculum markdown/JSON, project PDFs, and optional project-code zip inputs.
- Runs a modular evaluation pipeline: extract, deterministic metrics, retrieval, and LLM judge.
- Adds provenance guardrails, evaluation focus profiles, evaluator guidance, model budgets, and durable checkpoints for interrupted runs.
- Includes retrieval with FAISS, offline fake modes for testing, and CI drift gates without live model dependencies.

## Key Decision Choices

- Combined deterministic metrics with LLM judging instead of relying on either alone.
- Built retrieval against benchmark decks so evaluations can cite comparative evidence rather than only judge impressions.
- Added durable async job semantics with retries, backoff, and resumable checkpoints for long evaluation runs.
- Kept all provider calls backend-only and introduced provider-budget controls to avoid runaway evaluation cost.
- Treated evidence provenance as a product requirement, separating candidate evidence from benchmark evidence and surfacing remediation warnings.

## Tech Stack

- FastAPI, Python, SQLite, SQLAlchemy-style persistence patterns
- React, Vite, TypeScript, Playwright
- python-pptx, pypdf, FAISS, Pydantic
- Optional OpenAI, Gemini, or MLAPI providers for embeddings, judging, and captioning

## Architecture Snapshot

SlideBench separates evaluation into upload/extraction, benchmark-set management, retrieval indexing, and report generation. The backend stores deck artifacts, builds benchmark corpora, runs deterministic analytics and retrieval, then invokes an LLM judge when configured. The frontend provides benchmark and candidate management, progress tracking, and master-detail report review. Long-running work is backed by durable job state and run artifacts under local data directories.

## Portfolio Content Angle

This should be positioned as an evaluation product, not a deck-generation tool: benchmarking AI outputs, combining deterministic analytics with LLM review, and enforcing provenance-aware quality signals.

## Evidence Gaps / Refresh Notes

- Repo is private, so the public writeup should focus on evaluation design, product workflow, and governance choices.
- Strong candidate for a writeup that contrasts “AI generation” with “AI evaluation,” which fits the portfolio’s evaluation-system narrative.
- If later visuals are refreshed, prioritize the report viewer, benchmark-set workflow, and evaluation pipeline diagram.
