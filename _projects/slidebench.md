---
layout: project
title: "SlideBench AI Evaluation Workbench"
description: "Benchmarking system for AI-generated slides and learning artifacts with retrieval, provenance checks, and judge-assisted evaluation."
date: 2026-03-24
categories: [evaluation-systems, edtech, multimodal, benchmarking, ai-platforms]
image: /assets/images/project-covers/slidebench.jpg
technologies: [FastAPI, React, TypeScript, SQLite, FAISS, python-pptx, Playwright]
repo_private: true
repo_note: "Source code is private. The public case study and article focus on evaluation design, provenance controls, and product workflow."
blog_post: /evaluation-systems/multimodal/edtech/2026/03/24/building-slidebench-to-evaluate-ai-generated-slides.html
---

## Business context

As teams generate more slide decks and learning artifacts with AI, quality review becomes subjective, slow, and hard to reproduce. A useful benchmark product needs extraction, deterministic metrics, benchmark comparison, retrieval-backed evidence, and durable reports rather than a single judging prompt.

## Outcome

- Built a FastAPI backend and React frontend for uploading, indexing, evaluating, and reviewing decks.
- Supports multiple artifact types including PPTX, markdown curriculum files, PDFs, and optional project-code inputs.
- Runs a modular evaluation flow across extraction, deterministic metrics, retrieval, and LLM judging.
- Adds provenance guardrails, evaluation profiles, per-run model budgets, and resumable checkpoints for long evaluations.

## Key decisions

- Combined deterministic analytics with LLM judging instead of relying on either alone.
- Added retrieval over benchmark decks so evaluations can cite comparative evidence.
- Kept provider calls backend-only and introduced budget controls for evaluation cost.
- Treated provenance tagging and remediation warnings as part of the product, not a reporting afterthought.

## System design

Artifacts are uploaded into a local evaluation workspace, extracted into typed intermediate artifacts, indexed into benchmark sets, and evaluated through a staged pipeline. Deterministic metrics and retrieval outputs provide structured evidence before optional judge calls produce report sections. The frontend focuses on benchmark management, job progress, and report review, while the backend owns evaluation state and run artifacts.

## Stack

- FastAPI, Python, SQLite, Pydantic
- React, Vite, TypeScript, Playwright
- python-pptx, pypdf, FAISS
- Optional OpenAI, Gemini, and MLAPI provider adapters
