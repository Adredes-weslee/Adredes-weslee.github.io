# Intelligent Content Analyzer

## Project Thesis

A multilingual educational document assistant that supports upload, retrieval, QA, and summarization through a modular microservice architecture.

## Business Problem

Students and knowledge workers often struggle to extract useful answers from long documents, especially across multiple languages and formats. A usable solution needs ingestion, retrieval, generation, and quality control to work together, not just a single QA endpoint.

## Outcome and Evidence

- Provides a live API gateway and Streamlit UI.
- Supports document upload, question answering, and summarization.
- Uses hybrid retrieval, optional reranking, optional Redis cache, and optional Langfuse tracing.
- Supports both local single-process mode and multi-service HTTP mode.

## Key Decision Choices

- Chose a service-oriented architecture instead of one monolithic app.
- Kept a dual deployment model: in-process for simplicity, HTTP microservices for realism and scaling.
- Used hybrid retrieval rather than dense-only search.
- Added an evaluation and confidence stage so answers can be scored, not just generated.
- Designed the system to preserve query language in the response where possible.

## Tech Stack

- FastAPI microservices
- Streamlit UI
- FAISS
- Redis optional cache
- Langfuse optional tracing
- Docker Compose
- Embeddings, retrieval, generation, and evaluation services

## Architecture Snapshot

The API gateway orchestrates upload, retrieval, QA, and summary routes. Ingest parses and chunks documents, retrieval blends sparse and dense search with FAISS, generation handles answer and summary prompts, and evaluation computes confidence signals.

## Portfolio Content Angle

This should be presented as a systems project for document intelligence: microservices, orchestration, multilingual behavior, and retrieval quality.

## Evidence Gaps / Refresh Notes

- Later content pass should decide whether to position this as edtech, enterprise knowledge tooling, or both.
- Good candidate for a more visual architecture diagram on the site because the repo already has C4-style documentation.

