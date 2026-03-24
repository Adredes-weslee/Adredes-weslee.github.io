---
layout: project
title: "Intelligent Content Analyzer"
description: "A multilingual document intelligence system built as modular retrieval, generation, and evaluation services."
date: 2026-03-23
categories: [llm, rag, microservices, document-intelligence, edtech]
image: /assets/images/project-covers/intelligent-content-analyzer.jpg
technologies: [FastAPI, Streamlit, FAISS, Redis, Langfuse, Docker Compose, Hybrid retrieval, Evaluation services]
github: https://github.com/Adredes-weslee/intelligent-content-analyzer
blog_post: /ai/rag/document-intelligence/2026/03/24/building-service-oriented-document-intelligence.html
streamlit_app: https://adredes-weslee-intelligent-content-analyzer-uiapp-stwg9a.streamlit.app/
---

## Business context

Students and knowledge workers often need more than a single question-answering endpoint. They need document upload, retrieval, summarization, confidence signals, and multilingual behavior to work together as one product rather than as disconnected experiments.

## Outcome

- Provides a live API gateway and Streamlit UI for upload, question answering, and summarization.
- Uses hybrid retrieval, optional reranking, optional Redis cache, and optional Langfuse tracing.
- Supports both local single-process mode and multi-service HTTP mode.
- Preserves the query language in the response path where possible, which makes the system more usable across multilingual inputs.

## Key decisions

- Chose a service-oriented architecture instead of one monolithic app.
- Kept a dual deployment model: in-process for simplicity, HTTP microservices for realism and scaling.
- Used hybrid retrieval rather than dense-only search.
- Added evaluation and confidence scoring so answers can be judged, not just generated.

## System design

An API gateway orchestrates upload, retrieval, QA, and summary routes. Ingest services parse and chunk documents, retrieval blends sparse and dense search with FAISS, generation handles answer and summary prompts, and evaluation services compute confidence-oriented signals for the UI and API.

## Stack

- FastAPI microservices, Streamlit UI, and Docker Compose
- FAISS, Redis, Langfuse, and hybrid retrieval components
- Dedicated services for ingestion, retrieval, generation, and evaluation
