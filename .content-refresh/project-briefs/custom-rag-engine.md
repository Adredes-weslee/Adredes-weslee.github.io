# Custom RAG Engine for Enterprise Document QA

## Project Thesis

A local-first enterprise RAG system built for document understanding and repository-aware question answering without sending data to external APIs.

## Business Problem

Teams need searchable, explainable answers across internal documentation and code-adjacent knowledge, but enterprise privacy and deployment constraints often rule out hosted LLM pipelines.

## Outcome and Evidence

- Delivers a working local RAG application with ingestion, embeddings, FAISS indexing, retrieval, QA, and evaluation layers.
- Supports GPU acceleration with CPU fallback.
- Uses code-aware and text-aware embedding paths to better handle mixed repository content.
- Explicitly documents deployment constraints and positions local deployment as the correct production mode.

## Key Decision Choices

- Prioritized local inference with Ollama over external hosted models for privacy.
- Split code and non-code retrieval artifacts instead of using one generic embedding path.
- Added GPU-aware FAISS and model loading to exploit local hardware where available.
- Built evaluation and model-check surfaces into the system rather than treating QA quality as ad hoc.
- Rejected Streamlit Community Cloud as a primary deployment target because it breaks the local-model assumption.

## Tech Stack

- UI: Streamlit
- Retrieval: FAISS, LangChain-style orchestration
- Models: Ollama local LLMs
- Data processing: custom ingestion, extraction, and enhancement pipeline
- Performance: CUDA / GPU acceleration with CPU fallback
- Testing and utilities: Python test suite, compatibility checks, logging

## Architecture Snapshot

Files are discovered and extracted, optionally enhanced, embedded into separate indexes, and stored in FAISS plus docstores. Queries flow through retrieval, question handling, and a RAG chain, with evaluation support for response quality.

## Portfolio Content Angle

Present this as a privacy-first enterprise AI system: local inference, hybrid embeddings, code-aware retrieval, and deployment realism.

## Evidence Gaps / Refresh Notes

- Later content pass should extract any measured retrieval or latency results from test outputs.
- Strong candidate for a case study about designing around hard infrastructure constraints rather than ideal cloud assumptions.

