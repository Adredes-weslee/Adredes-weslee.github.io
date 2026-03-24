---
layout: project
title: "Custom RAG Engine for Enterprise QA"
description: "A privacy-first RAG system for enterprise document QA using local inference and GPU-aware retrieval."
date: 2024-10-29
categories: [nlp, machine-learning, rag, enterprise-ai, local-deployment]
image: /assets/images/project-covers/rag-engine-project.jpg
technologies: [Python, Streamlit, Ollama, FAISS, LangChain, GPU acceleration, Code-aware embeddings]
github: https://github.com/Adredes-weslee/Custom-RAG-Engine-for-Enterprise-Document-QA
blog_post: /ai/nlp/rag/2024/10/29/building-effective-rag-systems.html
---

## Business context

Enterprise teams need searchable, explainable answers across internal documentation and code-adjacent knowledge, but privacy and infrastructure constraints often make hosted RAG stacks the wrong fit. This project was built around the premise that local inference is a product requirement, not a compromise.

## Outcome

- Delivered a working local RAG application with ingestion, embeddings, FAISS indexing, retrieval, QA, and evaluation layers.
- Supports GPU acceleration with CPU fallback so the same system can adapt to different hardware.
- Uses code-aware and text-aware embedding paths to better handle mixed repository content.
- Explicitly documents why local deployment is the correct production mode for the system.

## Key decisions

- Prioritized local Ollama inference over external hosted models for privacy and control.
- Split code and non-code retrieval artifacts instead of forcing everything through one embedding path.
- Added GPU-aware FAISS and model loading so the system can exploit local hardware when available.
- Built evaluation and model-check surfaces into the workflow instead of treating QA quality as ad hoc.

## System design

Files are discovered and extracted, optionally enhanced, embedded into separate indexes, and stored in FAISS plus supporting docstores. Queries flow through retrieval, question handling, and a RAG chain, with evaluation hooks for response quality and deployment diagnostics.

## Stack

- Python, Streamlit, Ollama, FAISS, and LangChain-style orchestration
- Local LLM inference with GPU-aware performance paths
- Custom ingestion, enhancement, retrieval, and evaluation modules
