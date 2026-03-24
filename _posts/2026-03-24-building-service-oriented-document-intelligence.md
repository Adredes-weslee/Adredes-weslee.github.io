---
layout: post
title: "Building Service-Oriented Document Intelligence"
date: 2026-03-24 09:00:00 +0800
categories: [ai, rag, document-intelligence]
tags: [fastapi, streamlit, faiss, redis, langfuse, hybrid-retrieval, reranking, multilingual, system-design]
author: Wes Lee
feature_image: /assets/images/article-heroes/2026-03-24-building-service-oriented-document-intelligence.jpg
---

## Introduction: Why this needed more than a single QA app

Many document-QA demos stop at a neat answer box. That is enough for a prototype, but it is not enough for a system that has to ingest files reliably, preserve multilingual behavior, expose debuggable failure points, and give users some signal about whether the answer should be trusted.

This project was built around that gap. Instead of packaging upload, retrieval, generation, and evaluation into one opaque service, the system decomposes those responsibilities into separate FastAPI services behind a gateway. The result is not just a nicer codebase. It is a product shape that makes it easier to inspect, debug, scale, and replace individual parts as the retrieval and generation stack evolves.

> Related: For the concise portfolio summary, see the [Intelligent Content Analyzer project page](/projects/intelligent-content-analyzer/).
>
> Demo: The public interface is available [here](https://adredes-weslee-intelligent-content-analyzer-uiapp-stwg9a.streamlit.app/).

## The first architectural decision: keep the gateway thin and the services explicit

The repo uses an API gateway as the single public entry point, then routes work to dedicated services for ingest, retrieval, embeddings, generation, and evaluation. That split matters because each part fails differently:

- ingestion has to parse mixed file types and chunk them cleanly
- retrieval has to balance sparse and dense recall
- generation has to follow response structure and citation behavior
- evaluation has to estimate groundedness instead of pretending every answer is equally trustworthy

The gateway therefore acts as an orchestrator, not as the entire application. It exposes the product routes users care about, but the retrieval, generation, and evaluation logic stay in their own services where they can be tested and replaced independently.

This is also why the repo supports two run modes. In local development, the gateway can import the other services in-process to keep setup light. In the more realistic deployment path, the gateway talks to them over HTTP. That dual-mode design is a practical compromise: one path for iteration speed, one path for service realism.

## Retrieval is treated as a ranking problem, not just an embedding lookup

The retrieval service combines BM25-style lexical search with dense vector retrieval backed by FAISS, then optionally reranks results before they reach the generation step. That is a materially better fit for educational and technical documents than dense-only search.

Dense retrieval helps with semantic similarity, but it can miss exact terminology, formulas, or section-specific language. Lexical retrieval helps with precision around those details, but on its own it can miss paraphrases and multilingual variation. The hybrid path is an attempt to get both:

- lexical recall for exact phrasing and technical terms
- dense recall for semantic similarity
- reranking when the candidate set needs cleaner ordering

The repo documentation also shows threshold checks and refinement loops when recall is weak. That is an important product choice. The system is not assuming retrieval succeeded just because some chunks were returned. It tries to decide whether the evidence base is strong enough before generation proceeds.

## Confidence is a first-class output, not a post-hoc apology

One of the strongest choices in this repo is that evaluation is a dedicated service. The generation step does not simply emit an answer and stop. The evaluation layer blends retrieval signals, rerank signals, and answer-quality heuristics into a confidence-oriented output.

That matters for the user experience because document QA systems fail in different ways:

- retrieval can pull weak evidence
- generation can overstate confidence
- multilingual queries can produce good language handling but weak grounding
- summarization can look coherent while still missing the source material

By isolating evaluation and confidence logic, the system leaves room for stricter groundedness checks, LLM-as-judge experiments, and feedback loops later without tangling that logic into the generation service itself. From a portfolio perspective, that is the difference between "RAG app" and "document intelligence system."

## Multilingual behavior changed the product requirements

The repo includes multilingual sample documents and explicitly tries to answer in the same language as the query. That is not just a nice extra. It changes what "good enough" means:

- embeddings need to behave reasonably across languages
- prompts need to preserve the user's language rather than force one response language
- evaluation has to judge answer quality even when the query language and source language differ

This is one of the reasons the system benefits from modularity. Once multilingual behavior enters the picture, it becomes much easier to justify separate services and clearer interfaces instead of burying everything inside a single notebook-to-demo pipeline.

## Ops and observability were built into the shape of the repo

Another good design choice is that the retrieval service exposes operational endpoints such as status and storage debugging, while shared tracing and cache code live outside any one service. Optional Redis caching and partial Langfuse integration point in the same direction: the system is trying to be observable, not just functional.

That shows up in a few practical ways:

- exact and semantic caching reduce repeat cost and latency
- FAISS persistence gives the retrieval layer a stable local index rather than a purely ephemeral workflow
- debug endpoints make it easier to inspect whether documents were chunked and indexed correctly
- shared tracing creates a path to service-level latency and error analysis

The repo also documents where this is still incomplete. Langfuse tracing is only partially wired. End-to-end tests across the full upload -> retrieve -> generate -> evaluate flow can still go further. CPU reranking on hosted infrastructure remains slower than ideal. Those gaps do not weaken the project. They make the architectural intent more credible because the repo is explicit about what is solved and what is still rough.

## The main lesson: better document products come from clearer boundaries

The project page summarizes this work as modular retrieval, generation, and evaluation services. The deeper takeaway is why that matters. Document intelligence products become more defensible when they make room for inspection:

- the retrieval stack can improve without rewriting the UI
- the evaluation layer can get stricter without destabilizing ingestion
- deployment can switch between local and service-oriented modes
- caching and tracing can evolve without rewriting the core application routes

That is the right pattern for a portfolio piece like this. The valuable part is not that it answers questions over documents. The valuable part is that it treats grounding, inspection, and operational flexibility as part of the product design from the beginning.
