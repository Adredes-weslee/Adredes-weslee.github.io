---
layout: project
title: "Learning Personalization Engine"
description: "A private-code personalization system that turns learning-platform events into mastery traces, recommendation inventories, and teacher-facing review surfaces."
date: 2026-04-19
categories: [ai-platforms, edtech, personalization, learning-analytics, decision-support]
image: /assets/images/project-covers/learning-personalization-engine.jpg
technologies: [Python, Bayesian Knowledge Tracing, pytest, Playwright probes, JSON fixtures, recommendation rules]
repo_private: true
repo_note: "Source code is private. The public case study and article focus on event contracts, personalization logic, review surfaces, and verification."
blog_post: /ai-platforms/edtech/personalization/2026/04/19/building-learning-personalization-engine.html
---

## Business context

Learning platforms often collect useful activity signals without turning them into a clear personalization workflow. The hard part is not only predicting learner state. The system also needs a defensible event contract, assumptions that can be reviewed, recommendations that teachers can inspect, and testable boundaries between platform observation and instructional action.

## Outcome

- Built a private-code engine for normalizing learning events, estimating mastery, and generating recommendation inventories.
- Added probe-backed ingestion paths for platform observations while keeping public documentation anonymized.
- Implemented local validation scripts and 32 passing tests across event parsing, knowledge tracing, recommendation rules, stores, metadata, and end-to-end pilot outputs.
- Produced teacher-view summaries that keep personalization advice reviewable instead of hiding it behind an opaque score.

## Key decisions

- Treated platform observations as evidence that must be normalized before personalization logic can use it.
- Kept mastery estimation, recommendation rules, and teacher-facing summaries as separate layers.
- Used fixtures and deterministic tests so the engine can be verified without exposing private platform data.
- Avoided raw platform screenshots in the public portfolio because the useful story is the architecture, not the private UI.

## System design

Browser probes and local fixtures feed a normalized event layer. Profile rollups and knowledge-tracing utilities convert those events into mastery estimates and learner state. Recommendation inventory logic then maps that state into reviewable next actions, while pilot scripts generate teacher-facing summaries and audit artifacts. The repository keeps credentials, deployment-specific details, and platform observations outside the public case study.

## Stack

- Python package with event contracts, BKT-style tracing, recommendation rules, and local stores
- pytest coverage for core model and workflow behavior
- Playwright-based platform probes for authenticated inspection when credentials are available
- JSON fixtures and scriptable validation outputs for repeatable pilot checks
