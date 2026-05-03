---
layout: post
title: "Engineering NLP for Financial Disclosure Review"
date: 2025-05-09 09:45:00 +0800
categories: [nlp, finance, machine-learning, data-science]
tags: [nlp, finance, machine-learning, data-science, text-analysis, python]
author: Wes Lee
feature_image: /assets/images/article-heroes/2025-05-09-nlp-earnings-report-analysis.jpg
project_page: /projects/nlp-earnings-analyzer/
display_title: "Engineering NLP for Financial Disclosures"
archive_title: "NLP for Financial Disclosures"
---

## Introduction: financial text needs a domain-aware pipeline

Earnings reports are dense, repetitive, and full of domain-specific language. A generic text-analysis demo can tokenize the reports, but it usually fails to preserve what makes financial disclosure review useful: versioned preprocessing, financial sentiment language, topic exploration, and a dashboard that makes the analysis inspectable.

This project is best understood as a financial NLP workbench. It turns earnings-report text into cleaned datasets, sentiment and topic artifacts, feature outputs, and a Streamlit dashboard for exploratory review.

> Related: For the concise portfolio summary, see the [Earnings Report Intelligence Platform project page](/projects/nlp-earnings-analyzer/).
>
> Demo: The public Streamlit surface is available [here](https://adredes-weslee-nlp-earnings-report-streamlit-app-0uttcu.streamlit.app/). The full local dashboard is best run from the repo's `environment.yaml` / Python 3.11 Conda environment.

## The useful architecture is the pipeline, not a single model

The repo is organized around a multi-stage workflow:

- load and clean the raw earnings text
- create versioned processed datasets
- build embeddings and text features
- run financial sentiment and topic analysis
- persist artifacts under `models/`
- expose the outputs through the dashboard

That structure matters because disclosure analysis is iterative. Analysts need to compare data versions, review topic behavior, inspect sentiment outputs, and understand which artifacts are loaded. A one-shot notebook would be easier to write, but the pipeline structure makes the work easier to reproduce and review.

## Financial sentiment needs domain-specific handling

Financial language does not map cleanly to generic positive and negative word lists. Words such as "liability," "charge," or "provision" depend heavily on context. The repo therefore keeps finance-domain resources such as Loughran-McDonald-style sentiment alongside heavier NLP methods where the local environment supports them.

The important claim is not that every transformer path is active in every deployment. The cleaner claim is that the system separates:

- lean hosted demo behavior
- full local Conda dashboard behavior
- persisted artifacts and model state
- documented limitations around unavailable modes

That distinction keeps the public portfolio page honest while still showing the fuller engineering intent.

## The dashboard is a review surface

The dashboard is useful because it gives the pipeline an interface. It includes analysis views for uploaded or sampled text, dataset exploration, topic inspection, model review, prediction simulation, and performance summaries.

The strongest pattern is that the dashboard does not pretend all artifacts are always present. The repo includes sample processed files and checked-in model artifacts for a bounded demo path, while the full pipeline can be rerun locally for deeper analysis. That makes the project more credible than a dashboard that silently assumes a perfect environment.

## Refresh caveat: keep the claims tied to visible evidence

The prior article version leaned too hard into benchmark-style performance claims. The repo does contain performance documentation and saved artifacts, but the portfolio story should stay closer to what the code visibly supports:

- financial disclosure preprocessing
- versioned data outputs
- sentiment and topic-analysis modules
- a Streamlit review dashboard
- a clear distinction between public demo and full local environment

That is still a strong project. It just should be framed as a financial NLP workbench rather than a production-grade market-prediction system.

## The broader lesson: reproducibility is the product feature

The main takeaway is that domain-specific NLP becomes useful when the workflow is reproducible and inspectable. For financial text, that means the code needs to preserve preprocessing choices, artifact versions, model-loading behavior, and dashboard assumptions.

This project works best as a portfolio case when those engineering choices are foregrounded. The value is not "NLP can read earnings reports." The value is that a disclosure-review pipeline can be structured so its data, artifacts, and interface stay explainable.
