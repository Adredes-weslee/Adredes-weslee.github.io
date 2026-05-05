---
layout: post
title: "Building Longevity Lab for Health Risk Scenario Modeling"
date: 2026-05-02 14:00:00 +0800
categories: [health-ai, public-health, model-evaluation]
tags: [fastapi, react, public-health, model-cards, calibration, provenance, causal-inference]
author: Wes Lee
feature_image: /assets/images/article-heroes/2026-05-02-building-longevity-lab-for-health-risk-scenario-modeling.jpg
project_page: /projects/longevity-lab/
display_title: "Building Longevity Lab for Health Scenario Modeling"
archive_title: "Longevity Lab Health Scenario Modeling"
---

## Introduction: a health-risk product needs more than a score

Longevity Lab is a useful portfolio case because it is not just a model wrapped in a dashboard. It is a product prototype for health-risk communication, and that changes the engineering problem.

The system has to answer several questions at once:

- which public data sources were used
- whether the app is running in demo or artifact-backed mode
- how a current scenario differs from a what-if scenario
- what the active model artifacts and evidence endpoint can and cannot support
- where predictive associations stop and causal claims would require a separate analysis

> Related: for the shorter case-study version, see the [Longevity Lab project page](/projects/longevity-lab/).

> Demo: the public Render deployment is available at [longevity-lab-frontend.onrender.com](https://longevity-lab-frontend.onrender.com/). Initial load can take about a minute while the service wakes and loads sample data.

## The key product decision was separating evidence surfaces

Many risk tools collapse everything into one interface. Longevity Lab deliberately separates the surfaces:

- Explorer for scenario comparison
- Data Evidence for active scoring inputs, source status, provenance, report evidence, and inactive gaps
- Model Cards for active artifact behavior and limitations
- Scenario Lab for structured comparison summaries
- a separate causal workbench for explicit causal questions

That separation is the main design lesson. It makes the product more inspectable and reduces the chance that a user mistakes a predictive score for diagnosis or causal advice.

## Artifact-backed scoring changes the app from demo to system

The backend can run in demo mode, but the stronger path is artifact-backed scoring. Training and benchmark scripts produce calibrated bundles with manifests, metrics, subgroup slices, pollutant ablations, explanation records, and model-card-ready summaries. The current artifact-backed scope covers eight BRFSS-derived conditions: heart disease, chronic lung disease, asthma, stroke, depression, diabetes, chronic kidney disease, and arthritis.

That matters because the frontend is not only consuming a prediction endpoint. It is consuming a runtime state and evidence model. The user should know when the app is serving a real artifact, what model family produced it, whether uncertainty is declared, and what caveats apply.

The architecture supports that by keeping preprocessing, artifact loading, API schemas, the evidence-status endpoint, and frontend contracts aligned. This is the difference between a working prototype and a product surface that can be reviewed.

## Public-data provenance is treated as product infrastructure

The repo is built around public, script-downloadable sources such as BRFSS, EPA AirData, ACS, SVI, and CDC PLACES-oriented context. The important part is not the number of sources. It is the source discipline.

The roadmap sets clear boundaries:

- public access only
- documented provenance
- feasible geography and time joins
- bias and ecological-fallacy review
- local development and free-tier deployment constraints

That is a pragmatic approach for a public-health portfolio project. It keeps the system useful without pretending that a row-joined public dataset can answer every clinical or causal question.

## Causal analysis stays separate from prediction

One of the strongest choices is that causal inference is not mixed into the main risk score. The repo defines a separate causal workbench with explicit questions, estimands, adjustment candidates, exclusions, DAG assumptions, negative controls, and sensitivity checks.

That boundary is important. A scenario slider can show how a model score changes under a hypothetical input, but that is not the same thing as estimating an intervention effect. Longevity Lab keeps those concepts apart, which makes the product more trustworthy.

## The broader lesson: risk communication is an interface and evidence problem

The strongest part of Longevity Lab is not any single model family. It is the way the repo connects modeling, provenance, runtime mode, model cards, and scenario comparison into one inspectable system.

The takeaway is straightforward:

- public-health models need evidence-status surfaces, not just predictions
- model-card metadata should be part of the product interface
- causal language needs explicit assumptions and separate workflows
- deployment mode should be visible to users
- local-first artifacts make development and review more defensible

That is the kind of engineering discipline a health-risk communication product needs before it asks anyone to trust the interface.
