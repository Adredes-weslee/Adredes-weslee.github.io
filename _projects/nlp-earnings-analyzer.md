---
layout: project
title: "Earnings Report Intelligence Platform"
description: "A financial NLP pipeline for sentiment, topic, and disclosure analysis across earnings reports."
date: 2025-05-09
categories: [nlp, finance, machine-learning, data-science]
image: /assets/images/project-covers/nlp-earnings-analyzer.jpg
technologies: [Python, Streamlit, scikit-learn, Transformers, BERTopic, Loughran-McDonald, Financial NLP]
github: https://github.com/Adredes-weslee/NLP_earnings_report
blog_post: /nlp/finance/machine-learning/data-science/2025/05/09/nlp-earnings-report-analysis.html
streamlit_app: https://adredes-weslee-nlp-earnings-report-streamlit-app-0uttcu.streamlit.app/
---

## Business context

Earnings reports are dense, repetitive, and full of signals that are hard to compare consistently across companies. This project was built to turn those disclosures into something analysts can explore through sentiment, topics, extracted metrics, and downstream modeling.

## Outcome

- Built an end-to-end workflow from data processing to embeddings, sentiment, topic modeling, feature extraction, and modeling.
- Combined finance-specific lexicons with transformer-based methods instead of relying on generic sentiment tooling.
- Added an interactive dashboard for exploratory analysis and model review, with a lean public demo surface and a heavier local research stack.
- Included data and experiment versioning so the analysis can be reproduced.

## Key decisions

- Combined traditional and transformer-based NLP rather than forcing a single method across every task.
- Used finance-domain resources such as Loughran-McDonald and FinBERT-style methods where domain language matters.
- Treated topic modeling as a core analysis layer instead of a side experiment.
- Added versioning so dataset and experiment changes stay traceable.

## System design

Reports are collected, cleaned, and transformed into multiple analysis paths for lexicon-based sentiment, transformer inference, topic discovery, feature extraction, and modeling. Those outputs are then surfaced through the dashboard for comparison and exploratory review. The public Streamlit surface stays lightweight, while the verified full local dashboard path uses the repo's Conda environment to support the heavier NLP stack cleanly.

## Stack

- Python, scikit-learn, transformers, BERTopic, and finance-domain lexicons
- Streamlit for interactive analysis, with `environment.yaml` as the verified full local run path
- Topic modeling, feature extraction, and experiment versioning utilities
