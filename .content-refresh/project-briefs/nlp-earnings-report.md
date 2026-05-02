# NLP Earnings Report Analysis

## Project Thesis

A financial NLP workbench that combines domain-specific preprocessing, sentiment analysis, topic exploration, and dashboard review for earnings-report disclosures.

## Business Problem

Earnings reports are dense, repetitive, and high-volume. Analysts need structured ways to extract what management is signaling, what themes are emerging, and whether those signals can help explain or predict market behavior.

## Outcome and Evidence

- Builds an end-to-end pipeline from data processing to embeddings, sentiment, topic modeling, feature extraction, and dashboard review.
- Uses financial-domain lexicons rather than generic sentiment alone.
- Supports an interactive dashboard for exploratory analysis and model surfaces.
- Includes versioning to make dataset and experiment configuration reproducible.

## Key Decision Choices

- Combined traditional and heavier NLP representations where the local environment supports them instead of committing to one NLP paradigm.
- Used Loughran-McDonald-style finance sentiment and documented the split between lean hosted demo and full Conda local stack.
- Treated topic modeling as a core analysis layer, not a side experiment.
- Added data versioning to make report-analysis experiments reproducible.

## Tech Stack

- Python 3.11
- Streamlit
- scikit-learn
- Hugging Face Transformers
- BERTopic
- LDA / NMF / Gensim topic modeling

## Architecture Snapshot

Raw earnings data flows through preprocessing and versioning, then into embedding, sentiment, topic-modeling, and feature-extraction modules. Those features feed predictive models and the interactive dashboard.

## Portfolio Content Angle

Frame this as financial disclosure intelligence, not generic text analytics or a production-grade market-prediction system. The key is domain adaptation, reproducible preprocessing, and dashboard review.

## Evidence Gaps / Refresh Notes

- Later content pass should pull out the most credible predictive or explanatory result.
- Strong candidate for a narrative about extracting usable signals from messy financial disclosures.
