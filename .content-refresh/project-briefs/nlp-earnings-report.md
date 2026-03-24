# NLP Earnings Report Analysis

## Project Thesis

A financial NLP platform that combines domain-specific sentiment analysis, topic modeling, and predictive features to turn earnings reports into market-facing signals.

## Business Problem

Earnings reports are dense, repetitive, and high-volume. Analysts need structured ways to extract what management is signaling, what themes are emerging, and whether those signals can help explain or predict market behavior.

## Outcome and Evidence

- Builds an end-to-end pipeline from data processing to embeddings, sentiment, topic modeling, feature extraction, and modeling.
- Uses financial-domain lexicons rather than generic sentiment alone.
- Supports an interactive dashboard for exploratory analysis and model surfaces.
- Includes versioning to make dataset and experiment configuration reproducible.

## Key Decision Choices

- Combined traditional and transformer-based representations instead of committing to one NLP paradigm.
- Used Loughran-McDonald and FinBERT-style methods for finance-specific sentiment.
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

Frame this as financial language intelligence, not generic text analytics. The key is domain adaptation plus the combination of interpretability and prediction.

## Evidence Gaps / Refresh Notes

- Later content pass should pull out the most credible predictive or explanatory result.
- Strong candidate for a narrative about extracting usable signals from messy financial disclosures.

