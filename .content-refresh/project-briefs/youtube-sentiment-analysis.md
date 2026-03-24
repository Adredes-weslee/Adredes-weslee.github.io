# Advanced YouTube Comment Sentiment Analysis Platform

## Project Thesis

An end-to-end NLP system for collecting, labeling, analyzing, and serving large-scale YouTube comment sentiment with cloud-aware transformer deployment.

## Business Problem

Social comment streams are large, noisy, and operationally awkward to deploy against. This project addresses both the analytics question and the deployment question: how to process real user text at scale and still deliver a usable public demo.

## Outcome and Evidence

- Collected and processed 114,109 comments from a high-profile YouTube video.
- Produced a sentiment split of 78.8% negative and 21.2% positive.
- Supports environment-aware model choice: larger local model vs smaller cloud model.
- Delivers a multi-page Streamlit dashboard with prediction, exploration, and research views.

## Key Decision Choices

- Built the full data pipeline from API collection through preprocessing and inference.
- Used VADER for large-scale preprocessing and transformer models for interactive inference.
- Added environment detection to switch between model sizes for local versus cloud deployment.
- Optimized specifically for Streamlit Community Cloud constraints rather than pretending local conditions apply everywhere.

## Tech Stack

- Python 3.11+
- YouTube Data API v3
- Streamlit
- Hugging Face Transformers
- PyTorch
- VADER sentiment
- Plotly and supporting analytics libraries

## Architecture Snapshot

Comments are collected from the YouTube API, cleaned and normalized, labeled for dataset analysis, and then surfaced through a Streamlit dashboard that also offers real-time transformer inference for individual text samples.

## Portfolio Content Angle

Lead with scale plus deployment realism: 114K+ comments, transformer inference, and adaptive cloud-friendly model packaging.

## Evidence Gaps / Refresh Notes

- Later content pass should decide whether to foreground the dataset size, the sentiment insight, or the deployment tradeoff.
- Strong candidate for a portfolio case study around “productionizing NLP under lightweight hosting constraints.”

