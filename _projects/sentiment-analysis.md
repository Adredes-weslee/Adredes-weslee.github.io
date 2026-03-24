---
layout: project
title: "YouTube Sentiment Analysis Platform"
description: "Large-scale comment intelligence built from collection pipeline to transformer-powered dashboard."
date: 2024-12-15
categories: [nlp, machine-learning, transformers, social-media-analytics, streamlit]
image: /assets/images/project-covers/sentiment-analysis.jpg
technologies: [Python, Streamlit, Transformers, VADER, PyTorch, Plotly, YouTube Data API, Audience analytics]
github: https://github.com/Adredes-weslee/Sentiment-Analysis-and-NLP-for-a-Youtube-Video
blog_post: /nlp/machine-learning/transformers/2024/12/15/building-youtube-comment-sentiment-analyzer.html
streamlit_app: https://adredes-weslee-sentiment-analysis-and-nlp-f-dashboardapp-kqphrr.streamlit.app/
---

## Business context

Digital teams often have plenty of audience comments and very little structured insight. This project was built to show the full path from data collection to scalable sentiment analysis, with deployment choices shaped by the constraints of lightweight hosting.

## Outcome

- Collected and processed 114,109 comments from a high-profile YouTube video.
- Produced a sentiment split of 78.8% negative and 21.2% positive in the analyzed dataset.
- Combined VADER for large-scale preprocessing with transformer models for richer interactive inference.
- Delivered a multi-page Streamlit dashboard for prediction, exploration, and research views.

## Key decisions

- Built the full data pipeline from API collection through preprocessing and inference.
- Used lighter methods for bulk processing and transformer models where deeper inference mattered.
- Added environment detection to switch between model sizes for local versus cloud deployment.
- Optimized the product around Streamlit Community Cloud constraints instead of pretending every environment supports a full local stack.

## System design

The system collects comments through the YouTube API, cleans and structures the dataset, runs scalable preprocessing and sentiment layers, and then exposes the outputs through interactive dashboard pages for exploration and targeted inference.

## Stack

- Python, YouTube Data API, VADER, transformers, and PyTorch
- Streamlit and Plotly for the product surface
- Data collection, preprocessing, sentiment inference, and deployment-aware model routing
