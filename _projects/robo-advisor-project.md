---
layout: project
title: "AI Portfolio Advisory System"
description: "Foundation-model risk profiling and objective-aware portfolio optimization for personalized investing."
date: 2025-06-24
categories: [foundation-models, reinforcement-learning, fintech, ai, tabular-ai]
image: /assets/images/project-covers/robo-advisor-project.jpg
technologies: [Python, TabPFN, PyTorch, Streamlit, Deep Q-Networks, Portfolio optimization, Market regime detection]
github: https://github.com/Adredes-weslee/Using_Artificial_Intelligence_to_Develop_a_Robo_Advisor
blog_post: /ai/finance/foundation-models/reinforcement-learning/2025/06/24/robo-advisor-risk-profiling-portfolio-optimization.html
streamlit_app: https://adredes-weslee-using-artificial-intelligenc-dashboardapp-juewyb.streamlit.app/
---

## Business context

Most robo-advisors rely on static questionnaires and a small set of fixed portfolio templates. This project explored a more adaptive alternative: using tabular foundation models for risk profiling and objective-aware reinforcement learning for portfolio construction.

## Outcome

- Combined TabPFN-based risk profiling with PyTorch DQN portfolio optimization.
- Supports multiple investment objectives and market-regime-aware behavior instead of one fixed recommendation path.
- The local TabPFN workflow reports risk-prediction accuracy above 0.85 R^2 in the documented setup.
- Ships a Streamlit dashboard that keeps the system usable in both local and cloud-constrained environments.

## Key decisions

- Used TabPFN for risk assessment instead of only traditional tabular ML.
- Trained objective-specific agents instead of one universal portfolio policy.
- Added market-regime awareness so recommendations respond to changing conditions.
- Designed separate local and cloud execution paths so the product remains usable when full RL inference is not feasible.

## System design

User inputs feed a risk-profiling layer, then route into objective-aware portfolio logic that selects or simulates the appropriate optimization path. The application adapts between full local-AI mode and lighter cloud-compatible behavior while preserving the same overall user journey.

## Stack

- Python, TabPFN, PyTorch, and portfolio optimization utilities
- Streamlit for the advisory product surface
- Risk profiling, regime detection, and objective-aware RL components
