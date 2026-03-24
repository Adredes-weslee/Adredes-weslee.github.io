---
layout: project
title: "ML Trading Strategist"
description: "A comparative trading research platform spanning rule-based, tree-based, and reinforcement learning strategies."
date: 2025-05-12
categories: [machine-learning, finance, reinforcement-learning, data-science]
image: /assets/images/project-covers/ml-trading-strategist.jpg
technologies: [Python, Streamlit, scikit-learn, pandas, NumPy, YAML, Reinforcement learning, Backtesting]
github: https://github.com/Adredes-weslee/ML-Trading-Strategist
blog_post: /ai/finance/machine-learning/reinforcement-learning/2025/05/12/ml-trading-strategist-comparing-learning-approaches.html
streamlit_app: https://adredes-weslee-ml-trading-strategist-app-pu7qym.streamlit.app/
---

## Business context

Most trading projects over-focus on one model family and under-specify execution costs, which makes the comparison look stronger than it is. This project was designed as a research platform for comparing rule-based, supervised, and reinforcement-learning strategies under more realistic backtesting assumptions.

## Outcome

- Supports benchmark, manual, tree-based, and Q-learning strategy families in one framework.
- Includes transaction costs, market impact, and slippage so results are not based on frictionless assumptions.
- In the repo's JPM backtests, the tree-based strategy reached 42.7% cumulative return with a 1.48 Sharpe ratio.
- Exposes the framework through a Streamlit dashboard for strategy analysis and comparison.

## Key decisions

- Benchmarked multiple strategy families rather than optimizing only one approach.
- Modeled commission, impact, and portfolio constraints to keep the research closer to live trading conditions.
- Used YAML configuration so experiments remain reproducible and comparable.
- Built extensibility into indicators, strategies, and simulation components instead of hardcoding one workflow.

## System design

Market data flows through preprocessing and technical-indicator pipelines, then into manual, tree-based, or Q-learning strategy modules. Generated trades are evaluated by a market simulator with cost modeling, and the dashboard surfaces comparative performance metrics and strategy behavior.

## Stack

- Python, pandas, NumPy, scikit-learn, and YAML-driven configuration
- Custom indicator library, strategy modules, and cost-aware market simulation
- Streamlit for interactive experiment review
