# ML Trading Strategist

## Project Thesis

A modular quantitative trading research platform that compares rule-based, supervised, and reinforcement learning strategies under realistic backtesting assumptions.

## Business Problem

Most trading projects showcase one strategy in isolation. This project instead addresses the broader problem of how to compare different learning paradigms under a common framework with realistic execution costs and portfolio context.

## Outcome and Evidence

- Supports benchmark, manual, tree-based, and Q-learning strategies.
- Includes portfolio management, technical indicators, transaction-cost modeling, and interactive exploration.
- Uses a config-driven architecture designed for reproducible strategy research.
- Exposes the framework through a Streamlit dashboard.

## Key Decision Choices

- Benchmarked multiple strategy families rather than optimizing only one.
- Included transaction cost, market impact, and slippage to avoid unrealistic backtests.
- Used configuration files for reproducible experimentation.
- Built extensibility into indicators, strategies, and cost models instead of hardcoding one workflow.

## Tech Stack

- Python 3.11
- Streamlit
- scikit-learn
- pandas / NumPy
- YAML configuration
- Custom market simulator and portfolio manager

## Architecture Snapshot

The UI layer feeds a configuration manager, which drives strategy modules. Indicator generation and market simulation feed portfolio analytics and performance metrics, which are surfaced back through the dashboard.

## Portfolio Content Angle

Position this as a research platform for strategy comparison, not just a single trading bot. The credibility comes from fair comparisons and realistic backtesting.

## Evidence Gaps / Refresh Notes

- Later content pass should extract the strongest comparative result between manual, tree-based, and RL strategies.
- Good candidate for a portfolio page with “approach comparison” instead of a single hero metric.

