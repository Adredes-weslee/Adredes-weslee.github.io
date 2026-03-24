# AI-Powered Robo-Advisor

## Project Thesis

A next-generation robo-advisor that combines foundation-model risk profiling with multi-objective reinforcement learning and cloud-aware portfolio optimization.

## Business Problem

Personalized portfolio advice requires more than generic Modern Portfolio Theory. The system needs to understand investor risk, adapt portfolio policy to different objectives, and still remain deployable under constrained environments.

## Outcome and Evidence

- Combines TabPFN risk profiling with PyTorch DQN-based portfolio optimization.
- Supports multiple investment objectives and market-regime-aware behavior.
- Distinguishes local full-AI mode from cloud fallback mode rather than pretending both environments can support the same pipeline.
- Ships a Streamlit dashboard for risk profiling and optimization workflows.

## Key Decision Choices

- Used TabPFN for risk assessment instead of only traditional tabular ML.
- Trained objective-specific RL agents instead of a single universal portfolio policy.
- Added market-regime awareness to reflect changing conditions.
- Designed separate local and cloud execution paths so the product remains usable when full RL inference is not feasible.
- Incorporated transfer-learning style reuse for agent management.

## Tech Stack

- Python 3.11+
- Streamlit
- PyTorch
- TabPFN
- Portfolio math / MPT utilities
- Market analysis and regime-detection utilities

## Architecture Snapshot

Survey and market data feed a risk model and a portfolio-optimization engine. Objective selection and regime awareness shape the decision policy, and an environment check decides whether to use full RL or a cloud-safe optimization fallback before results reach the dashboard.

## Portfolio Content Angle

Position this as an applied AI product, not just a finance notebook: foundation models, RL, decision objectives, and deployment-aware design.

## Evidence Gaps / Refresh Notes

- Later content pass should surface one or two concrete portfolio outcomes or objective comparisons.
- Strong candidate for a hero project because it combines advanced modeling with a user-facing financial product.

