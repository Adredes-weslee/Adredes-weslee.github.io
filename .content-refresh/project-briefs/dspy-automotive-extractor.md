# DSPy Automotive Extractor

## Project Thesis

A structured extraction system that turns prompt design into a measurable optimization workflow for automotive complaint data.

## Business Problem

Manual prompt engineering is slow, hard to reproduce, and often insecure for enterprise use cases. Teams that need structured extraction from unstructured complaints need a repeatable optimization process, not intuition-driven prompt tinkering.

## Outcome and Evidence

- Best-performing setup reached 51.33% F1.
- Phase 1 showed reasoning fields improved all tested strategies.
- Phase 2 showed meta-optimization did not beat the best reasoning-field baseline.
- Ships both local and cloud-compatible dashboards for demonstrating results.

## Key Decision Choices

- Treated prompting as a DSPy compilation problem rather than manual prompt crafting.
- Used local Ollama inference to preserve privacy and avoid external API dependency.
- Designed a two-phase experimental program to test reasoning fields separately from meta-optimization.
- Used explicit structured evaluation metrics instead of anecdotal output quality.
- Added Langfuse observability so experiments are inspectable and reproducible.

## Tech Stack

- Python 3.11+
- DSPy
- Ollama
- Langfuse
- Pydantic
- Streamlit
- scikit-learn and supporting analytics libraries

## Architecture Snapshot

The pipeline loads and cleans NHTSA complaint data, defines DSPy signatures and strategies, compiles extraction programs, runs optimization experiments, and persists results for analysis dashboards.

## Portfolio Content Angle

Position this as an LLM systems and evaluation project, not only an NLP demo. The key differentiator is the experimental decision logic and the insight that more complex prompt optimization can regress performance.

## Evidence Gaps / Refresh Notes

- Later content pass should turn the “reasoning fields beat meta-optimization” result into the headline insight.
- This is one of the strongest candidates for a blog-driven case study because the repo already contains a research arc.

