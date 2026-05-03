# Learning Personalization Engine

## Source inspection

- Repository: `Adredes-weslee/learning-personalization-engine`
- Visibility: private
- Local clone: `repos/learning-personalization-engine`
- Sync ref: `5cd2e11de3d3cfbdcfa3871182c4796ce867a69e`
- Verification: `python -m pytest` passed 32 tests on 2026-05-03.

## Public framing

The public portfolio surface should describe this as a learning-personalization system for a private education platform without naming the employer or deployment-specific account details. The important story is the architecture: platform probes, event normalization, Bayesian knowledge tracing, recommendation inventory, teacher-facing review surfaces, and contract tests.

## What the repo contains

- Python package under `src/learning_personalization_engine` for event contracts, profile rollups, knowledge tracing, recommendation inventory, pilot config, and local stores.
- Browser probe scripts that inspect an existing learning platform surface when credentials are provided.
- Pilot/demo validation scripts that run against local fixtures and produce teacher-view summaries.
- Tests covering event normalization, BKT-style mastery updates, metadata parsing, recommendation behavior, store behavior, and end-to-end pilot outputs.

## What to avoid publicly

- Do not name the employer, deployment account, or login surfaces.
- Do not use raw screenshots from the private platform probes.
- Do not present the system as a deployed public demo; it is a private-code case study.

## Portfolio emphasis

- Learning events are treated as product evidence, not just telemetry.
- Recommendation output stays reviewable rather than becoming an opaque automation layer.
- Personalization claims are backed by tests, fixtures, and explicit assumptions.
- The system is strongest as a bridge between platform observations and teacher-facing decision support.
