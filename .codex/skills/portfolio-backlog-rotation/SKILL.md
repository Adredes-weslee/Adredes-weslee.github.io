---
name: portfolio-backlog-rotation
description: Pick one or two older or secondary repos, inspect them deeply, and decide whether to promote, refresh, or defer their portfolio treatment.
---

# Portfolio Backlog Rotation

## What this skill is for

Use this skill when you want to revisit older, weaker, or secondary repos so the portfolio backlog moves forward without ad hoc triage every time.

## When to use it

Use this skill when:
- you want to resurface one or two older repos
- you want to decide whether a repo deserves promotion or only minor refresh work
- you want a bounded monthly-style backlog review

## When not to use it

Do not use this skill when:
- the repo is brand new
- the repo already needs a known drift sync
- the task is only demo monitoring

## Required inputs

- [`.codex/automations/repo-registry.yml`](C:/Users/tcmk_/Downloads/Adredes-weslee.github.io/.codex/automations/repo-registry.yml)
- local clone or GitHub access
- existing website surfaces if the repo is already represented

## Required workflow

1. Pick one or two repos that are:
   - `secondary`
   - weakly surfaced
   - missing strong visuals or article polish
   - newly relevant to current positioning
2. Clone or refresh the repos locally if needed.
3. Inspect the real codebase, not just the README.
4. Inspect the existing website surfaces too, so the decision considers repo quality and current public representation together.
5. Classify each repo as:
   - `promote`
   - `refresh`
   - `defer`
6. If work is justified, update:
   - `.content-refresh/project-briefs/`
   - `_projects/`
   - optionally `_posts/`
   - `.codex/automations/repo-registry.yml`

## Expected outputs

- chosen backlog repos
- one classification per repo
- reason for that classification
- list of website surfaces that would be touched

## Validation and stop condition

Stop when:
- one or two repos were examined deeply enough to justify the classification
- the recommendation is grounded in repo inspection
- no portfolio work is drafted for repos that should be deferred

Prefer small steady upgrades over broad backlog churn.
