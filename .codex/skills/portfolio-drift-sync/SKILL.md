---
name: portfolio-drift-sync
description: Compare a tracked backing repo against the last synced commit, inspect the real changes, and update only the minimum portfolio surfaces needed to keep claims accurate.
---

# Portfolio Drift Sync

## What this skill is for

Use this skill when an already-tracked repo changed and the portfolio may need targeted updates.

## When to use it

Use this skill when:
- a tracked repo has new commits since `last_repo_sync_ref`
- demo status changed
- architecture, metrics, outputs, or visibility changed
- a project page or article might now be stale

## When not to use it

Do not use this skill when:
- the repo is completely new and has no website surface yet
- the task is only to monitor demos/CI
- the change is obviously unrelated to public portfolio claims

## Required inputs

- [`.codex/automations/repo-registry.yml`](C:/Users/tcmk_/Downloads/Adredes-weslee.github.io/.codex/automations/repo-registry.yml)
- local clone in `repos/`
- target website files in `_projects/`, `_posts/`, and related top-level pages

## Required workflow

1. Look up the repo in `repo-registry.yml`.
2. Compare current repo HEAD against `last_repo_sync_ref`.
3. If there is no meaningful drift, return a short no-op summary.
4. If there is drift, inspect the actual code diff and current repo state.
5. Prioritize only claim-relevant changes:
   - runnable entrypoints
   - demo URL or live demo health
   - architecture boundaries
   - metrics or outputs cited publicly
   - visibility changes
6. Update only the minimum needed surfaces:
   - `_projects/<slug>.md`
   - matching `_posts/<slug>.md` when the article is affected
   - `streamlit-apps.md` or homepage/demo wording when demo status changed
   - `.codex/automations/repo-registry.yml`
7. Write the inspected repo HEAD back to `last_repo_sync_ref` once the review-ready patch is prepared.

## Expected outputs

- a short drift summary
- targeted website edits only where justified
- updated `last_repo_sync_ref`
- optionally updated `public_demo_url` or `has_public_demo`

## Validation and stop condition

Stop when:
- the repo was compared against `last_repo_sync_ref`
- any public claim changes are traceable to actual repo changes
- unrelated pages were not rewritten
- the registry sync ref matches the inspected repo HEAD

Prefer in-place edits over broad rewrites. Do not auto-merge or auto-push by default.
