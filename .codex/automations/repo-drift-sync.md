# Repo Drift Sync

Use this automation to keep existing portfolio surfaces aligned with the backing repos.

## Goal

Revisit already-tracked repos whose code, README, demos, or CI changed since the last portfolio sync, then draft the minimum website changes needed to keep the site honest and current.

## Inputs

- [`repo-registry.yml`](/C:/Users/tcmk_/Downloads/Adredes-weslee.github.io/.codex/automations/repo-registry.yml)
- local clones in `repos/`
- website pages in `_projects/` and `_posts/`

## Required Behavior

1. For each tracked repo, compare the current repo HEAD against `last_repo_sync_ref` in `repo-registry.yml`.
2. Prioritize repos where one of the following changed materially:
   - README first screen
   - runnable entrypoint
   - demo URL or demo health
   - architecture or service boundary
   - metrics or outputs now used on the site
   - repo visibility
3. For changed repos:
   - inspect the actual diff and the current codebase
   - verify whether website claims are still accurate
   - update:
     - `_projects/<slug>.md`
     - matching article if needed
     - `streamlit-apps` or homepage copy if demo status changed
     - `repo-registry.yml`
   - write the new inspected HEAD back to `last_repo_sync_ref` once the review-ready patch is prepared
4. If drift is only README polish and does not affect the portfolio claim surface, prefer updating the registry sync marker without rewriting site content.
5. Open a reviewable PR or leave a review-ready patch set. Do not auto-merge.

## Guardrails

- Do not rewrite all pages every time.
- Prefer targeted claim sync.
- Private repos remain review-required.
- If a demo broke, say so explicitly and downgrade the portfolio wording until fixed.
