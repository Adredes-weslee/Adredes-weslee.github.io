# New Repo Intake

Use this automation when a new GitHub repo may need portfolio coverage.

## Goal

Detect repos that are not yet represented in the portfolio, clone them, inspect the real codebase, and draft website-ready outputs for review.

## Inputs

- [`repo-registry.yml`](/C:/Users/tcmk_/Downloads/Adredes-weslee.github.io/.codex/automations/repo-registry.yml)
- the GitHub repos visible to the authenticated user
- the local website repo

## Required Behavior

1. Compare the current GitHub repo set against `repo-registry.yml`.
2. Ignore:
   - `Adredes-weslee/Adredes-weslee.github.io`
   - archived forks
   - pure data dumps
   - repos explicitly listed in `excluded_repos`
3. For each newly detected candidate repo:
   - clone it into `repos/`
   - inspect the codebase, not just `README.md`
   - inspect manifests, entrypoints, tests, workflows, demos, docs, screenshots, and architecture surfaces
   - determine:
     - business problem
     - outcome
     - architecture
     - stack
     - demo status
     - whether a long-form article is justified
     - whether the repo should be public-case-study only because it is private
4. Create or update:
   - `.content-refresh/project-briefs/<slug>.md`
   - `_projects/<slug>.md`
   - optionally `_posts/<date>-<slug>.md`
   - `repo-registry.yml`
   - include `last_repo_sync_ref` for the repo HEAD inspected during intake
   - include `public_demo_url` when the repo exposes a real public demo
5. If the repo lacks grounded visuals, queue or draft image work instead of inventing unsupported claims.
6. Open a reviewable PR or leave a review-ready patch set. Do not auto-merge.

## Writing Standard

- Project page: concise case-study surface
- Article: deeper architecture, decisions, tradeoffs, experiments
- Never copy the repo README as-is
- Private repos: keep code private, but allow public case-study framing if the content is safe

## Output Checklist

- new repo cloned
- registry updated
- project brief created
- project page drafted
- article drafted if justified
- private/public handling explicit
- review queue or PR prepared
