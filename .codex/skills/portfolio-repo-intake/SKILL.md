---
name: portfolio-repo-intake
description: Clone and inspect a new portfolio repo, create a project brief from real code evidence, draft the website project page, optionally draft a deeper article, and update the repo registry.
---

# Portfolio Repo Intake

## What this skill is for

Use this skill when a new GitHub repo should be considered for portfolio coverage and the website should be updated from real repo inspection rather than a README-only summary.

## When to use it

Use this skill when:
- a new repo was added under `Adredes-weslee` or another allowlisted owner
- an older repo was never turned into a website project page
- a private repo needs a public-safe case study surface

## When not to use it

Do not use this skill when:
- the repo is already represented and only needs a small claim update
- the task is only demo monitoring
- the repo is a data dump, archived fork, or throwaway experiment

## Required inputs

- [`.codex/automations/repo-registry.yml`](C:/Users/tcmk_/Downloads/Adredes-weslee.github.io/.codex/automations/repo-registry.yml)
- local website repo
- GitHub access or an existing local clone

## Required workflow

1. Check `repo-registry.yml` to confirm the repo is new or not yet fully represented.
2. Clone the repo into `repos/` if it is not already present locally.
3. Inspect the real codebase, not just `README.md`.
   Required surfaces:
   - manifests such as `requirements.txt`, `environment.yml`, `pyproject.toml`, `package.json`
   - runnable entrypoints
   - tests and workflows
   - demo surfaces and screenshots
   - docs, diagrams, architecture notes, and assets
4. Decide:
   - business problem
   - outcome and evidence
   - architecture
   - stack
   - public demo status
   - whether an article is justified
   - whether the repo must be handled as a private public-safe case study
5. Create or update:
   - `.content-refresh/project-briefs/<slug>.md`
   - `_projects/<slug>.md`
   - optionally `_posts/<date>-<slug>.md`
   - `.codex/automations/repo-registry.yml`
6. If visuals are weak, record image work instead of inventing claims.

## Expected outputs

- a repo-backed project brief
- a drafted project page
- an article draft when justified
- a registry entry with:
  - `visibility`
  - `tier`
  - `project_slug`
  - `article_slug` when present
  - `has_public_demo`
  - `public_demo_url` when present
  - `last_repo_sync_ref`

## Validation and stop condition

Stop when:
- the repo has been cloned or refreshed
- the brief is grounded in repo inspection
- the drafted website surfaces are materially different from the README
- private/public handling is explicit
- the registry entry is complete enough for future drift checks

Do not auto-merge or auto-push by default. Leave a review-ready diff or PR.
