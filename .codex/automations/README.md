# Portfolio Automations

These files are the repo-local operating contract for Codex Automations.

They are intended to back `review-first` automations in the Codex app, not silent direct pushes to `main`.

## Recommended Automations

1. `new-repo-intake.md`
   - Trigger: scheduled daily or manual run
   - Purpose: detect new repos not yet represented on the site, clone them, inspect the real codebase, and draft website surfaces

2. `repo-drift-sync.md`
   - Trigger: weekly
   - Purpose: revisit already-tracked repos whose code moved since the last website sync

3. `backlog-rotation.md`
   - Trigger: monthly
   - Purpose: surface one or two older or secondary repos for deeper inspection and possible portfolio promotion

4. `demo-health-watch.md`
   - Trigger: daily
   - Purpose: check public demo repos, CI health, and live demo URLs, then draft follow-up work when drift appears

## Review Policy

- Prefer `clone + inspect + synthesize + PR`.
- Do not trust `README.md` as the primary truth source.
- For new or changed repos, inspect manifests, runnable entrypoints, tests, CI, demo surfaces, assets, and architecture docs.
- For private repos, default to public case-study language and never expose private code details beyond what is already intentionally documented.
- Website updates should land as PRs or review-queue tasks, not automatic direct merges.

## Registry

The canonical tracked fleet lives in:

- [`repo-registry.yml`](/C:/Users/tcmk_/Downloads/Adredes-weslee.github.io/.codex/automations/repo-registry.yml)

This registry is the source of truth for:

- which repos are already represented on the website
- which repos have project pages, articles, and demos
- which repos are private
- which repos should be considered flagship, supporting, or backlog work

## Default Include / Exclude Rules

Include by default:

- repos owned by `Adredes-weslee`
- explicitly allowlisted external repos that map to Wes Lee's portfolio work
- repos with meaningful code, assets, or runnable surfaces

Exclude by default:

- the website repo itself: `Adredes-weslee/Adredes-weslee.github.io`
- archived forks, throwaway experiments, and pure data drops
- generated artifact directories inside repos
- repo README-only summaries that do not inspect the codebase

## Output Contract

When an automation decides a repo should be added or refreshed on the site, the preferred outputs are:

- `.content-refresh/project-briefs/<slug>.md`
- `_projects/<slug>.md`
- optionally `_posts/<date>-<slug>.md`
- image refresh tasks when grounded visuals are missing
- a reviewable PR or queued diff summary
