# Backlog Rotation

Use this automation to gradually revisit older, weaker, or non-demo repos so they do not get forgotten.

## Goal

Pick one or two lower-priority repos each run, inspect them deeply, and decide whether they deserve better portfolio treatment, a fresh project brief, or no action.

## Selection Rules

Prefer repos that are:

- marked `secondary` or not yet upgraded recently in `repo-registry.yml`
- missing strong visuals
- missing an article or case-study polish
- newly relevant to current positioning

Avoid in the same run:

- repos already touched by `new-repo-intake`
- repos already touched by `repo-drift-sync`
- the website repo itself

## Required Behavior

1. Select one or two repos from the backlog.
2. Clone or refresh them locally.
3. Inspect the real codebase, not only the README.
4. Produce one of:
   - `promote`: draft stronger project/article surfaces
   - `refresh`: tighten existing surfaces
   - `defer`: no website action, but note why
5. Update `.content-refresh/project-briefs/` if the repo needed deeper synthesis.
6. Update `repo-registry.yml` with the new rotation result and sync date.

## Desired Outcome

A steady monthly trickle of portfolio-quality upgrades without requiring manual repo triage every time.

