# Demo Health Watch

Use this automation for the repos with public demos.

## Goal

Watch public demo repos for runtime drift, CI failures, or dead public URLs, then prepare follow-up fixes or website wording changes.

## Scope

Public-demo repos are the ones in `repo-registry.yml` with `has_public_demo: true`.

## Required Behavior

1. Check the repos in `repo-registry.yml` where `has_public_demo: true`, using `public_demo_url` as the canonical live surface:
   - latest CI state
   - public demo URL availability
   - obvious README/demo-entrypoint drift
2. If a demo is healthy, update the sync marker only.
3. If a demo is degraded:
   - inspect the repo
   - determine whether the break is:
     - docs drift
     - runtime failure
     - dependency/env drift
     - hosting-only issue
   - draft repo fixes if appropriate
   - draft website wording changes if the site needs to be more conservative
4. Leave a review-ready patch or PR summary. Do not auto-merge.

## Output

- repo-level issue summary
- recommended fix
- website impact summary
- updated `repo-registry.yml` sync metadata if appropriate, including `last_repo_sync_ref` when repo inspection happened
