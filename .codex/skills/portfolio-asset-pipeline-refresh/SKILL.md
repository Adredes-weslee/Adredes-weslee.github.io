---
name: portfolio-asset-pipeline-refresh
description: Refresh and validate generated portfolio imagery for project covers, project evidence panels, article heroes, homepage images, sidebar/profile imagery, and demo/article card thumbnails. Use when visual assets look stale, cropped, text-heavy, overlapping, inconsistent, or out of sync with project/article content.
---

# Portfolio Asset Pipeline Refresh

## What this skill is for

Use this skill to update generated website imagery from grounded project sources, then verify that the resulting pages present those images cleanly.

## When to use it

Use this skill when:
- project cards, article cards, or demo cards have bad crops
- article hero images have overlapping or unreadable text
- project evidence images need to reflect updated repos
- homepage or sidebar editorial imagery needs refreshing
- generated assets should be regenerated after project/article changes

## When not to use it

Do not use this skill when:
- the task is only to edit written content
- the user wants AI image generation instead of repo-grounded compositing
- source screenshots or repo evidence are missing and no fallback is acceptable

## Required inputs

- generator scripts:
  - `scripts/generate_project_covers.py`
  - `scripts/generate_project_evidence.py`
  - `scripts/generate_editorial_images.py`
- source images under `output/project-sources/`, `repos/`, and existing `assets/images/`
- affected pages in `_projects/`, `_posts/`, `index.html`, `blog.md`, `projects.md`, and `streamlit-apps.md`

## Required workflow

1. Identify the affected visual surface:
   - project covers
   - project evidence panels
   - article heroes
   - homepage/editorial imagery
   - Streamlit/demo cards
2. Start with an audit-only pass unless the user explicitly asked to regenerate assets.
   - Compile the generator scripts.
   - Confirm referenced asset paths exist.
   - Count expected files in `project-covers`, `project-evidence`, and `article-heroes`.
3. Audit current source images before regenerating.
   - Prefer wide, readable UI or repo evidence.
   - Avoid tall screenshots that will lose key text in 16:9 frames.
   - Avoid text-heavy compositions where labels collide with summary copy or chips.
4. Patch the relevant generator script when the source choice, layout, typography, or crop logic is the real issue.
5. Regenerate only the necessary asset family when possible.
6. Build the site.
7. Run visual QA on affected routes with `portfolio-visual-regression-qa`.
8. Check for:
   - broken images
   - hidden or overlapping text inside generated images
   - bad card crops
   - mismatched article/project/demo labels

## Expected outputs

- list of regenerated assets
- generator script changes when applicable
- screenshot evidence for affected pages
- concise before/after note for any previously broken surface

## Validation and stop condition

Stop only when:
- generated files exist at the paths referenced by source pages/layouts
- generator scripts compile
- Jekyll build passes
- affected pages show readable images with no obvious crop failure
- no generated text overlaps chips, panels, or other text
- `git diff --stat` contains only intentional source, generator, or asset changes

Do not rewrite unrelated assets just to make timestamps fresh.
