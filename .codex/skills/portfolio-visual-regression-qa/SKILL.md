---
name: portfolio-visual-regression-qa
description: Verify the Jekyll portfolio visually after layout, CSS, navigation, image-card, sidebar, gutter, responsive, or zoom-related changes. Use when Codex must run Playwright screenshots, inspect DOM geometry, compare desktop/mobile breakpoints, and prove that visible pages do not overlap, clip, blank, or drift.
---

# Portfolio Visual Regression QA

## What this skill is for

Use this skill to test user-facing portfolio layout changes with a real browser before claiming a visual fix is complete.

## When to use it

Use this skill when:
- desktop gutter, sidebar, drawer, or page-width behavior changes
- cards, thumbnails, article heroes, or project images may crop poorly
- a page looks blank until refresh or after navigation
- responsive behavior must be checked across multiple viewport widths
- the user asks for screenshots or visual proof

## When not to use it

Do not use this skill for:
- text-only content edits that do not affect layout
- repo README work outside the website
- image generation without page rendering checks

## Required inputs

- Jekyll source in this repo
- `bundle exec ruby -S jekyll build --destination _site_check`
- a local static server over `_site_check`
- Playwright CLI or another real browser automation path
- affected routes, or the default set: `/`, `/about/`, `/projects/`, `/blog/`, `/streamlit-apps/`, `/contact/`

## Required workflow

1. Check `git status --short` and note existing local changes.
2. Stop stale local preview/browser processes that could serve old CSS or old builds.
3. Build with `bundle exec ruby -S jekyll build --destination _site_check`.
4. Serve `_site_check` on a fresh safe local port. Avoid Chromium unsafe ports such as `4045`.
5. Capture screenshots for affected routes at relevant viewport widths.
   - For desktop gutter/sidebar issues, use at least `1920x1080`, `2400x1350`, and `3200x1800`.
   - For normal responsive work, include `1366x768`, `1440x900`, `1920x1080`, and one mobile width.
6. Inspect DOM bounds when screenshots show ambiguity.
   - Check content left/right bounds.
   - Check sidebar/drawer bounds.
   - Check image/card bounds and whether text is clipped or hidden.
7. Compare screenshots before and after the change when possible.
8. If visual failures remain, keep iterating. Do not report success from CSS reasoning alone.

## Expected outputs

- local preview URL
- screenshot paths with absolute paths
- short pass/fail table by route and viewport
- any DOM measurements used to justify the verdict
- a scoped list of changed files

## Validation and stop condition

Stop only when:
- Jekyll build passes
- affected pages render in Playwright without blank content
- screenshots show no incoherent overlap, clipping, or hidden text
- desktop gutter/sidebar geometry is stable at the tested wide widths
- any remaining risk is explicitly named

Do not commit or push unless the user asks.
