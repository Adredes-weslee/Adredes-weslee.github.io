# Repo-local Codex overlay

## Template
- `small-hybrid-probe`

## Repo Shape
- Jekyll portfolio site built on Hydejack and deployed via GitHub Pages.
- Primary source files live in root pages such as `about.md`, `projects.md`, `blog.md`, `contact.md`, and `streamlit-apps.md`.
- Project detail pages live in `_projects/`; blog content lives in `_posts/`; reusable HTML templates live in `_layouts/`; theme overrides and assets live under `assets/`.
- Site-wide settings, navigation, collection config, and theme wiring live in `_config.yml`, `Gemfile`, and `Gemfile.lock`.
- `_site/` is generated output, not source of truth.
- `repos/` contains cloned backing repositories for research and sync work; it is local workspace material, not site source.

## Validation
- Install dependencies with `bundle install` when Ruby gems are missing or stale.
- Run `bundle exec jekyll build` for a production-style validation pass before finishing substantial site changes.
- Run `bundle exec jekyll serve` for local review when changing layouts, navigation, styling, or content that needs visual verification.
- If `_config.yml` changes, restart the Jekyll server because config changes are not hot-reloaded.

## Guardrails
- Do not edit `_site/`, `.jekyll-cache/`, or other generated/cache folders as if they were source files.
- Keep local-only research material out of commits. Current ignored workspace items include `repos/`, `about_old.md`, and `Wes_Lee_Resume.pdf`.
- Treat `_projects/` pages and related `_posts/` entries as coupled surfaces when a project name, positioning, or repo link changes.
- Keep project claims aligned with the actual backing repositories in `repos/` or on GitHub; do not introduce features, metrics, or deployment claims that are not supported by the underlying work.
- Preserve the existing Hydejack/Jekyll structure unless there is a deliberate migration plan; prefer targeted edits over theme-level churn.

## Drift Surface
- `README.md`
- `_config.yml`
- Root navigation/content pages: `about.md`, `projects.md`, `blog.md`, `contact.md`, `streamlit-apps.md`, `index.html`
- `_projects/*`
- `_posts/*`
- `_layouts/*`
- `assets/css/*`
- `assets/js/*`
- `assets/images/*` when images, branding, or project thumbnails change
- `.gitignore` when local workflow files or source inputs need to stay untracked
