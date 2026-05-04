---
name: portfolio-gpt-image-evaluator-optimizer
description: Evaluate, regenerate, and converge GPT-generated portfolio image assets for this Jekyll portfolio. Use when project covers, project evidence images, or article heroes were generated with gpt-image-2/Codex image generation and need rubric-based QA, text-safety checks, prompt repair, Codex exec regeneration, contact-sheet review, or local/live website image validation.
---

# Portfolio GPT Image Evaluator Optimizer

## What This Skill Is For

Use this skill to run the evaluator-optimizer loop for AI-generated portfolio images.

The workflow is intentionally stricter than normal visual refresh work: it treats the 54 canonical images as a manifest-backed asset set, evaluates them against the local rubric, regenerates only failing assets, and stops only after the generated files are accepted and the website still builds.

## When To Use It

Use this skill when:
- GPT-generated project covers, project evidence panels, or article heroes look too similar, text-like, off-topic, or low quality.
- The user asks for an evaluator/optimizer pass on generated image assets.
- The user asks to regenerate failed portfolio images with Codex exec or `gpt-image-2`.
- The image prompts, rubric, or accepted image set need to become canonical.

Do not use this skill for:
- Old screenshot/compositing-only refreshes. Use `portfolio-asset-pipeline-refresh`.
- Pure CSS/card crop issues. Use `portfolio-visual-regression-qa`.
- Content-only project/article edits.

## Required Inputs

Use these repo-local artifacts:
- `.content-refresh/image-prompts/gpt-image-prompts.jsonl`
- `.content-refresh/image-prompts/image-evaluation-rubric.md`
- `scripts/generate_gpt_image_assets.py`
- `scripts/run_codex_exec_image_manifest.py`
- `assets/images/project-covers/*.jpg`
- `assets/images/project-evidence/*.jpg`
- `assets/images/article-heroes/*.jpg`

Expected canonical counts:
- 18 project covers
- 18 project evidence images
- 18 article heroes
- 54 total generated portfolio images

## Workflow

1. Check the worktree first.
   - Run `git status --short`.
   - Do not overwrite unrelated user changes.

2. Verify the manifest and current asset inventory.
   - Confirm the manifest has 54 rows.
   - Confirm the three canonical asset directories each contain 18 JPGs.
   - Confirm every manifest `output` path exists or identify missing assets before generating.

3. Run a dry-run selection before generation.
   - Use `python scripts/run_codex_exec_image_manifest.py --dry-run`.
   - Add `--slug` or `--kind` when testing a smaller subset.

4. Evaluate current images against the rubric.
   - Read `.content-refresh/image-prompts/image-evaluation-rubric.md`.
   - Check text safety first: no readable words, letters, numbers, labels, fake UI text, logos, watermarks, company names, or client names.
   - Score project specificity, role fit, sibling distinction, portfolio diversity, website usability, and editorial polish.
   - Fail any image that violates text safety, has average score below 4.0, or has any non-text criterion below 3.

5. Repair prompts only for failed assets.
   - Keep project context grounded in the cloned repo and existing content.
   - Change the prompt direction that caused the failure: medium, metaphor, camera angle, palette, subject, sibling distinction, or text-safety wording.
   - Avoid broad prompt rewrites for assets that already passed.

6. Regenerate failed assets sequentially with Codex exec.
   - Use `python scripts/run_codex_exec_image_manifest.py --slug <slug> --kind <kind>`.
   - Do not parallelize Codex exec image generation unless the user explicitly accepts the risk.
   - The harness copies accepted generated PNGs into the intended JPG output path.

7. Re-evaluate regenerated assets.
   - Repeat steps 4 through 6 until every regenerated asset passes or the failure mode is explicitly blocked.
   - Use subagents as independent judges only when the task is large enough to justify parallel review.

8. Rebuild and verify the website.
   - Run `bundle exec jekyll build --trace`.
   - Confirm generated pages reference existing images.
   - For publish-ready work, crawl local or live pages and verify image URLs return HTTP 200 with image content types.

9. Report the result.
   - List accepted assets, regenerated assets, rejected assets, and unresolved risks.
   - Include the exact validation commands and pass/fail counts.
   - Do not commit or push unless the user asks.

## Useful Commands

```powershell
python scripts/run_codex_exec_image_manifest.py --dry-run
```

```powershell
python scripts/run_codex_exec_image_manifest.py --slug creator-ai --kind project-cover
```

```powershell
bundle exec jekyll build --trace
```

```powershell
@'
from pathlib import Path
import json
root = Path(r"C:\Users\tcmk_\Downloads\Adredes-weslee.github.io")
manifest = root / ".content-refresh" / "image-prompts" / "gpt-image-prompts.jsonl"
rows = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines() if line.strip()]
missing = [row["output"] for row in rows if not (root / row["output"]).exists()]
print("manifest_rows", len(rows))
print("missing_outputs", len(missing))
for rel in ["assets/images/project-covers", "assets/images/project-evidence", "assets/images/article-heroes"]:
    print(rel, len(list((root / rel).glob("*.jpg"))))
if missing:
    print("\n".join(missing))
    raise SystemExit(1)
'@ | python -
```

## Validation And Stop Condition

Stop only when:
- manifest row count and asset counts are correct
- every accepted asset passes the rubric
- regenerated assets have no readable text or sensitive names
- Jekyll build passes
- image references resolve locally or live, depending on the task
- the final diff contains only intentional prompt, script, report, or asset changes

If image generation fails because Codex exec or the image tool is unavailable, stop with a clear blocked status and keep any existing accepted images untouched.
