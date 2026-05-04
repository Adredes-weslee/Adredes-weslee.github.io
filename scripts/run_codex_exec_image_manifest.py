from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
PROMPT_DIR = ROOT / ".content-refresh" / "image-prompts"
DEFAULT_MANIFEST = PROMPT_DIR / "gpt-image-prompts.jsonl"
REPORT_DIR = PROMPT_DIR / "codex-exec-reports"
TMP_DIR = PROMPT_DIR / "accepted-png"


PATH_RE = re.compile(r"GENERATED_PATH\s*=\s*(.+)")
ACCEPTED_RE = re.compile(r"ACCEPTED\s*=\s*yes", re.IGNORECASE)


def load_rows(manifest_path: Path, *, slugs: set[str] | None, kinds: set[str] | None) -> list[dict]:
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if slugs:
        rows = [row for row in rows if row["slug"] in slugs]
    if kinds:
        rows = [row for row in rows if row["kind"] in kinds]
    return rows


def report_name(row: dict, index: int) -> str:
    digest = hashlib.sha1(row["output"].encode("utf-8")).hexdigest()[:10]
    return f"{index:03d}__{row['kind']}__{row['slug']}__{digest}.txt"


def build_exec_prompt(row: dict) -> str:
    return textwrap.dedent(
        f"""
        Generate exactly one project-bound portfolio image using the built-in Codex image generation tool.
        Do not edit repository files. Do not run image API code. Use the built-in image tool only.

        Asset metadata:
        - kind: {row['kind']}
        - slug: {row['slug']}
        - intended output: {row['output']}
        - model intent: {row.get('model', 'gpt-image-2')}
        - quality intent: {row.get('quality', 'high')}
        - aspect ratio: 16:9 landscape

        Prompt to use:
        {row['prompt']}

        Mandatory validation before final answer:
        - The generated image must be a polished editorial portfolio visual that follows the project-specific art direction.
        - The only visual system constraint is text safety: no readable text, no fake UI copy, no letters, no numbers, no labels, no captions, no badges, no pills, no watermarks, no logos, no company names, no client names.
        - Do not use real screenshots or branded UI. Abstract interface-like shapes are allowed only if every mark is unreadable and decorative.
        - Make this asset visually distinguishable from other portfolio projects and from the sibling assets for the same project.
        - Use the requested medium, palette, metaphor, camera angle, and composition. Avoid defaulting to the same teal glass dashboard style.
        - It must be relevant to the asset metadata and prompt context.
        - It must be suitable as a 16:9 website card/hero image with safe margins.

        After generating, inspect the rendered image visually. Then locate the newest generated image path under the Codex generated_images directory.
        Final response must be exactly these fields, one per line:
        ACCEPTED=yes_or_no
        GENERATED_PATH=absolute_path_or_blank
        REASON=short visual validation note
        """
    ).strip()


def run_codex_exec(row: dict, index: int, *, timeout: int) -> Path:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORT_DIR / report_name(row, index)
    prompt = build_exec_prompt(row)
    codex_cmd = shutil.which("codex.cmd") or shutil.which("codex")
    if codex_cmd is None:
        raise RuntimeError("Could not find codex on PATH")
    comspec = os.environ.get("ComSpec", "cmd.exe")
    command = [
        comspec,
        "/d",
        "/s",
        "/c",
        codex_cmd,
        "exec",
        "--cd",
        str(ROOT),
        "--sandbox",
        "danger-full-access",
        "--output-last-message",
        str(report_path),
        "-",
    ]
    completed = subprocess.run(
        command,
        input=prompt,
        text=True,
        encoding="utf-8",
        errors="ignore",
        cwd=ROOT,
        timeout=timeout,
        check=False,
        capture_output=True,
    )
    stderr_path = report_path.with_suffix(".stderr.txt")
    if completed.stderr:
        stderr_path.write_text(completed.stderr, encoding="utf-8", errors="ignore")
    if completed.stdout:
        print(completed.stdout.strip(), flush=True)
    if not report_path.exists() or report_path.stat().st_size == 0:
        report_path.write_text(completed.stdout or "", encoding="utf-8", errors="ignore")
    if completed.returncode != 0:
        raise RuntimeError(f"codex exec failed for {row['kind']} {row['slug']} with exit {completed.returncode}")
    return report_path


def parse_report(report_path: Path) -> tuple[bool, Path | None, str]:
    text = report_path.read_text(encoding="utf-8", errors="ignore")
    accepted = bool(ACCEPTED_RE.search(text))
    path_match = PATH_RE.search(text)
    generated_path = Path(path_match.group(1).strip().strip("`")) if path_match else None
    reason = ""
    for line in text.splitlines():
        if line.startswith("REASON="):
            reason = line.partition("=")[2].strip()
            break
    return accepted, generated_path, reason


def copy_as_jpeg(source: Path, output: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, TMP_DIR / source.name)
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.save(output, format="JPEG", quality=94, optimize=True, progressive=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Codex exec image generation for a GPT image manifest.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--slug", action="append", help="Limit to one or more project slugs.")
    parser.add_argument("--kind", action="append", help="Limit to one or more row kinds.")
    parser.add_argument("--limit", type=int, help="Limit rows for smoke testing.")
    parser.add_argument("--start", type=int, default=0, help="Zero-based row offset after filtering.")
    parser.add_argument("--timeout", type=int, default=420, help="Timeout per codex exec call in seconds.")
    parser.add_argument("--dry-run", action="store_true", help="Print selected rows without generating.")
    args = parser.parse_args()

    rows = load_rows(
        args.manifest,
        slugs=set(args.slug) if args.slug else None,
        kinds=set(args.kind) if args.kind else None,
    )
    rows = rows[args.start :]
    if args.limit is not None:
        rows = rows[: args.limit]

    if args.dry_run:
        for idx, row in enumerate(rows, start=args.start):
            print(f"{idx:03d} {row['kind']} {row['slug']} -> {row['output']}")
        return

    summary_path = PROMPT_DIR / "codex-exec-image-results.jsonl"

    def write_result(idx: int, row: dict, status: str, generated_path: str, reason: str, report_path: str) -> None:
        with summary_path.open("a", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "index": idx,
                        "kind": row["kind"],
                        "slug": row["slug"],
                        "output": row["output"],
                        "status": status,
                        "generated_path": generated_path,
                        "reason": reason,
                        "report": report_path,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    for idx, row in enumerate(rows, start=args.start):
        print(f"generating {idx:03d} {row['kind']} {row['slug']} -> {row['output']}", flush=True)
        report_path = run_codex_exec(row, idx, timeout=args.timeout)
        accepted, generated_path, reason = parse_report(report_path)
        if not accepted or generated_path is None:
            write_result(idx, row, "rejected", "", reason, str(report_path))
            print(f"rejected {idx:03d}: {reason}", flush=True)
            continue
        output_path = ROOT / row["output"]
        copy_as_jpeg(generated_path, output_path)
        write_result(idx, row, "accepted", str(generated_path), reason, str(report_path))
        print(f"accepted {idx:03d}: {generated_path} -> {output_path}", flush=True)

    print(f"wrote results: {summary_path}")


if __name__ == "__main__":
    main()
