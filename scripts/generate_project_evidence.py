from __future__ import annotations

from pathlib import Path
import textwrap

from PIL import Image, ImageDraw, ImageFilter

from generate_project_covers import (
    PROJECTS,
    ROOT,
    crop_cover,
    font,
    hex_to_rgb,
    paste_panel,
    repo_panel,
)


ASSETS_DIR = ROOT / "assets" / "images"
EVIDENCE_DIR = ASSETS_DIR / "project-evidence"
CANVAS_SIZE = (1600, 900)


def save_rgb(image: Image.Image, path: Path, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = Image.new("RGB", image.size, (246, 240, 231))
    rgb.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
    rgb.save(path, quality=quality)


def wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def make_background(project: dict) -> Image.Image:
    width, height = CANVAS_SIZE
    base = Image.new("RGBA", CANVAS_SIZE, (247, 242, 234, 255))
    draw = ImageDraw.Draw(base)
    accent = hex_to_rgb(project["palette"][2])
    deep = hex_to_rgb(project["palette"][1])

    glow = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    glow_draw.ellipse((1020, -80, 1640, 520), fill=accent + (36,))
    glow_draw.ellipse((-220, 560, 360, 1120), fill=deep + (22,))
    glow_draw.rounded_rectangle((34, 34, width - 34, height - 34), radius=34, outline=(18, 34, 36, 22), width=2)
    base.alpha_composite(glow)

    grid = Image.new("RGBA", CANVAS_SIZE, (0, 0, 0, 0))
    grid_draw = ImageDraw.Draw(grid)
    for x in range(58, width - 58, 56):
        grid_draw.line((x, 58, x, height - 58), fill=(18, 34, 36, 10), width=1)
    for y in range(58, height - 58, 56):
        grid_draw.line((58, y, width - 58, y), fill=(18, 34, 36, 10), width=1)
    base.alpha_composite(grid)
    return base


def image_panel(path_like: str | None, size: tuple[int, int], project: dict, *, compact: bool = False) -> Image.Image:
    if path_like:
        path = ROOT / path_like
        if path.exists():
            return crop_cover(path, size)
    return repo_panel(project, size, compact=compact)


def chip(draw: ImageDraw.ImageDraw, x: int, y: int, text: str, accent: tuple[int, int, int]) -> int:
    chip_font = font(18, bold=True)
    bbox = draw.textbbox((0, 0), text, font=chip_font)
    width = bbox[2] - bbox[0] + 28
    height = bbox[3] - bbox[1] + 16
    draw.rounded_rectangle((x, y, x + width, y + height), radius=15, fill=accent + (255,))
    draw.text((x + 14, y + 8), text, font=chip_font, fill=(16, 25, 30))
    return width


def insight_panel(project: dict, size: tuple[int, int]) -> Image.Image:
    panel = Image.new("RGBA", size, (13, 31, 36, 255))
    panel_draw = ImageDraw.Draw(panel)
    accent = hex_to_rgb(project["palette"][2])
    light = hex_to_rgb(project["palette"][3])

    texture = Image.new("RGBA", size, (0, 0, 0, 0))
    texture_draw = ImageDraw.Draw(texture)
    texture_draw.ellipse((size[0] - 180, -60, size[0] + 80, 180), fill=accent + (42,))
    texture_draw.ellipse((-100, size[1] - 180, 140, size[1] + 80), fill=(255, 255, 255, 18))
    for x in range(28, size[0] - 20, 34):
        texture_draw.line((x, 22, x, size[1] - 22), fill=(255, 255, 255, 12), width=1)
    panel.alpha_composite(texture)

    panel_draw.rounded_rectangle((16, 16, size[0] - 16, size[1] - 16), radius=24, outline=(255, 255, 255, 26), width=2)
    panel_draw.text((30, 28), "System focus", font=font(16, bold=True), fill=light)
    panel_draw.text((30, 58), project["label"].upper(), font=font(12, serif=True), fill=(224, 230, 235))
    panel_draw.multiline_text((30, 94), wrap(project["tagline"], 24), font=font(22, bold=True, serif=True), fill=(255, 255, 255), spacing=8)

    chip_x = 30
    for entry in project["chips"][:2]:
        chip_x += chip(panel_draw, chip_x, size[1] - 60, entry, accent) + 10

    return panel


def add_caption(draw: ImageDraw.ImageDraw, x: int, y: int, label: str, body: str) -> None:
    draw.text((x, y), label.upper(), font=font(16, bold=True), fill=(16, 34, 36))
    draw.text((x, y + 26), body, font=font(15), fill=(90, 110, 114))


def compose(project: dict) -> None:
    canvas = make_background(project)
    draw = ImageDraw.Draw(canvas)
    accent = hex_to_rgb(project["palette"][2])

    draw.text((74, 64), "PROJECT EVIDENCE", font=font(20, bold=True), fill=accent)
    draw.text((74, 98), project["title"], font=font(44, bold=True, serif=True), fill=(18, 33, 36))
    draw.text((74, 154), project["label"], font=font(18), fill=(96, 114, 118))

    primary = image_panel(project.get("primary"), (900, 520), project)
    secondary = image_panel(project.get("secondary"), (470, 240), project, compact=True)
    if project.get("secondary") is None:
        secondary = insight_panel(project, (470, 240))
    repo = repo_panel(project, (470, 240), compact=True)

    paste_panel(canvas, primary, (74, 212), radius=28)
    paste_panel(canvas, secondary, (1054, 212), radius=24)
    paste_panel(canvas, repo, (1054, 492), radius=24)

    add_caption(draw, 76, 752, "Product surface", "Representative UI or repo artifact from the actual implementation.")
    add_caption(draw, 1054, 752, "Supporting evidence", "Chart, system note, or artifact grounding the technical story.")
    add_caption(draw, 1054, 816, "Repo surface", "Top-level structure from the cloned repository.")

    chip_x = 76
    chip_y = 816
    for entry in project["chips"][:3]:
        chip_x += chip(draw, chip_x, chip_y, entry, accent) + 12

    save_rgb(canvas, EVIDENCE_DIR / f"{project['slug']}.jpg")


def main() -> None:
    for project in PROJECTS:
        compose(project)
        print(f"generated evidence {project['slug']}")


if __name__ == "__main__":
    main()
