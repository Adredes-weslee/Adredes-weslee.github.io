from __future__ import annotations

from pathlib import Path
import math
import random
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "assets" / "images" / "project-covers"
WIDTH = 1600
HEIGHT = 900
RNG = random.Random(42)


PROJECTS = [
    {
        "slug": "agentive-inventory",
        "title": "Agentive Inventory",
        "label": "AI ops / decision support",
        "tagline": "Forecast-driven planning with approvals, guardrails, and audit trails.",
        "chips": ["Human review", "FastAPI + Streamlit", "EOQ / ROP logic"],
        "palette": ["#0f2022", "#1d3e38", "#d4aa2a", "#f5eee1"],
        "motif": "bars",
        "repo_dir": "agentive-inventory",
        "primary": "output/project-sources/agentive-inventory-github.png",
    },
    {
        "slug": "creator-ai",
        "title": "Creator AI",
        "label": "Edtech / AI platform",
        "tagline": "Backend-first orchestration for learning-asset generation, validation, and review.",
        "chips": ["Workflow orchestration", "Quality gates", "Review queues"],
        "palette": ["#111723", "#213248", "#f0b55b", "#f5eee3"],
        "motif": "docs",
        "repo_dir": "creator-ai",
        "primary": "repos/creator-ai/docs/ui-refresh-assets/_after/home.png",
        "secondary": "repos/creator-ai/docs/ui-refresh-assets/_after/reviews.png",
    },
    {
        "slug": "customer-segmentation",
        "title": "Retail Price Optimization",
        "label": "Pricing / analytics",
        "tagline": "Segmentation, elasticity modeling, and constrained price optimization.",
        "chips": ["RFM + clustering", "Elasticity", "Gurobi"],
        "palette": ["#172033", "#214261", "#f6a13b", "#f8f1e4"],
        "motif": "rings",
        "repo_dir": "price-optimization",
        "primary": "output/project-sources/customer-segmentation-github.png",
        "secondary": "assets/images/Customer Segment Distribution.jpg",
    },
    {
        "slug": "dengue-forecasting",
        "title": "Dengue Forecasting",
        "label": "Forecasting / public health",
        "tagline": "Operational dengue forecasting paired with intervention economics.",
        "chips": ["16-week horizon", "9.5% MAPE", "DALY comparison"],
        "palette": ["#1a2726", "#274d43", "#f46c4f", "#f7f1e6"],
        "motif": "grid",
        "repo_dir": "Dengue-Case-Prediction-and-Cost-Benefits-Analysis",
        "primary": "output/project-sources/dengue-forecasting-github.png",
    },
    {
        "slug": "dspy-automotive-extractor",
        "title": "DSPy Automotive Extractor",
        "label": "LLM evaluation / extraction",
        "tagline": "Structured extraction experiments that benchmark prompt optimization.",
        "chips": ["51.33% F1", "Local LLMs", "Langfuse traces"],
        "palette": ["#121928", "#20304c", "#66b0ff", "#f5efe5"],
        "motif": "grid",
        "repo_dir": "dspy-automotive-extractor",
        "primary": "output/project-sources/dspy-automotive-extractor-github.png",
    },
    {
        "slug": "elliptic-gnn-project",
        "title": "Elliptic Graph ML",
        "label": "Graph ML / fraud detection",
        "tagline": "Leakage-safe illicit transaction detection with operational metrics.",
        "chips": ["Temporal splits", "Precision@K", "Calibration"],
        "palette": ["#101726", "#1f2c48", "#4bd2b8", "#f5efe5"],
        "motif": "network",
        "repo_dir": "elliptic-gnn-project",
    },
    {
        "slug": "hdb-resale-prices",
        "title": "HDB Resale Predictor",
        "label": "Housing / regression",
        "tagline": "A Singapore housing estimator narrowed to practical user inputs.",
        "chips": ["0.9261 R^2", "Deployable app", "User-facing features"],
        "palette": ["#1f2328", "#394048", "#f0b04d", "#f6f0e6"],
        "motif": "bars",
        "repo_dir": "Making-Predictions-on-HDB-Resale-Price",
        "primary": "output/project-sources/hdb-resale-prices-github.png",
    },
    {
        "slug": "intelligent-content-analyzer",
        "title": "Intelligent Content Analyzer",
        "label": "Document intelligence / RAG",
        "tagline": "Multilingual upload, retrieval, generation, and evaluation services.",
        "chips": ["FastAPI services", "Hybrid retrieval", "Multilingual"],
        "palette": ["#101d1f", "#214447", "#f1c56a", "#f4efe5"],
        "motif": "docs",
        "repo_dir": "intelligent-content-analyzer",
        "primary": "output/project-sources/intelligent-content-analyzer-github.png",
    },
    {
        "slug": "slidebench",
        "title": "SlideBench",
        "label": "Evaluation / multimodal",
        "tagline": "Benchmarking AI-generated slides with retrieval, provenance, and judge-assisted scoring.",
        "chips": ["Deterministic metrics", "Retrieval + judge", "Provenance"],
        "palette": ["#151a24", "#25404f", "#f18c52", "#f5eee4"],
        "motif": "grid",
        "repo_dir": "Benchmarking-AI-Generated-Slides",
        "primary": "repos/Benchmarking-AI-Generated-Slides/docs/screenshots/report.png",
        "secondary": "repos/Benchmarking-AI-Generated-Slides/docs/screenshots/evaluate.png",
    },
    {
        "slug": "ml-trading-strategist",
        "title": "ML Trading Strategist",
        "label": "Finance / strategy research",
        "tagline": "Comparative platform for rule-based, tree-based, and RL trading.",
        "chips": ["42.7% return", "1.48 Sharpe", "Cost-aware backtests"],
        "palette": ["#111722", "#24324c", "#73d4ff", "#f3eee3"],
        "motif": "bars",
        "repo_dir": "ML-Trading-Strategist",
        "primary": "output/project-sources/ml-trading-strategist-github.png",
        "secondary": "assets/images/Strategy Performance Comparison.jpg",
    },
    {
        "slug": "nlp-earnings-analyzer",
        "title": "Earnings Report Intelligence",
        "label": "Financial NLP",
        "tagline": "Sentiment, topic, and disclosure analysis across earnings reports.",
        "chips": ["FinBERT-style methods", "BERTopic", "Versioned experiments"],
        "palette": ["#16202b", "#223f4a", "#ff8e4b", "#f5efe3"],
        "motif": "docs",
        "repo_dir": "NLP_earnings_report",
        "primary": "output/project-sources/nlp-earnings-analyzer-github.png",
    },
    {
        "slug": "rag-engine-project",
        "title": "Custom RAG Engine",
        "label": "Enterprise QA / local-first",
        "tagline": "Privacy-first document QA with local inference and GPU-aware retrieval.",
        "chips": ["Local Ollama", "FAISS", "Code-aware retrieval"],
        "palette": ["#10141f", "#1c2a41", "#a7e156", "#f5eee2"],
        "motif": "docs",
        "repo_dir": "Custom-RAG-Engine-for-Enterprise-Document-QA",
        "primary": "output/project-sources/rag-engine-project.png",
        "secondary": "output/project-sources/rag-engine-project-github.png",
    },
    {
        "slug": "robo-advisor-project",
        "title": "AI Portfolio Advisory",
        "label": "Foundation models / fintech",
        "tagline": "Risk profiling and objective-aware portfolio optimization.",
        "chips": ["TabPFN", "DQN", "9 objectives"],
        "palette": ["#131824", "#20344f", "#f0b253", "#f6efe4"],
        "motif": "rings",
        "repo_dir": "Using_Artificial_Intelligence_to_Develop_a_Robo_Advisor",
        "primary": "output/project-sources/robo-advisor-project-github.png",
        "secondary": "repos/Using_Artificial_Intelligence_to_Develop_a_Robo_Advisor/data/output/evaluation_balanced.png",
    },
    {
        "slug": "sentiment-analysis",
        "title": "YouTube Sentiment Analysis",
        "label": "Audience intelligence / NLP",
        "tagline": "Large-scale comment analysis from collection pipeline to dashboard.",
        "chips": ["114K comments", "Transformers", "Multi-page app"],
        "palette": ["#151726", "#223553", "#ff5d6c", "#f7f0e6"],
        "motif": "grid",
        "repo_dir": "Sentiment-Analysis-and-NLP-for-a-Youtube-Video",
        "primary": "output/project-sources/sentiment-analysis.png",
        "secondary": "output/project-sources/sentiment-analysis-github.png",
    },
    {
        "slug": "workforce-risk-intelligence",
        "title": "Workforce Risk Intelligence",
        "label": "AI ops / workforce intelligence",
        "tagline": "Incident-centric monitoring, forecasting, and alerting from governed public signals.",
        "chips": ["Incident pipeline", "Retrenchment forecast", "Next.js + FastAPI"],
        "palette": ["#0f2022", "#1f3e41", "#d4aa2a", "#f5eee2"],
        "motif": "network",
        "repo_dir": "ntuc-workforce-intel",
    },
    {
        "slug": "wet-bulb-temperature",
        "title": "Wet-Bulb Temperature",
        "label": "Climate analytics / resilience",
        "tagline": "Heat-stress analysis for Singapore through wet-bulb modeling.",
        "chips": ["40+ years", "Streamlit platform", "Time-series views"],
        "palette": ["#162028", "#235268", "#81d0d4", "#f4efe5"],
        "motif": "waves",
        "repo_dir": "Data-Analysis-of-Wet-Bulb-Temperature",
        "primary": "output/project-sources/wet-bulb-temperature-github.png",
        "secondary": "repos/Data-Analysis-of-Wet-Bulb-Temperature/data/output/wet_bulb_time_series.png",
    },
]


def font(size: int, *, bold: bool = False, serif: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates: list[str]
    if serif and bold:
        candidates = [
            "C:/Windows/Fonts/georgiab.ttf",
            "C:/Windows/Fonts/timesbd.ttf",
        ]
    elif serif:
        candidates = [
            "C:/Windows/Fonts/georgia.ttf",
            "C:/Windows/Fonts/times.ttf",
        ]
    elif bold:
        candidates = [
            "C:/Windows/Fonts/seguisb.ttf",
            "C:/Windows/Fonts/bahnschrift.ttf",
            "C:/Windows/Fonts/arialbd.ttf",
        ]
    else:
        candidates = [
            "C:/Windows/Fonts/segoeui.ttf",
            "C:/Windows/Fonts/consola.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]

    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


TITLE_FONT = lambda: font(72, bold=True)
LABEL_FONT = lambda: font(22, bold=True)
BODY_FONT = lambda: font(28)
CHIP_FONT = lambda: font(22, bold=True)
META_FONT = lambda: font(18, serif=True)
CODE_FONT = lambda: font(20)


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.lstrip("#")
    return tuple(int(value[i : i + 2], 16) for i in (0, 2, 4))


def gradient_background(colors: list[str]) -> Image.Image:
    top = hex_to_rgb(colors[0])
    mid = hex_to_rgb(colors[1])
    accent = hex_to_rgb(colors[2])
    img = Image.new("RGBA", (WIDTH, HEIGHT), top + (255,))
    pixels = img.load()

    for y in range(HEIGHT):
        blend = y / (HEIGHT - 1)
        if blend < 0.55:
            local = blend / 0.55
            rgb = tuple(int(top[i] * (1 - local) + mid[i] * local) for i in range(3))
        else:
            local = (blend - 0.55) / 0.45
            rgb = tuple(int(mid[i] * (1 - local) + accent[i] * 0.12 * local + top[i] * 0.06 * local) for i in range(3))
        for x in range(WIDTH):
            pixels[x, y] = rgb + (255,)

    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((1030, -120, 1580, 380), fill=hex_to_rgb(colors[2]) + (24,))
    draw.ellipse((-220, 430, 360, 1040), fill=(255, 255, 255, 14))
    draw.rectangle((0, 0, WIDTH, HEIGHT), outline=(255, 255, 255, 12), width=2)
    return Image.alpha_composite(img, overlay)


def draw_motif(base: Image.Image, motif: str, accent: tuple[int, int, int], *, local: bool = False) -> None:
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = base.size

    if local:
        x_start = 24
        x_end = width - 24
        y_start = 24
        y_end = height - 24
        alpha = 34
    else:
        x_start = 760
        x_end = 1510
        y_start = 104
        y_end = 770
        alpha = 24

    if motif == "network":
        nodes = []
        for _ in range(16):
            x = RNG.randint(x_start + 40, x_end - 40)
            y = RNG.randint(y_start + 40, y_end - 40)
            nodes.append((x, y))
        for start in nodes:
            for end in RNG.sample(nodes, k=3):
                draw.line((start, end), fill=accent + (18,), width=2)
        for x, y in nodes:
            r = RNG.randint(8, 14)
            draw.ellipse((x - r, y - r, x + r, y + r), fill=accent + (alpha + 18,))
    elif motif == "docs":
        for idx in range(7):
            x = x_start + 90 + idx * 42
            y = y_start + idx * 12
            draw.rounded_rectangle((x, y, x + 160, y + 220), radius=18, outline=(255, 255, 255, 12), fill=accent + (10,))
            for line in range(5):
                line_y = y + 30 + line * 30
                draw.rounded_rectangle((x + 20, line_y, x + 124, line_y + 8), radius=4, fill=(255, 255, 255, 12))
    elif motif == "bars":
        baseline = y_end - 30
        for idx in range(14):
            x = x_start + 24 + idx * 38
            bar_height = 90 + (idx % 5) * 50
            draw.rounded_rectangle((x, baseline - bar_height, x + 24, baseline), radius=10, fill=accent + (alpha,))
    elif motif == "waves":
        for row in range(6):
            points = []
            for step in range(20):
                x = x_start + step * max(18, (x_end - x_start) // 20)
                y = y_start + 44 + row * 38 + math.sin((step + row) * 0.65) * 15
                points.append((x, y))
            draw.line(points, fill=accent + (alpha + 8,), width=4)
    elif motif == "rings":
        center_x = (x_start + x_end) // 2
        center_y = (y_start + y_end) // 2
        scale = 0.58 if local else 1
        for idx, radius in enumerate((300, 235, 170, 110)):
            radius = int(radius * scale)
            box = (center_x - radius, center_y - radius, center_x + radius, center_y + radius)
            draw.arc(box, start=18 * idx, end=286 + 14 * idx, fill=accent + (alpha + 12,), width=10)
    else:
        for col in range(8):
            x = x_start + col * max(28, (x_end - x_start) // 8)
            draw.line((x, y_start, x, y_end), fill=(255, 255, 255, 10), width=2)
        for row in range(8):
            y = y_start + row * max(24, (y_end - y_start) // 8)
            draw.line((x_start, y, x_end, y), fill=accent + (10,), width=2)

    base.alpha_composite(overlay)


def crop_cover(image_path: Path, size: tuple[int, int]) -> Image.Image:
    image = Image.open(image_path).convert("RGBA")
    return ImageOps.fit(image, size, method=Image.LANCZOS)


def rounded_image(image: Image.Image, radius: int) -> Image.Image:
    mask = Image.new("L", image.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, image.size[0], image.size[1]), radius=radius, fill=255)
    rounded = Image.new("RGBA", image.size, (0, 0, 0, 0))
    rounded.paste(image, mask=mask)
    return rounded


def shadow_box(size: tuple[int, int], radius: int) -> Image.Image:
    shadow = Image.new("RGBA", (size[0] + 48, size[1] + 48), (0, 0, 0, 0))
    draw = ImageDraw.Draw(shadow)
    draw.rounded_rectangle((24, 24, size[0] + 24, size[1] + 24), radius=radius, fill=(0, 0, 0, 82))
    return shadow.filter(ImageFilter.GaussianBlur(18))


def draw_chip(draw: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, fill: tuple[int, int, int], text_fill: tuple[int, int, int]) -> int:
    bbox = draw.textbbox((0, 0), text, font=CHIP_FONT())
    width = bbox[2] - bbox[0] + 36
    height = bbox[3] - bbox[1] + 20
    x, y = xy
    draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill=fill)
    draw.text((x + 18, y + 9), text, font=CHIP_FONT(), fill=text_fill)
    return width


def draw_text_block(base: Image.Image, project: dict) -> None:
    draw = ImageDraw.Draw(base)
    light = hex_to_rgb(project["palette"][3])
    accent = hex_to_rgb(project["palette"][2])

    draw.text((90, 86), "PROJECT CASE STUDY", font=LABEL_FONT(), fill=light)
    draw.line((90, 118, 254, 118), fill=accent + (255,), width=3)
    draw.text((90, 142), project["label"].upper(), font=META_FONT(), fill=(230, 223, 210))

    title_lines = textwrap.fill(project["title"], width=18)
    draw.multiline_text((90, 214), title_lines, font=TITLE_FONT(), fill=(255, 255, 255), spacing=6)

    tagline = textwrap.fill(project["tagline"], width=38)
    draw.multiline_text((90, 432), tagline, font=BODY_FONT(), fill=(229, 231, 235), spacing=10)

    chip_y = 630
    chip_x = 90
    chip_fill = accent + (255,)
    chip_text = (15, 24, 31)
    for chip in project["chips"]:
        chip_x += draw_chip(draw, (chip_x, chip_y), chip, chip_fill, chip_text) + 16


def paste_panel(base: Image.Image, panel: Image.Image, xy: tuple[int, int], radius: int) -> None:
    shadow = shadow_box(panel.size, radius)
    base.alpha_composite(shadow, (xy[0] - 24, xy[1] - 16))
    base.alpha_composite(rounded_image(panel, radius), xy)


def repo_entries(project: dict) -> list[str]:
    repo_dir = project.get("repo_dir")
    if not repo_dir:
        return []
    repo_root = ROOT / "repos" / repo_dir
    if not repo_root.exists():
        return []

    entries = []
    for child in sorted(repo_root.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if child.name.startswith(".git"):
            continue
        entries.append(f"/{child.name}" if child.is_dir() else child.name)
        if len(entries) >= 7:
            break
    return entries


def repo_panel(project: dict, size: tuple[int, int], *, compact: bool = False) -> Image.Image:
    panel = Image.new("RGBA", size, (14, 21, 28, 255))
    accent = hex_to_rgb(project["palette"][2])
    draw_motif(panel, project["motif"], accent, local=True)
    draw = ImageDraw.Draw(panel)
    draw.rounded_rectangle((22, 22, size[0] - 22, size[1] - 22), radius=22, outline=(255, 255, 255, 28), width=2)

    pad_x = 32 if compact else 42
    title_y = 26 if compact else 36
    draw.text((pad_x, title_y), "Repo map" if compact else "Cloned repo surface", font=(font(16, bold=True) if compact else LABEL_FONT()), fill=(239, 233, 224))
    draw.text((pad_x, title_y + (24 if compact else 36)), project.get("repo_dir", project["slug"]), font=(font(12, serif=True) if compact else META_FONT()), fill=(220, 225, 231))

    entries = repo_entries(project)
    start_y = 78 if compact else 126
    step = 20 if compact else 38
    entry_font = font(14 if compact else 20)
    max_items = 3 if compact else 6
    for idx, entry in enumerate(entries[:max_items]):
        draw.text((pad_x, start_y + idx * step), entry, font=entry_font, fill=(250, 250, 250))

    footer_y = size[1] - (58 if compact else 88)
    chip_line = " | ".join(project["chips"][:2])
    draw.text((pad_x, footer_y), chip_line, font=(font(12, bold=True) if compact else font(18, bold=True)), fill=(255, 255, 255))
    remainder = len(entries) - min(len(entries), max_items)
    footer_text = f"+ {remainder} more top-level entries" if remainder > 0 else project["label"].title()
    draw.text((pad_x, footer_y + (16 if compact else 30)), footer_text, font=(font(10, serif=True) if compact else font(14, serif=True)), fill=(220, 225, 231))
    return panel


def optional_panel(path_like: str | None, size: tuple[int, int], project: dict) -> Image.Image | None:
    if not path_like:
        return None
    path = ROOT / path_like
    if not path.exists():
        return repo_panel(project, size, compact=size[0] < 400)
    return crop_cover(path, size)


def compose(project: dict) -> None:
    canvas = gradient_background(project["palette"])
    accent = hex_to_rgb(project["palette"][2])
    draw_motif(canvas, project["motif"], accent)
    draw_text_block(canvas, project)

    primary = optional_panel(project.get("primary"), (690, 510), project)
    if primary is None:
        primary = repo_panel(project, (690, 510))
    paste_panel(canvas, primary, (826, 124), radius=28)

    secondary = optional_panel(project.get("secondary"), (280, 190), project)
    if secondary is None:
        secondary = repo_panel(project, (280, 190), compact=True)
    paste_panel(canvas, secondary, (1156, 536), radius=24)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rgb = Image.new("RGB", canvas.size, (245, 239, 230))
    rgb.paste(canvas, mask=canvas.split()[-1])
    rgb.save(OUTPUT_DIR / f"{project['slug']}.jpg", quality=92)


def main() -> None:
    for project in PROJECTS:
        compose(project)
        print(f"generated {project['slug']}")


if __name__ == "__main__":
    main()
