from __future__ import annotations

from pathlib import Path
from shutil import copyfile
import textwrap

from PIL import Image, ImageDraw, ImageFilter, ImageFont, ImageOps

from generate_project_covers import (
    PROJECTS,
    ROOT,
    crop_cover,
    font,
    hex_to_rgb,
    paste_panel,
    repo_panel,
    rounded_image,
    shadow_box,
)


ASSETS_DIR = ROOT / "assets" / "images"
PROJECT_COVERS_DIR = ASSETS_DIR / "project-covers"
ARTICLE_HERO_DIR = ASSETS_DIR / "article-heroes"
OUTPUT_DIR = ROOT / "output" / "playwright"
ARTICLE_SIZE = (1600, 900)
HOME_SIZE = (1400, 1400)
SIDEBAR_SIZE = (1400, 2000)

PROJECT_BY_SLUG = {project["slug"]: project for project in PROJECTS}

ARTICLE_HEROES = [
    {
        "filename": "2026-03-24-building-operational-workforce-risk-intelligence-from-public-signals.jpg",
        "project_slug": "workforce-risk-intelligence",
        "label": "Workforce intelligence / AI ops",
        "headline": "Public signals turned into operator-facing workforce risk intelligence",
        "summary": "An incident-centric monitoring system that connects governed public evidence, retrenchment forecasts, alerts, and dashboards.",
        "chips": ["Incident pipeline", "Forecast views", "Langfuse traces"],
    },
    {
        "filename": "2026-03-24-designing-creator-ai-as-a-backend-first-content-generation-platform.jpg",
        "project_slug": "creator-ai",
        "label": "Edtech / AI platform",
        "headline": "A backend-first platform for learning-asset generation",
        "summary": "A staged workflow for discovery, retrieval, generation, validation, review, and export with deterministic verification paths.",
        "chips": ["Workflow orchestration", "Quality gates", "BFF + services"],
    },
    {
        "filename": "2026-03-24-building-slidebench-to-evaluate-ai-generated-slides.jpg",
        "project_slug": "slidebench",
        "label": "Evaluation systems / multimodal",
        "headline": "Benchmarking AI-generated slides with retrieval and provenance checks",
        "summary": "A durable evaluation workflow that combines deterministic analytics, benchmark retrieval, and optional judge-assisted review.",
        "chips": ["Deterministic metrics", "Retrieval + judge", "Provenance"],
    },
    {
        "filename": "2026-03-24-building-service-oriented-document-intelligence.jpg",
        "project_slug": "intelligent-content-analyzer",
        "label": "Document intelligence / service architecture",
        "headline": "Service-oriented document intelligence for QA and summarization",
        "summary": "A document workflow that separates ingest, retrieval, generation, and evaluation into inspectable services.",
        "chips": ["Hybrid retrieval", "FastAPI services", "Multilingual"],
    },
    {
        "filename": "2026-03-23-human-in-the-loop-inventory-planning.jpg",
        "project_slug": "agentive-inventory",
        "label": "Operations systems / planning",
        "headline": "Human-in-the-loop inventory planning with guardrails",
        "summary": "A planning workflow that combines forecasting, approvals, procurement policy, and explainable operator review.",
        "chips": ["Approvals", "Forecasting", "Audit trails"],
    },
    {
        "filename": "2026-03-22-building-leakage-safe-graph-ml-for-illicit-transaction-detection.jpg",
        "project_slug": "elliptic-gnn-project",
        "label": "Graph ML / fraud detection",
        "headline": "Leakage-safe graph ML for illicit transaction detection",
        "summary": "An evaluation-heavy graph ML workflow built around temporal discipline, calibration, and investigation-budget metrics.",
        "chips": ["Temporal splits", "Precision@K", "Calibration"],
    },
    {
        "filename": "2025-06-24-robo-advisor-risk-profiling-portfolio-optimization.jpg",
        "project_slug": "robo-advisor-project",
        "label": "Foundation models / finance",
        "headline": "TabPFN risk models for portfolio construction",
        "summary": "A production robo-advisor that links investor profiling, regime-aware modeling, and objective-aware portfolio logic.",
        "chips": ["TabPFN", "9 objectives", "PyTorch + Streamlit"],
    },
    {
        "filename": "2025-06-18-forecasting-dengue-cases-and-cost-benefit-analysis.jpg",
        "project_slug": "dengue-forecasting",
        "label": "Forecasting / public health",
        "headline": "Operational dengue forecasting for intervention planning",
        "summary": "A forecasting workflow designed for policy use, with outbreak outlooks tied to intervention economics and planning tradeoffs.",
        "chips": ["16-week horizon", "9.5% MAPE", "Policy dashboard"],
    },
    {
        "filename": "2025-06-13-dspy-prompt-optimization-automotive-intelligence.jpg",
        "project_slug": "dspy-automotive-extractor",
        "label": "LLM evaluation / extraction",
        "headline": "DSPy prompt optimization as a measurable engineering loop",
        "summary": "A structured extraction system that treats prompt strategy as an experiment surface instead of ad hoc prompting.",
        "chips": ["51.33% F1", "Local LLMs", "Langfuse traces"],
    },
    {
        "filename": "2025-05-12-ml-trading-strategist-comparing-learning-approaches.jpg",
        "project_slug": "ml-trading-strategist",
        "label": "Finance / strategy research",
        "headline": "Comparing rule-based, supervised, and RL trading systems",
        "summary": "A research platform that benchmarks multiple trading paradigms with cost-aware backtests and portfolio-level outputs.",
        "chips": ["42.7% return", "1.48 Sharpe", "Backtesting"],
    },
    {
        "filename": "2025-05-09-nlp-earnings-report-analysis.jpg",
        "project_slug": "nlp-earnings-analyzer",
        "label": "Financial NLP",
        "headline": "Earnings report intelligence from sentiment to topic signals",
        "summary": "An analysis workflow for extracting sentiment, themes, and financial disclosure patterns from earnings reports.",
        "chips": ["BERTopic", "FinBERT-style methods", "Experiment tracking"],
    },
    {
        "filename": "2024-12-15-building-youtube-comment-sentiment-analyzer.jpg",
        "project_slug": "sentiment-analysis",
        "label": "Audience intelligence / NLP",
        "headline": "Large-scale YouTube comment analysis with transformer pipelines",
        "summary": "A full workflow from collection to sentiment analysis, surfacing audience signals across more than one hundred thousand comments.",
        "chips": ["114K comments", "Transformers", "Multi-page app"],
    },
    {
        "filename": "2024-10-29-building-effective-rag-systems.jpg",
        "project_slug": "rag-engine-project",
        "label": "Enterprise knowledge systems",
        "headline": "Local-first RAG systems built around retrieval quality",
        "summary": "A privacy-first QA stack for technical documents with hybrid retrieval, self-hosted inference, and production constraints.",
        "chips": ["Local Ollama", "FAISS", "Code-aware retrieval"],
    },
    {
        "filename": "2024-08-15-customer-segmentation-price-optimization.jpg",
        "project_slug": "customer-segmentation",
        "label": "Pricing / analytics",
        "headline": "From customer segmentation to constrained price optimization",
        "summary": "A pricing workflow that combines RFM segments, elasticity estimates, and optimization models for commercial action.",
        "chips": ["RFM + clustering", "Elasticity", "Gurobi"],
    },
    {
        "filename": "2023-06-18-predicting-hdb-resale-prices.jpg",
        "project_slug": "hdb-resale-prices",
        "label": "Housing / regression",
        "headline": "Singapore resale-price prediction narrowed to practical inputs",
        "summary": "A user-facing housing estimator designed around the questions real buyers can answer without complex data preparation.",
        "chips": ["0.9261 R^2", "Deployable app", "User-facing features"],
    },
    {
        "filename": "2023-05-15-predicting-heat-stress-with-wet-bulb-temperature.jpg",
        "project_slug": "wet-bulb-temperature",
        "label": "Climate analytics / resilience",
        "headline": "Heat-stress analysis through wet-bulb temperature modeling",
        "summary": "A climate analytics workflow focused on Singapore heat stress, long-range time series, and operational interpretation.",
        "chips": ["40+ years", "Streamlit platform", "Time-series views"],
    },
]

HOME_FLAGSHIPS = [
    "creator-ai",
    "intelligent-content-analyzer",
    "slidebench",
    "robo-advisor-project",
]

DEMO_IMAGE_SLUGS = [
    "agentive-inventory",
    "customer-segmentation",
    "dengue-forecasting",
    "dspy-automotive-extractor",
    "elliptic-gnn-project",
    "hdb-resale-prices",
    "intelligent-content-analyzer",
    "ml-trading-strategist",
    "nlp-earnings-analyzer",
    "rag-engine-project",
    "robo-advisor-project",
    "sentiment-analysis",
    "wet-bulb-temperature",
]


def save_rgb(image: Image.Image, path: Path, *, quality: int = 92) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rgb = Image.new("RGB", image.size, (244, 238, 229))
    rgb.paste(image, mask=image.split()[-1] if image.mode == "RGBA" else None)
    rgb.save(path, quality=quality)


def wrap(text: str, width: int) -> str:
    return textwrap.fill(text, width=width)


def fit_multiline_block(
    draw: ImageDraw.ImageDraw,
    text: str,
    *,
    origin: tuple[int, int],
    box_size: tuple[int, int],
    max_font_size: int,
    min_font_size: int,
    fill: tuple[int, int, int],
    serif: bool = False,
    bold: bool = False,
    line_spacing: int = 8,
    width_chars: int = 22,
) -> int:
    wrapped = wrap(text, width_chars)
    x, y = origin
    box_width, box_height = box_size

    for size in range(max_font_size, min_font_size - 1, -2):
        block_font = font(size, bold=bold, serif=serif)
        bbox = draw.multiline_textbbox((x, y), wrapped, font=block_font, spacing=line_spacing)
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        if width <= box_width and height <= box_height:
            draw.multiline_text((x, y), wrapped, font=block_font, fill=fill, spacing=line_spacing)
            return y + height

    fallback_font = font(min_font_size, bold=bold, serif=serif)
    draw.multiline_text((x, y), wrapped, font=fallback_font, fill=fill, spacing=line_spacing)
    bbox = draw.multiline_textbbox((x, y), wrapped, font=fallback_font, spacing=line_spacing)
    return y + (bbox[3] - bbox[1])


def add_grid(image: Image.Image, *, spacing: int = 64, alpha: int = 22) -> None:
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = image.size
    for x in range(0, width, spacing):
        draw.line((x, 0, x, height), fill=(255, 255, 255, alpha), width=1)
    for y in range(0, height, spacing):
        draw.line((0, y, width, y), fill=(255, 255, 255, alpha), width=1)
    image.alpha_composite(overlay)


def make_gradient(size: tuple[int, int], palette: list[str]) -> Image.Image:
    width, height = size
    top = hex_to_rgb(palette[0])
    middle = hex_to_rgb(palette[1])
    accent = hex_to_rgb(palette[2])
    image = Image.new("RGBA", size, top + (255,))
    pixels = image.load()

    for y in range(height):
        blend = y / max(height - 1, 1)
        if blend < 0.55:
            local = blend / 0.55
            rgb = tuple(int(top[idx] * (1 - local) + middle[idx] * local) for idx in range(3))
        else:
            local = (blend - 0.55) / 0.45
            rgb = tuple(
                int(middle[idx] * (1 - local) + accent[idx] * 0.16 * local + top[idx] * 0.04 * local)
                for idx in range(3)
            )
        for x in range(width):
            pixels[x, y] = rgb + (255,)

    overlay = Image.new("RGBA", size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    draw.ellipse((width - 520, -140, width + 180, 520), fill=accent + (40,))
    draw.ellipse((-260, height - 640, 320, height - 60), fill=(255, 255, 255, 18))
    draw.rectangle((0, 0, width, height), outline=(255, 255, 255, 18), width=2)
    image.alpha_composite(overlay)
    add_grid(image)
    return image


def image_panel(path_like: str | None, size: tuple[int, int], project: dict, *, compact: bool = False) -> Image.Image:
    if path_like:
        path = ROOT / path_like
        if path.exists():
            return crop_cover(path, size)
    return repo_panel(project, size, compact=compact)


def draw_chip_row(draw: ImageDraw.ImageDraw, origin: tuple[int, int], chips: list[str], accent: tuple[int, int, int]) -> None:
    x, y = origin
    chip_font = font(22, bold=True)
    for chip in chips:
        box = draw.textbbox((0, 0), chip, font=chip_font)
        width = box[2] - box[0] + 34
        height = box[3] - box[1] + 18
        draw.rounded_rectangle((x, y, x + width, y + height), radius=16, fill=accent + (255,))
        draw.text((x + 17, y + 8), chip, font=chip_font, fill=(15, 24, 31))
        x += width + 14


def compose_article(hero: dict) -> None:
    project = PROJECT_BY_SLUG[hero["project_slug"]]
    palette = project["palette"]
    accent = hex_to_rgb(palette[2])
    canvas = make_gradient(ARTICLE_SIZE, palette)

    copy_panel = Image.new("RGBA", (560, 748), (8, 16, 24, 112))
    draw = ImageDraw.Draw(copy_panel)
    draw.rounded_rectangle((0, 0, 559, 747), radius=34, fill=(8, 16, 24, 112), outline=(255, 255, 255, 20), width=2)
    copy_panel = copy_panel.filter(ImageFilter.GaussianBlur(0.4))
    paste_panel(canvas, copy_panel, (56, 78), radius=34)

    draw = ImageDraw.Draw(canvas)
    light = hex_to_rgb(palette[3])
    draw.text((98, 122), "TECHNICAL ARTICLE", font=font(22, bold=True), fill=light)
    draw.line((98, 156, 292, 156), fill=accent + (255,), width=4)
    draw.text((98, 178), hero["label"].upper(), font=font(17, serif=True), fill=(227, 228, 231))

    headline_bottom = fit_multiline_block(
        draw,
        hero["headline"],
        origin=(98, 236),
        box_size=(470, 250),
        max_font_size=54,
        min_font_size=38,
        fill=(255, 255, 255),
        serif=True,
        bold=True,
        line_spacing=8,
        width_chars=21,
    )
    summary_y = headline_bottom + 30
    summary_bottom = fit_multiline_block(
        draw,
        hero["summary"],
        origin=(98, summary_y),
        box_size=(430, 165),
        max_font_size=24,
        min_font_size=19,
        fill=(232, 236, 239),
        line_spacing=10,
        width_chars=25,
    )
    chip_y = min(max(summary_bottom + 28, 676), 690)
    draw_chip_row(draw, (98, chip_y), hero["chips"][:2], accent)

    primary = image_panel(project.get("primary"), (840, 610), project)
    paste_panel(canvas, primary, (684, 88), radius=30)

    secondary_path = project.get("secondary")
    if secondary_path:
        secondary = image_panel(secondary_path, (260, 156), project, compact=True)
        paste_panel(canvas, secondary, (1246, 664), radius=22)

    save_rgb(canvas, ARTICLE_HERO_DIR / hero["filename"])


def compose_home_hero() -> None:
    palette = ["#0d4148", "#082b31", "#ff8b2b", "#f5eee1"]
    accent = hex_to_rgb(palette[2])
    canvas = make_gradient(HOME_SIZE, palette)
    draw = ImageDraw.Draw(canvas)

    soft_panel = Image.new("RGBA", (580, 480), (8, 16, 24, 94))
    panel_draw = ImageDraw.Draw(soft_panel)
    panel_draw.rounded_rectangle((0, 0, 579, 479), radius=34, fill=(8, 16, 24, 94), outline=(255, 255, 255, 18), width=2)
    paste_panel(canvas, soft_panel, (84, 122), radius=34)

    draw.text((124, 170), "SELECTED SYSTEMS", font=font(24, bold=True), fill=(244, 239, 229))
    draw.line((124, 206, 350, 206), fill=accent + (255,), width=4)
    draw.multiline_text(
        (124, 248),
        "Selected\nsystems",
        font=font(98, bold=True, serif=True),
        fill=(255, 255, 255),
        spacing=6,
    )
    draw.multiline_text(
        (124, 488),
        "Case studies across foundation models,\nretrieval, evaluation, and operator-facing tools.",
        font=font(28),
        fill=(228, 236, 239),
        spacing=10,
    )

    positions = [
        ((720, 118), (420, 270), HOME_FLAGSHIPS[0]),
        ((914, 348), (360, 230), HOME_FLAGSHIPS[1]),
        ((676, 676), (420, 270), HOME_FLAGSHIPS[2]),
        ((940, 872), (330, 210), HOME_FLAGSHIPS[3]),
    ]
    for (x, y), size, slug in positions:
        project = PROJECT_BY_SLUG[slug]
        panel = image_panel(project.get("primary"), size, project)
        paste_panel(canvas, panel, (x, y), radius=28)

    save_rgb(canvas, ASSETS_DIR / "home-hero-flagship.jpg")


def compose_sidebar_background() -> None:
    palette = ["#0d4148", "#082b31", "#4fb1ba", "#f5eee1"]
    canvas = make_gradient(SIDEBAR_SIZE, palette)
    overlay = Image.new("RGBA", SIDEBAR_SIZE, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    width, height = SIDEBAR_SIZE

    draw.ellipse((width - 480, 80, width + 120, 680), fill=(255, 139, 43, 34))
    draw.ellipse((-240, height - 760, 340, height - 140), fill=(255, 255, 255, 20))
    draw.rounded_rectangle((84, 980, 1316, 1764), radius=56, fill=(7, 20, 24, 72), outline=(255, 255, 255, 14), width=2)
    canvas.alpha_composite(overlay)

    muted_covers = [
        PROJECT_COVERS_DIR / "robo-advisor-project.jpg",
        PROJECT_COVERS_DIR / "dspy-automotive-extractor.jpg",
        PROJECT_COVERS_DIR / "rag-engine-project.jpg",
    ]
    positions = [(164, 1010), (300, 1170), (132, 1330)]
    sizes = [(900, 392), (820, 352), (880, 372)]

    for path, position, size in zip(muted_covers, positions, sizes):
        if not path.exists():
            continue
        panel = crop_cover(path, size).filter(ImageFilter.GaussianBlur(1.5))
        dimmed = Image.new("RGBA", panel.size, (0, 0, 0, 0))
        dimmed.paste(panel, mask=rounded_image(panel, 32))
        tint = Image.new("RGBA", panel.size, (7, 20, 24, 132))
        dimmed.alpha_composite(tint)
        shadow = shadow_box(panel.size, 32)
        canvas.alpha_composite(shadow, (position[0] - 24, position[1] - 16))
        canvas.alpha_composite(rounded_image(dimmed, 32), position)

    save_rgb(canvas, ASSETS_DIR / "sidebar-bg.jpg")


def sync_demo_images() -> None:
    for slug in DEMO_IMAGE_SLUGS:
        source = PROJECT_COVERS_DIR / f"{slug}.jpg"
        destination = ASSETS_DIR / f"{slug}.jpg"
        if source.exists():
            copyfile(source, destination)


def refresh_profile_image() -> None:
    source = ASSETS_DIR / "logo.png"
    if not source.exists():
        return
    profile = Image.open(source).convert("RGB")
    profile = ImageOps.fit(profile, (720, 720), method=Image.LANCZOS)
    profile.save(ASSETS_DIR / "profile.jpg", quality=95)


def build_contact_sheet() -> None:
    thumbs = []
    for hero in ARTICLE_HEROES:
        path = ARTICLE_HERO_DIR / hero["filename"]
        if path.exists():
            thumbs.append((hero["filename"], Image.open(path).convert("RGB")))

    if not thumbs:
        return

    sheet = Image.new("RGB", (1750, 2300), (244, 238, 229))
    draw = ImageDraw.Draw(sheet)
    title_font = font(34, bold=True)
    label_font = font(18)
    draw.text((60, 44), "Editorial image review", font=title_font, fill=(19, 32, 34))

    x = 60
    y = 124
    card_width = 520
    card_height = 260
    gutter_x = 34
    gutter_y = 90
    count = 0

    for name, image in thumbs:
        thumb = ImageOps.fit(image, (card_width, card_height), method=Image.LANCZOS)
        sheet.paste(thumb, (x, y))
        draw.text((x, y + card_height + 16), name, font=label_font, fill=(48, 64, 66))
        count += 1
        if count % 3 == 0:
            x = 60
            y += card_height + gutter_y
        else:
            x += card_width + gutter_x

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    sheet.save(OUTPUT_DIR / "article-hero-contact-sheet.jpg", quality=92)


def main() -> None:
    sync_demo_images()
    for hero in ARTICLE_HEROES:
        compose_article(hero)
        print(f"generated article hero {hero['filename']}")
    compose_home_hero()
    print("generated home hero")
    compose_sidebar_background()
    print("generated sidebar background")
    refresh_profile_image()
    print("refreshed profile image")
    build_contact_sheet()
    print("generated contact sheet")


if __name__ == "__main__":
    main()
