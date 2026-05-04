from __future__ import annotations

import argparse
import base64
import json
import os
import re
from pathlib import Path

from generate_editorial_images import ARTICLE_HEROES
from generate_project_covers import PROJECTS, ROOT


PROJECT_BY_SLUG = {project["slug"]: project for project in PROJECTS}
PROMPT_DIR = ROOT / ".content-refresh" / "image-prompts"
DEFAULT_MODEL = "gpt-image-2"
DEFAULT_SIZE = "2048x1152"
DEFAULT_QUALITY = "high"


PROMPT_GUIDANCE = {
    "use_case": "stylized-concept",
    "style": (
        "High-end editorial illustration for a portfolio website. Use a confident, polished "
        "visual language, but vary the medium, palette, texture, lighting, and camera treatment "
        "according to the project-specific art direction."
    ),
    "composition": (
        "16:9 landscape, generous safe margins, one dominant focal system, supporting "
        "secondary forms, strong negative space, balanced for card crops on desktop."
    ),
    "avoid": (
        "No readable text, no fake UI copy, no badges, no pills, no labels, no logos, "
        "no watermarks, no company names, no client names, no screenshots."
    ),
}


PROJECT_ART_DIRECTIONS = {
    "agentive-inventory": {
        "palette": "warm warehouse amber, muted sage, graphite, paper-white, signal red accents",
        "cover": "isometric supply-chain planning table made only of solid inventory blocks, smooth route ribbons, plain token discs, and constraint rails; no upright panels, no boards, no cards, no marks",
        "evidence": "close-up audit bench of forecast ribbons, budget constraint rails, stamped decision tokens, and traceable checkpoints",
        "article": "editorial scene of a calm control room where inventory flows are routed through human approval gates represented only by unmarked colored shapes",
        "medium": "tactile 3D paper-and-clay diorama with realistic shadows and material grain",
    },
    "creator-ai": {
        "palette": "electric cobalt, soft porcelain, graphite, restrained violet, warm task-light amber",
        "cover": "backstage orchestration machine with modular service bays, artifact trays, validation gates, and review lanes",
        "evidence": "macro view of interlocking workflow cassettes, deterministic checkpoints, replay cartridges, and observability threads",
        "article": "abstract theatre of staged decision-making: separate pools of light reveal discovery, retrieval, validation, and human review as distinct moments in a narrative sequence",
        "medium": "futurist product-industrial illustration with satin metal, translucent acrylic, and precise studio lighting",
    },
    "customer-segmentation": {
        "palette": "retail coral, teal, cream, graphite, citrus yellow, muted plum",
        "cover": "market mosaic of customer cohort islands and price-response contours arranged like a commercial strategy map",
        "evidence": "analytical pricing workbench with elasticity bands, constraint fences, segment tiles, and optimization paths",
        "article": "abstract retail shelf landscape where demand clusters bend around price and margin fields",
        "medium": "editorial collage mixing cut paper, ceramic tiles, and soft 3D surfaces",
    },
    "dengue-forecasting": {
        "palette": "tropical green, monsoon blue, fever coral, sand, dark botanical shadows",
        "cover": "public-health weather field with mosquito-risk currents, rainfall layers, intervention routes, and forecast horizons",
        "evidence": "policy decision map made from climate layers, intervention cost paths, warning thresholds, and horizon bands",
        "article": "ground-level field-surveillance scene in monsoon rain: water traps, abstract mosquito-vector traces, sampling vessels, forecast uncertainty ribbons, and intervention materials arranged as an editorial still life",
        "medium": "atmospheric environmental illustration with translucent map layers and scientific field-note texture",
    },
    "dspy-automotive-extractor": {
        "palette": "garage graphite, racing blue, steel, safety orange, cool white",
        "cover": "automotive engine cutaway made of language fragments passing through schema extraction channels",
        "evidence": "workbench of prompt strategy modules, compiled program components, smooth evaluation instruments with no ticks, and unmarked extraction slots",
        "article": "messy automotive signal scrap, complaint-noise fragments, and loose mechanical parts being sorted into clean validated evidence components through physical gauges with no markings",
        "medium": "technical product render with machined metal, enamel highlights, and controlled studio reflections",
    },
    "elliptic-gnn-project": {
        "palette": "midnight navy, turquoise, signal green, caution amber, deep violet",
        "cover": "forensic transaction map with wallet-node trails, suspicious flow clusters, separated illicit pathways, safe temporal boundaries, and calibrated risk halos",
        "evidence": "physical graph-evidence bench with blank node tokens, adjacency-matrix blocks, temporal split dividers, leakage-safe test fixtures, and calibration weights",
        "article": "noir audit scene where transaction paths cross strict leakage-safe boundary walls and suspicious clusters are isolated under forensic light",
        "medium": "cinematic data-art illustration with luminous particles, depth haze, and sharp geometric boundaries",
    },
    "hdb-resale-prices": {
        "palette": "Singapore concrete grey, sky blue, warm terracotta, garden green, cream",
        "cover": "urban housing model with layered feature surfaces, transit rings, and valuation gradients",
        "evidence": "architectural model desk with simplified housing blocks, input levers, confidence bands, and diagnostic cards",
        "article": "city-scale price landscape where practical user inputs narrow into an interpretable housing estimate",
        "medium": "architectural miniature photography style with tilt-shift depth and clean daylight",
    },
    "intelligent-content-analyzer": {
        "palette": "library green, parchment, ink black, cyan retrieval light, multilingual color glints",
        "cover": "document refinery where blank paper-like planes become retrieval routes, generation layers, and confidence checkpoints",
        "evidence": "service mesh of pure blank document slabs, punched paper shapes, retrieval beams, smooth evaluation lenses, and multilingual color prisms with no fine marks",
        "article": "quiet archive of blank paper fragments and abstract analysis panes transformed into a modular document-intelligence operating system, with perfectly unmarked surfaces",
        "medium": "premium editorial illustration combining archival paper texture, glass prisms, and soft volumetric light",
    },
    "learning-personalization-engine": {
        "palette": "chalk cream, learning blue, leaf green, soft orange, graphite",
        "cover": "personalized learning garden with anonymous event streams growing into mastery pathways and recommendation branches",
        "evidence": "teacher-review worktable with normalized event tokens, mastery-state layers, deterministic fixtures, and recommendation trays",
        "article": "learning journey map as a luminous pathway that adapts from evidence quality rather than guesswork",
        "medium": "warm editorial illustration with paper craft, chalk texture, and gentle classroom light",
    },
    "slidebench": {
        "palette": "laboratory white, ultraviolet, graphite, amber provenance lines, cool cyan",
        "cover": "artifact evaluation laboratory with blank artifact slabs, sealed bundles, and translucent specimens moving through retrieval and scoring instruments",
        "evidence": "forensic bench with blank artifact specimens, provenance threads, benchmark retrieval prisms, deterministic measurement rails, and review apparatus",
        "article": "abstract evaluation arena made only of smooth geometric artifact fragments, scoring lenses, provenance light paths, reliability balance forms, and judge-review shadows; no paper, no code specimens, no tables, no marks",
        "medium": "clean scientific editorial render with glass instruments, paper specimens, and controlled lab lighting",
    },
    "ml-trading-strategist": {
        "palette": "market black, electric cyan, brass, slate, muted green",
        "cover": "financial strategy arena with market-regime lanes, risk-return frontier arcs, buy-sell signal gates, abstract allocation blocks, and cost-friction barriers",
        "evidence": "quant research desk with abstract backtest lanes, cost slippage barriers, policy paths, and evaluation surfaces",
        "article": "market-regime landscape where rule-based, ensemble, and reinforcement-learning paths compete through changing terrain",
        "medium": "sleek financial editorial concept art with motion blur, polished metal, and luminous material bands",
    },
    "nlp-earnings-analyzer": {
        "palette": "paper ivory, financial green, ink navy, gold, charcoal",
        "cover": "earnings-report archive decomposed into semantic layers, sentiment currents, and topic constellations",
        "evidence": "financial signal material lab with blank statement slabs, unlabeled bar-like blocks, sentiment ribbons, topic clusters, and smooth corpus layers with no document marks",
        "article": "earnings-call intelligence scene with abstract audio waves, blank financial statement layers, sentiment separation fields, and metric material blocks without glyphs",
        "medium": "editorial printmaking mixed with subtle 3D paper relief and soft desk light",
    },
    "rag-engine-project": {
        "palette": "secure vault green, obsidian, laser cyan, parchment, muted lime",
        "cover": "private document vault connected to local retrieval machinery and source-evidence beams",
        "evidence": "close-up secure retrieval engine with blank document shards, vector-keyword channels, local model core, and feedback loops",
        "article": "archive cross-section showing an evidence chain: blank source plates, local retrieval channels, grounded answer path, and privacy boundary layers without vault portals",
        "medium": "cinematic security-tech illustration with vault textures, optical glass, and controlled glow",
    },
    "robo-advisor-project": {
        "palette": "deep finance blue, portfolio green, ivory, gold, risk red accents",
        "cover": "portfolio garden with risk-weather layers, objective compass, and adaptive allocation paths",
        "evidence": "advisory instrument table with risk-profile tiles, objective dials, scenario trails, and allocation terrain",
        "article": "portfolio tradeoff balance sculpture: allocation weights, client-risk frontier arcs, objective constraints, and shifting risk surfaces arranged without maps or compass symbols",
        "medium": "premium financial editorial render with cartographic texture and elegant instrument design",
    },
    "sentiment-analysis": {
        "palette": "night purple, neon pink, cobalt, white noise, warm signal yellow",
        "cover": "anonymous comment storm transformed through classifier prisms into sentiment bands and topic clouds",
        "evidence": "grounded classifier workbench with blank sample tiles, polarity color swatches, smooth threshold rails, review bins, and confidence material bands",
        "article": "anonymous social-comment observatory: hundreds of tiny blank speech-card tiles flowing into transformer-layer prisms, then separating into three distinct sentiment color fields and reviewable corpus clusters; no emoji shapes, no faces, no reaction icons, no play buttons, no media-platform symbols",
        "medium": "high-energy editorial data illustration with particles, prisms, and atmospheric depth",
    },
    "workforce-risk-intelligence": {
        "palette": "civic teal, alert amber, newsprint cream, graphite, muted red",
        "cover": "public-signal radar converting governed evidence streams into incidents, forecasts, and analyst-ready alerts",
        "evidence": "intelligence operations board made only of blank geometric blocks, source provenance streams, incident pipeline lanes, forecast horizon bands, alert thresholds, and review checkpoints",
        "article": "civic-risk lighthouse scanning public signals and turning noise into accountable workforce intelligence",
        "medium": "editorial civic-intelligence collage with radar light, paper texture, and disciplined grid structure",
    },
    "wet-bulb-temperature": {
        "palette": "humid cyan, heat orange, tropical green, slate cloud, pearl",
        "cover": "Singapore heat-stress atmosphere with humidity veils, temperature currents, and resilience thresholds",
        "evidence": "physical climate test chamber with smooth humidity veils, heat-gradient material sheets, a small anonymous city block model, wet-air vessels, and threshold color surfaces; no charts, no tick marks, no instrument labels",
        "article": "humid tropical city atmosphere visualized as wet-bulb risk fields and time-series currents",
        "medium": "atmospheric climate-science illustration with translucent weather layers and soft haze",
    },
    "longevity-lab": {
        "palette": "biomedical navy, cyan, oxygen blue, soft violet, clinical graphite",
        "cover": "abstract body-map health scenario field with organ-region energy surfaces, causal paths, and evidence gates",
        "evidence": "biomedical scenario console with model-card tiles, provenance threads, region-risk contours, and comparison levers",
        "article": "health-risk terrain where causal assumptions and evidence surfaces reshape a scenario map",
        "medium": "biomedical editorial concept art with luminous anatomy-inspired abstraction, no literal patient imagery",
    },
}


PROJECT_VISUAL_BRIEFS = {
    "agentive-inventory": {
        "subject": (
            "forecast-driven inventory planning system with reorder recommendations, approvals, "
            "budget guardrails, backtests, and audit trails"
        ),
        "context": (
            "The cloned repo is a human-in-the-loop inventory planning demo built around demand "
            "history, forecast generation, reorder guidance, approval decisions, and review surfaces."
        ),
        "metaphor": (
            "demand signals flowing into forecast layers, approval gates, budget constraints, "
            "and auditable recommendation cards"
        ),
        "evidence": (
            "forecast traces, reorder thresholds, approval checkpoints, budget guardrails, "
            "and audit-history artifacts"
        ),
    },
    "creator-ai": {
        "subject": (
            "backend-first learning-asset generation platform with intake, retrieval, "
            "generation, validation, export, and human review stages"
        ),
        "context": (
            "The cloned repo describes a multi-service operator platform with request intake, "
            "run orchestration, retrieval grounding, validation gates, export paths, contract-first "
            "APIs, replay tooling, and review workflows."
        ),
        "metaphor": (
            "a modular orchestration console represented by translucent service blocks, "
            "artifact cards, review gates, and observability traces"
        ),
        "evidence": (
            "workflow checkpoints, validation matrix, durable run artifacts, review queue, "
            "and API contract surfaces"
        ),
    },
    "customer-segmentation": {
        "subject": (
            "retail price optimization workflow combining customer segmentation, elasticity "
            "estimation, and constrained pricing recommendations"
        ),
        "context": (
            "The cloned repo turns anonymized transaction records into RFM-style segments, "
            "elasticity estimates, optimization constraints, and a staged commercial decision flow."
        ),
        "metaphor": (
            "customer cohorts, demand curves, constraint boundaries, and optimized price paths "
            "arranged as an analytical decision surface"
        ),
        "evidence": (
            "segment clusters, elasticity curves, optimization constraints, recommendation "
            "surfaces, and scenario comparison artifacts"
        ),
    },
    "dengue-forecasting": {
        "subject": (
            "dengue outbreak forecasting and intervention economics decision-support workflow"
        ),
        "context": (
            "The cloned repo combines public-health surveillance, weather, search-interest, and "
            "population inputs to forecast dengue risk and compare intervention economics."
        ),
        "metaphor": (
            "mosquito-borne risk signals, climate currents, forecast horizons, intervention "
            "paths, and policy tradeoff surfaces rendered as abstract environmental layers"
        ),
        "evidence": (
            "forecast bands, intervention economics, climate-risk gradients, and decision "
            "checkpoints rendered without people, beds, maps, or clinical imagery"
        ),
    },
    "dspy-automotive-extractor": {
        "subject": (
            "structured extraction benchmark for automotive complaint narratives and prompt "
            "optimization strategies"
        ),
        "context": (
            "The cloned repo defines extraction schemas, compares prompt/program strategies, "
            "tracks compiled artifacts, and evaluates make/model/year extraction performance."
        ),
        "metaphor": (
            "unstructured signal fragments passing through extraction circuits, schema gates, "
            "strategy branches, and evaluation traces"
        ),
        "evidence": (
            "schema fields, prompt strategy branches, compiled artifacts, metric traces, and "
            "comparison checkpoints"
        ),
    },
    "elliptic-gnn-project": {
        "subject": (
            "leakage-safe graph machine-learning workflow for illicit transaction detection"
        ),
        "context": (
            "The cloned private repo is represented as a public case study about graph learning, "
            "temporal splits, calibration, and operational precision metrics."
        ),
        "metaphor": (
            "transaction nodes and temporal graph layers moving through safe split boundaries, "
            "risk scoring, and calibration surfaces"
        ),
        "evidence": (
            "temporal graph partitions, precision checkpoints, calibration bands, and risk-score "
            "evidence trails"
        ),
    },
    "hdb-resale-prices": {
        "subject": (
            "Singapore public-housing resale price estimator narrowed to practical user inputs"
        ),
        "context": (
            "The cloned repo is a Streamlit price exploration and point-prediction app backed by "
            "research notebooks and user-facing feature choices."
        ),
        "metaphor": (
            "abstract public-housing blocks, feature layers, and regression confidence surfaces "
            "in an urban analytical scene without residents or people icons"
        ),
        "evidence": (
            "feature groups, prediction intervals, practical input paths, model diagnostics, and "
            "housing-market comparison surfaces without people, avatars, or pictograms"
        ),
    },
    "intelligent-content-analyzer": {
        "subject": (
            "document intelligence platform with upload, retrieval, generation, evaluation, and "
            "multilingual services"
        ),
        "context": (
            "The cloned repo separates document upload, hybrid retrieval, service orchestration, "
            "generation, confidence checks, and multilingual workflow support."
        ),
        "metaphor": (
            "documents entering a service mesh of retrieval indexes, generation layers, confidence "
            "checks, and multilingual routing"
        ),
        "evidence": (
            "document shards, retrieval paths, confidence signals, service boundaries, evaluation "
            "checkpoints, and multilingual routing traces"
        ),
    },
    "learning-personalization-engine": {
        "subject": (
            "learning-event normalization and personalization engine with mastery tracing "
            "and teacher-reviewable recommendations"
        ),
        "context": (
            "The cloned repo separates platform probes, normalized learning events, "
            "Bayesian-style mastery tracing, recommendation rules, deterministic fixtures, "
            "and teacher-facing review summaries."
        ),
        "metaphor": (
            "anonymous learning signals flowing into mastery-state layers and then into "
            "reviewable recommendation cards"
        ),
        "evidence": (
            "event contracts, Bayesian tracing paths, deterministic fixtures, recommendation "
            "inventory, and review summaries"
        ),
    },
    "slidebench": {
        "subject": (
            "artifact evaluation workbench for AI-generated decks, curricula, PDFs, and code "
            "bundles"
        ),
        "context": (
            "The cloned repo broadens beyond slide decks into artifact evaluation, with "
            "extraction, benchmark retrieval, deterministic metrics, provenance checks, "
            "durable jobs, budget controls, and judge-assisted reports."
        ),
        "metaphor": (
            "textless artifact materials passing through extraction prisms, retrieval fields, "
            "provenance threads, deterministic measurement layers, and review gates"
        ),
        "evidence": (
            "benchmark collections, provenance trails, measurement bands, run-budget controls, "
            "durable job paths, and review outcomes shown without UI panels or readable text"
        ),
    },
    "ml-trading-strategist": {
        "subject": (
            "trading strategy research platform comparing rule-based, supervised, and "
            "reinforcement-learning approaches"
        ),
        "context": (
            "The cloned repo supports benchmark, manual, tree-ensemble, and Q-learning strategies "
            "with cost-aware backtests and portfolio-level evaluation."
        ),
        "metaphor": (
            "parallel strategy lanes, regime ribbons, cost-friction gates, allocation fields, "
            "and risk-return surfaces rendered as abstract market materials"
        ),
        "evidence": (
            "strategy branches as smooth parallel ribbons, cost frictions as abstract gates, "
            "portfolio allocation as layered material bands, and comparison artifacts without "
            "node-link diagrams, network marks, people icons, or chart dashboards"
        ),
    },
    "nlp-earnings-analyzer": {
        "subject": (
            "financial-disclosure NLP workflow for earnings-report preprocessing, sentiment, "
            "topics, and reviewable analysis artifacts"
        ),
        "context": (
            "The cloned repo processes earnings-report text, creates cleaned datasets, runs NLP "
            "analysis, and exposes exploration through a dashboard-oriented workflow."
        ),
        "metaphor": (
            "financial documents decomposed into semantic layers, sentiment currents, topic "
            "clusters, and market-reaction evidence surfaces"
        ),
        "evidence": (
            "document preprocessing layers, sentiment flows, topic clusters, split artifacts, "
            "and review dashboards represented abstractly"
        ),
    },
    "rag-engine-project": {
        "subject": (
            "privacy-first enterprise document QA stack with local inference, hybrid retrieval, "
            "source snippets, and feedback loops"
        ),
        "context": (
            "The cloned repo loads a technical-document corpus, prebuilt indexes, retrieved "
            "source snippets, answer reasoning, and evaluation feedback for local-first QA."
        ),
        "metaphor": (
            "private document vaults connected to retrieval indexes, local model layers, source "
            "evidence fields, and feedback currents without thumbs, icons, or UI controls"
        ),
        "evidence": (
            "document shards, vector and keyword retrieval paths, source-evidence cards, local "
            "model boundaries, and feedback traces rendered without hand icons or node graphs"
        ),
    },
    "robo-advisor-project": {
        "subject": (
            "foundation-model portfolio advisory system with risk profiling and objective-aware "
            "investment recommendation logic"
        ),
        "context": (
            "The cloned repo combines risk profiling, dynamic portfolio objectives, strategy "
            "recommendations, and evaluation of portfolio behavior."
        ),
        "metaphor": (
            "risk-profile layers, objective dials, portfolio allocation paths, and scenario "
            "evaluation surfaces"
        ),
        "evidence": (
            "risk profiles, objective constraints, allocation paths, scenario traces, and "
            "portfolio evaluation artifacts"
        ),
    },
    "sentiment-analysis": {
        "subject": (
            "large-scale YouTube comment sentiment analysis pipeline from collection to "
            "transformer classification and dashboard exploration"
        ),
        "context": (
            "The cloned repo combines comment collection, processed corpora, transformer-based "
            "sentiment classification, corpus exploration, and notebook-derived research views."
        ),
        "metaphor": (
            "anonymous comment particles flowing into sentiment bands, topic clouds, classifier "
            "layers, and reviewable evidence surfaces without media-platform symbols"
        ),
        "evidence": (
            "comment streams, classifier confidence bands, sentiment clusters, corpus partitions, "
            "and research-summary artifacts without play buttons or social-media marks"
        ),
    },
    "workforce-risk-intelligence": {
        "subject": (
            "workforce risk intelligence system turning public signals into incidents, "
            "forecasts, alerts, and analyst review surfaces"
        ),
        "context": (
            "The cloned private repo is represented as a public case study about governed public "
            "source ingestion, incident pipelines, retrenchment forecasting, alerts, and dashboards."
        ),
        "metaphor": (
            "public signal streams passing through governance filters, incident abstractions, "
            "forecast layers, alert paths, and analyst review boundaries"
        ),
        "evidence": (
            "governed source traces, incident pipelines, forecast timelines, alert thresholds, "
            "and analyst-review evidence cards rendered as textless abstract layers"
        ),
    },
    "wet-bulb-temperature": {
        "subject": (
            "Singapore wet-bulb temperature analytics workflow for climate heat-stress "
            "interpretation"
        ),
        "context": (
            "The cloned repo combines ingestion, preprocessing, exploratory charts, regression, "
            "time-series interpretation, and a monthly climate-analysis app."
        ),
        "metaphor": (
            "tropical climate signal layers, humidity and temperature fields, heat-stress bands, "
            "and time-series analysis surfaces"
        ),
        "evidence": (
            "long-range climate series, wet-bulb thresholds, regression traces, heat-stress bands, "
            "and operational interpretation cards"
        ),
    },
    "longevity-lab": {
        "subject": (
            "health-risk scenario modeling platform with region-level risk surfaces, model cards, "
            "causal assumptions, and evidence-backed scoring"
        ),
        "context": (
            "The cloned repo and live demo show health scenario exploration with regional risk "
            "changes, runtime model artifacts, provenance, and reviewable evidence boundaries."
        ),
        "metaphor": (
            "health scenario layers represented by abstract biological signal fields, model-card "
            "tiles, causal paths, and evidence gates with no body silhouette or anatomy diagram"
        ),
        "evidence": (
            "regional risk surfaces, model cards, provenance traces, causal assumptions, "
            "and scenario comparison checkpoints without people, body maps, or organ diagrams"
        ),
    },
}


SENSITIVE_REPLACEMENTS = {
    "Elice": "the learning platform",
    "ELICE": "THE LEARNING PLATFORM",
    "elice": "learning-platform",
    "Singapore MOE": "a public education client",
    "SGMOE": "a public education client",
    "Ministry of Education": "a public education client",
}


def sanitize_context(text: str, *, max_chars: int) -> str:
    for source, replacement in SENSITIVE_REPLACEMENTS.items():
        text = text.replace(source, replacement)
    text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)
    text = re.sub(r"```.*?```", " ", text, flags=re.DOTALL)
    text = re.sub(r"!\[[^\]]*\]\([^)]+\)", " ", text)
    text = re.sub(r"!\[[^\]]*$", " ", text)
    text = re.sub(r"\[[^\]]+\]\([^)]+\)", " ", text)
    text = re.sub(r"https?://\S+", " ", text)
    text = re.sub(r"\$env:[A-Z0-9_]+=\"[^\"]*\"", " ", text)
    text = re.sub(r"`[^`]+`", " ", text)
    text = re.sub(r"\b[A-Z0-9_]{3,}\b", " ", text)
    text = re.sub(r"[#*_>|{}\[\]();]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def read_repo_context(project: dict, *, max_chars: int = 1600) -> str:
    repo_dir = project.get("repo_dir")
    if not repo_dir:
        return ""

    repo_root = ROOT / "repos" / repo_dir
    if not repo_root.exists():
        return ""

    candidate_names = [
        "README.md",
        "docs/README.md",
        "docs/architecture.md",
        "docs/ARCHITECTURE.md",
        "copilot-instructions.md",
    ]
    snippets: list[str] = []
    for name in candidate_names:
        path = repo_root / name
        if not path.exists() or not path.is_file():
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        text = " ".join(line.strip() for line in text.splitlines() if line.strip())
        if text:
            snippets.append(text[:max_chars])
        if len(" ".join(snippets)) >= max_chars:
            break

    return sanitize_context(" ".join(snippets), max_chars=max_chars)


def visual_brief(project: dict) -> dict[str, str]:
    fallback = {
        "subject": project["tagline"],
        "context": sanitize_context(read_repo_context(project, max_chars=700), max_chars=520),
        "metaphor": "layered system architecture with abstract data flows and operator-facing surfaces",
        "evidence": project.get("evidence_focus", project["tagline"]),
    }
    return PROJECT_VISUAL_BRIEFS.get(project["slug"], fallback)


def prompt_lines(
    *,
    asset_type: str,
    primary_request: str,
    subject: str,
    scene: str,
    composition: str,
    related_context: str,
    art_direction: dict[str, str],
    visual_role: str,
) -> list[str]:
    return [
        f"Use case: {PROMPT_GUIDANCE['use_case']}",
        f"Asset type: {asset_type}",
        f"Primary request: {primary_request}",
        f"Scene/backdrop: {scene}",
        f"Subject: {subject}",
        f"Visual role: {visual_role}",
        f"Style/medium: {PROMPT_GUIDANCE['style']} Project-specific medium: {art_direction['medium']}.",
        f"Composition/framing: {PROMPT_GUIDANCE['composition']} {composition}",
        "Lighting/mood: choose lighting that supports the project-specific medium; avoid repeating the same teal glass look across assets",
        f"Color palette: {art_direction['palette']}",
        "Materials/textures: use project-specific materials; do not default to generic translucent panels unless the art direction calls for it",
        "Text: none. No words, letters, numbers, typographic marks, readable symbols, UI labels, captions, badges, or pill text.",
        f"Context to preserve: {related_context}",
        f"Avoid: {PROMPT_GUIDANCE['avoid']}",
    ]


def project_prompt(project: dict) -> str:
    brief = visual_brief(project)
    art_direction = PROJECT_ART_DIRECTIONS[project["slug"]]
    return "\n".join(
        prompt_lines(
            asset_type="project cover card for a portfolio project page and project index",
            primary_request=f"Create a high-quality, non-literal visual for {project['title']}.",
            subject=art_direction["cover"],
            scene="project identity image, not a screenshot and not a generic dashboard",
            composition="Hero identity view with broad readable silhouette and enough breathing room for responsive crops.",
            related_context=f"{project['title']} is about {brief['subject']}. {brief['context']}",
            art_direction=art_direction,
            visual_role="Primary project identity image; it should be visually distinct from the evidence and article images.",
        )
    )


def evidence_prompt(project: dict) -> str:
    brief = visual_brief(project)
    art_direction = PROJECT_ART_DIRECTIONS[project["slug"]]
    return "\n".join(
        prompt_lines(
            asset_type="secondary project evidence card for an individual project page",
            primary_request=f"Create a distinct evidence-focused companion visual for {project['title']}.",
            subject=art_direction["evidence"],
            scene="implementation evidence image showing process, provenance, and technical grounding through abstract objects",
            composition="Closer and more concrete than the cover image; use a different camera angle, scale, and object arrangement.",
            related_context=f"Evidence focus: {brief['evidence']}. {brief['context']}",
            art_direction=art_direction,
            visual_role="Secondary evidence image; it should feel more implementation-grounded and materially different from the project cover.",
        )
    )


def article_prompt(hero: dict) -> str:
    project = PROJECT_BY_SLUG[hero["project_slug"]]
    brief = visual_brief(project)
    art_direction = PROJECT_ART_DIRECTIONS[project["slug"]]
    return "\n".join(
        prompt_lines(
            asset_type="article hero image for a technical writeup",
            primary_request=f"Create an editorial hero about this engineering idea: {hero['headline']}.",
            subject=(
                f"{art_direction['article']}; one focal metaphor for {hero['summary']} related to {project['title']}"
            ),
            scene="magazine-style article image that explains the essay thesis through metaphor, not through interface screens",
            composition=(
                "Single strong narrative focal metaphor. Use a noticeably different visual device, "
                "camera distance, and object family from both project images; avoid another machinery/workbench "
                "composition unless that is uniquely essential."
            ),
            related_context=f"Project context: {brief['subject']}. Repo context: {brief['context']}",
            art_direction=art_direction,
            visual_role="Article hero; it should read as a story or argument, not as the same project card repeated.",
        )
    )


def write_prompt_manifest(target_slugs: set[str] | None = None) -> Path:
    PROMPT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_path = PROMPT_DIR / "gpt-image-prompts.jsonl"
    rows: list[dict[str, str]] = []

    for project in PROJECTS:
        if target_slugs and project["slug"] not in target_slugs:
            continue
        rows.append(
            {
                "kind": "project-cover",
                "slug": project["slug"],
                "output": f"assets/images/project-covers/{project['slug']}.jpg",
                "mode": "codex-built-in image_gen; optional API fallback uses gpt-image-2",
                "model": DEFAULT_MODEL,
                "size": DEFAULT_SIZE,
                "quality": DEFAULT_QUALITY,
                "prompt": project_prompt(project),
            }
        )
        rows.append(
            {
                "kind": "project-evidence",
                "slug": project["slug"],
                "output": f"assets/images/project-evidence/{project['slug']}.jpg",
                "mode": "codex-built-in image_gen; optional API fallback uses gpt-image-2",
                "model": DEFAULT_MODEL,
                "size": DEFAULT_SIZE,
                "quality": DEFAULT_QUALITY,
                "prompt": evidence_prompt(project),
            }
        )

    for hero in ARTICLE_HEROES:
        project = PROJECT_BY_SLUG[hero["project_slug"]]
        if target_slugs and project["slug"] not in target_slugs:
            continue
        rows.append(
            {
                "kind": "article-hero",
                "slug": project["slug"],
                "output": f"assets/images/article-heroes/{hero['filename']}",
                "mode": "codex-built-in image_gen; optional API fallback uses gpt-image-2",
                "model": DEFAULT_MODEL,
                "size": DEFAULT_SIZE,
                "quality": DEFAULT_QUALITY,
                "prompt": article_prompt(hero),
            }
        )

    manifest_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n",
        encoding="utf-8",
    )
    return manifest_path


def generate_images(manifest_path: Path, *, model: str, size: str, limit: int | None = None) -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set; wrote prompts but cannot call the image API.")

    from openai import OpenAI

    client = OpenAI(api_key=api_key)
    rows = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]

    for row in rows:
        response = client.images.generate(
            model=model,
            prompt=row["prompt"],
            size=size,
            quality=DEFAULT_QUALITY,
            n=1,
        )
        encoded = response.data[0].b64_json
        if not encoded:
            raise RuntimeError(f"Image API returned no b64_json for {row['output']}")
        output_path = ROOT / row["output"]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_bytes(base64.b64decode(encoded))
        print(f"generated {row['kind']} {row['slug']} -> {row['output']}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare Codex-ready GPT Image portfolio prompts. The default workflow is Codex's "
            "built-in image_gen path; --api-generate is an optional API fallback."
        )
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--size", default=DEFAULT_SIZE)
    parser.add_argument("--slug", action="append", help="Limit to one or more project slugs.")
    parser.add_argument(
        "--api-generate",
        action="store_true",
        help="Optional fallback: call the OpenAI image API after writing prompts. Requires OPENAI_API_KEY.",
    )
    parser.add_argument("--generate", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--limit", type=int, help="Limit number of generated rows, useful for smoke tests.")
    args = parser.parse_args()

    manifest_path = write_prompt_manifest(set(args.slug) if args.slug else None)
    print(f"wrote prompt manifest: {manifest_path}")
    if args.api_generate or args.generate:
        generate_images(manifest_path, model=args.model, size=args.size, limit=args.limit)


if __name__ == "__main__":
    main()
