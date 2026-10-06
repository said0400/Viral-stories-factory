# src/test_images.py
"""Standalone image-pipeline test (no triage, no article writing, no publishing, no history changes).

Run from the repository root:

  # 1) Cloudflare only, text-to-image (cheapest, checks credentials/model/timeout)
  python -m src.test_images --step raw

  # 2) Restyle of a source photo (article hero, with reference)
  python -m src.test_images --step restyle --image-url https://example.com/photo.jpg --title "Cat on a roof"

  # 3) Facebook composite from original photos (no image generation, uses vision LLM only)
  python -m src.test_images --step facebook --image-url URL1 --image-url URL2 --image-url URL3 --layout all

  # 4) Full ImageGenerator.generate() (analysis + hero + facebook + quality checks)
  python -m src.test_images --step full --image-url URL1 --image-url URL2 --title "..." --scene "..."

  # everything
  python -m src.test_images --step all --image-url URL1 --image-url URL2

Outputs are saved to data/test_images/ (override with --out).
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import sys
import time
import traceback
from pathlib import Path

from PIL import Image

from . import visual_analyzer
from .cloudflare_client import CloudflareClient
from .config import Settings
from .gemini_client import GeminiClient, GeminiError, ImageGenError, ImageQuotaError
from .groq_client import GroqClient
from .image_generator import (
    REALISTIC_NEWS_STYLE,
    ImageGenerator,
    _faithful_prompt,
    crop_to_aspect,
)
from .llm_router import LLMRouter
from .logger import error, log, setup_logging, warn
from .models import SourceArticle
from .photo_composer import SQUARE_LAYOUTS, analyze_source_photos, compose_original_square, fetch_photos
from .utils import PoliteFetcher, utcnow

DEFAULT_PROMPT = (
    "A tabby cat sitting on a rooftop at sunrise, "
    f"{REALISTIC_NEWS_STYLE}. No text, no watermark."
)


# ------------------------------------------------------------------ helpers
def save_bytes(out: Path, name: str, data: bytes) -> Path:
    path = out / name
    path.write_bytes(data)
    try:
        with Image.open(io.BytesIO(data)) as im:
            log("TEST", f"saved {path}  ({im.size[0]}x{im.size[1]}, {len(data) / 1024:.0f} KB)")
    except Exception:
        log("TEST", f"saved {path}  ({len(data) / 1024:.0f} KB, could not decode)")
    return path


def save_image(out: Path, name: str, img: Image.Image) -> Path:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=90)
    return save_bytes(out, name, buf.getvalue())


def build_article(args: argparse.Namespace) -> SourceArticle:
    urls = [u.strip() for u in (args.image_url or []) if u.strip()]
    return SourceArticle(
        source_name="test",
        original_title=args.title,
        original_url=args.article_url or (urls[0] if urls else "https://example.com/test"),
        normalized_url=args.article_url or (urls[0] if urls else "https://example.com/test"),
        description=args.scene or args.title,
        article_text=args.scene or args.title,
        main_image_url=urls[0] if urls else "",
        additional_image_urls=urls[1:],
        discovered_at=utcnow(),
    )


def need_urls(args: argparse.Namespace, step: str) -> bool:
    if args.image_url:
        return True
    warn("TEST", f"step '{step}' needs at least one --image-url; skipped")
    return False


# ------------------------------------------------------------------ steps
def step_raw(cfg: Settings, args: argparse.Namespace, out: Path) -> bool:
    """Cloudflare text-to-image only."""
    cf = CloudflareClient(cfg)
    started = time.time()
    data, mime = cf.generate_image(args.prompt or DEFAULT_PROMPT, aspect_ratio="16:9", tag="TEST_RAW")
    log("TEST", f"raw generation OK in {time.time() - started:.1f}s ({mime})")
    save_bytes(out, "1_raw_cloudflare.jpg", data)
    return True


def step_restyle(cfg: Settings, args: argparse.Namespace, out: Path, fetcher: PoliteFetcher) -> bool:
    """Cloudflare image generation with a source photo as reference (article hero)."""
    if not need_urls(args, "restyle"):
        return False

    article = build_article(args)
    ref = visual_analyzer.acquire_source_image(article, fetcher)
    if not ref:
        raise ImageGenError("could not download a usable source image from --image-url")

    save_bytes(out, "2_source_reference.jpg", ref[0])
    cropped = crop_to_aspect((ref[0], ref[1]), "16:9")
    prompt = _faithful_prompt(
        cfg.cinematic_style, "16:9", False, title=args.title, scene_idea=args.scene
    )
    log("TEST", f"restyle prompt ({len(prompt)} chars): {prompt[:160]}...")

    cf = CloudflareClient(cfg)
    started = time.time()
    data, mime = cf.generate_image(prompt, references=[cropped], aspect_ratio="16:9", tag="TEST_RESTYLE")
    log("TEST", f"restyle OK in {time.time() - started:.1f}s ({mime})")
    save_bytes(out, "2_restyled_hero.jpg", data)
    return True


def step_facebook(
    cfg: Settings, args: argparse.Namespace, out: Path, fetcher: PoliteFetcher, llm: LLMRouter
) -> bool:
    """Original-photo Facebook composite. No image generation is used here."""
    if not need_urls(args, "facebook"):
        return False

    article = build_article(args)
    photos = fetch_photos(article, fetcher, limit=6)
    log("TEST", f"downloaded {len(photos)} usable photo(s)")
    if not photos:
        raise ImageGenError("no usable photo downloaded")

    for i, p in enumerate(photos):
        save_image(out, f"3_source_photo_{i}.jpg", p)

    selected = analyze_source_photos(llm, article, photos, limit=3)
    for i, s in enumerate(selected):
        log(
            "TEST",
            f"selected[{i}] relevance={s.relevance_score} impact={s.visual_impact_score} "
            f"focus={s.focus_box} reason={s.reason!r}",
        )
    if not selected:
        raise ImageGenError("analysis selected no photo")

    layouts = sorted(SQUARE_LAYOUTS - {"auto"}) if args.layout == "all" else [args.layout]
    for layout in layouts:
        canvas, used = compose_original_square(selected, layout)
        log("TEST", f"layout={layout}: used {used} photo(s)")
        save_image(out, f"3_facebook_{layout}.jpg", canvas)
    return True


def step_full(
    cfg: Settings, args: argparse.Namespace, out: Path, fetcher: PoliteFetcher, llm: LLMRouter
) -> bool:
    """Real ImageGenerator.generate(): visual analysis, hero, quality checks and Facebook composite."""
    if not need_urls(args, "full"):
        return False

    article = build_article(args)
    ref = visual_analyzer.acquire_source_image(article, fetcher)
    log("TEST", "reference image found" if ref else "no reference image")

    visual = visual_analyzer.analyze(llm, article, (ref[0], ref[1]) if ref else None)
    log("TEST", f"visual: subject={visual.subject_type} people={visual.contains_real_people} summary={visual.summary[:100]!r}")

    gen = ImageGenerator(cfg, llm, fetcher=fetcher)
    started = time.time()
    res = gen.generate(
        story_id="testrun",
        article=article,
        v=visual,
        title=args.title,
        article_scene=args.scene or args.title,
        facebook_scene=args.scene or args.title,
        facebook_detail_scene="",
        facebook_composition_type=args.layout.upper() if args.layout not in ("auto", "all") else "INSET_CIRCLE_RIGHT",
        source_ref=(ref[0], ref[1]) if ref else None,
        source_url=ref[2] if ref else "",
        source_sha=ref[3] if ref else "",
        source_ahash=ref[4] if ref else "",
        known=[],
    )
    log("TEST", f"full pipeline OK in {time.time() - started:.1f}s")
    log("TEST", f"article image : {res.path}")
    log("TEST", f"facebook image: {res.facebook_path}")
    log("TEST", f"notes         : {res.notes}")
    return True


# ------------------------------------------------------------------ main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Standalone image pipeline test")
    ap.add_argument("--step", choices=["raw", "restyle", "facebook", "full", "all"], default="raw")
    ap.add_argument("--image-url", action="append", default=[], help="source photo URL (repeat for several)")
    ap.add_argument("--article-url", default="", help="optional; only stored in the fake article")
    ap.add_argument("--title", default="A surprising story with a clear main subject")
    ap.add_argument("--scene", default="", help="factual scene description (used in prompts)")
    ap.add_argument("--prompt", default="", help="custom prompt for --step raw")
    ap.add_argument("--layout", default="auto", help="auto | all | " + " | ".join(sorted(SQUARE_LAYOUTS - {'auto'})))
    ap.add_argument("--out", default="data/test_images")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = dataclasses.replace(
        Settings.from_env(),
        dry_run=True,
        dry_run_generate_images=True,
        git_push_enabled=False,
        data_dir=str(out),
        dry_run_dir=str(out),
    )
    setup_logging(cfg.secret_values())

    step = args.step
    log("TEST", f"step={step} provider={cfg.image_provider} model={cfg.cloudflare_image_model} out={out}")

    if cfg.image_provider == "cloudflare" and not cfg.cloudflare_ready():
        error("TEST", "CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN are missing")
        return 2

    gem = GeminiClient(cfg)
    llm = LLMRouter(gem, GroqClient(cfg))
    fetcher = PoliteFetcher(cfg.user_agent, cfg.request_timeout, cfg.per_host_delay_seconds)

    plan = ["raw", "restyle", "facebook", "full"] if step == "all" else [step]
    results: dict[str, str] = {}

    for name in plan:
        log("TEST", f"=============== {name} ===============")
        try:
            if name == "raw":
                ok = step_raw(cfg, args, out)
            elif name == "restyle":
                ok = step_restyle(cfg, args, out, fetcher)
            elif name == "facebook":
                ok = step_facebook(cfg, args, out, fetcher, llm)
            else:
                ok = step_full(cfg, args, out, fetcher, llm)
            results[name] = "OK" if ok else "SKIPPED"
        except ImageQuotaError as exc:
            results[name] = f"QUOTA: {exc}"
        except (ImageGenError, GeminiError) as exc:
            results[name] = f"FAILED: {exc}"
        except Exception as exc:  # noqa: BLE001
            traceback.print_exc()
            results[name] = f"CRASH: {type(exc).__name__}: {exc}"

    print("\n========== RESULT ==========")
    for name, status in results.items():
        print(f"{name:10s} {status}")

    return 0 if all(v in ("OK", "SKIPPED") for v in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
