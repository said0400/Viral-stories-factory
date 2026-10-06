# src/test_facebook.py
"""Standalone test for the Facebook square image (original photos only).

No Cloudflare, no image generation, no publishing, no history changes.

Examples (run from the repository root):

  # A real article page (uses the same extractor as the factory)
  python -m src.test_facebook --article-url https://example.com/story --layout all

  # Direct image URLs
  python -m src.test_facebook --image-url URL1 --image-url URL2 --image-url URL3 --layout triptych

  # Local files (when the site blocks GitHub runners)
  python -m src.test_facebook --local a.jpg b.jpg c.jpg --layout all

  # Without any LLM key: center crops instead of vision analysis
  python -m src.test_facebook --local a.jpg b.jpg --no-ai --layout all

Outputs go to data/test_facebook/ (override with --out):
  source_<i>.jpg           the downloaded candidate photos
  facebook_<layout>.jpg    one composite per layout
  contact_sheet.jpg        all layouts in one preview grid
"""
from __future__ import annotations

import argparse
import dataclasses
import io
import sys
import traceback
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps

from . import extractor
from .config import Settings
from .gemini_client import GeminiClient
from .groq_client import GroqClient
from .llm_router import LLMRouter
from .logger import error, log, setup_logging, warn
from .models import SourceArticle
from .photo_composer import (
    SQUARE_LAYOUTS,
    SelectedPhoto,
    analyze_source_photos,
    compose_original_square,
    fetch_photos,
)
from .utils import PoliteFetcher, utcnow
from .visual_analyzer import ahash, hamming

DEFAULT_FOCUS = (250, 180, 750, 820)  # normalized 0..1000: x0, y0, x1, y1
THUMB = 360


# ------------------------------------------------------------------ helpers
def save_image(path: Path, img: Image.Image) -> None:
    img.convert("RGB").save(path, "JPEG", quality=90)
    log("FBTEST", f"saved {path}  ({img.size[0]}x{img.size[1]})")


def load_local(paths: list[str]) -> list[Image.Image]:
    photos: list[Image.Image] = []
    for p in paths:
        try:
            img = Image.open(p)
            img.load()
            photos.append(ImageOps.exif_transpose(img).convert("RGB"))
        except Exception as exc:  # noqa: BLE001
            warn("FBTEST", f"cannot open {p}: {type(exc).__name__}")
    return photos


def dedupe(photos: list[Image.Image]) -> list[Image.Image]:
    out: list[Image.Image] = []
    hashes: list[str] = []
    for p in photos:
        h = ahash(p)
        if any(hamming(h, other) <= 6 for other in hashes):
            continue
        hashes.append(h)
        out.append(p)
    return out


def make_article(args: argparse.Namespace, fetcher: PoliteFetcher) -> SourceArticle:
    urls = [u.strip() for u in args.image_url if u.strip()]

    if args.article_url:
        base = SourceArticle(
            source_name="test",
            original_title="test",
            original_url=args.article_url,
            normalized_url=args.article_url,
            discovered_at=utcnow(),
        )
        full = extractor.extract_article(base, fetcher)
        if full:
            log(
                "FBTEST",
                f"article extracted: {full.original_title[:70]!r} | "
                f"images: main={'yes' if full.main_image_url else 'no'}, extra={len(full.additional_image_urls)}",
            )
            return full
        warn("FBTEST", "extractor returned nothing for --article-url; falling back to --image-url")

    return SourceArticle(
        source_name="test",
        original_title=args.title,
        original_url=args.article_url or "https://example.com/test",
        normalized_url=args.article_url or "https://example.com/test",
        description=args.title,
        article_text=args.title,
        main_image_url=urls[0] if urls else "",
        additional_image_urls=urls[1:],
        discovered_at=utcnow(),
    )


def select_without_ai(photos: list[Image.Image], limit: int = 3) -> list[SelectedPhoto]:
    return [
        SelectedPhoto(p, DEFAULT_FOCUS, max(0, 80 - i * 5), 55, "center crop (--no-ai)")
        for i, p in enumerate(photos[:limit])
    ]


def contact_sheet(items: list[tuple[str, Image.Image]]) -> Image.Image:
    cols = 3
    rows = (len(items) + cols - 1) // cols
    label_h = 26
    sheet = Image.new("RGB", (cols * THUMB, rows * (THUMB + label_h)), (24, 24, 28))
    draw = ImageDraw.Draw(sheet)
    for i, (name, img) in enumerate(items):
        x, y = (i % cols) * THUMB, (i // cols) * (THUMB + label_h)
        sheet.paste(img.resize((THUMB, THUMB), Image.LANCZOS), (x, y + label_h))
        draw.text((x + 8, y + 6), name, fill=(255, 255, 255))
    return sheet


# ------------------------------------------------------------------ main
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Facebook original-photo composite test")
    ap.add_argument("--article-url", default="", help="article page; images are extracted like the factory does")
    ap.add_argument("--image-url", action="append", default=[], help="direct image URL (repeatable)")
    ap.add_argument("--local", nargs="*", default=[], help="local image files")
    ap.add_argument("--title", default="Facebook composite test", help="used as context for the vision analysis")
    ap.add_argument("--layout", default="auto", help="auto | all | " + " | ".join(sorted(SQUARE_LAYOUTS - {"auto"})))
    ap.add_argument("--no-ai", action="store_true", help="skip vision analysis (no LLM keys needed)")
    ap.add_argument("--limit", type=int, default=6, help="max candidate photos to download")
    ap.add_argument("--out", default="data/test_facebook")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg = dataclasses.replace(Settings.from_env(), dry_run=True, git_push_enabled=False)
    setup_logging(cfg.secret_values())

    if not (args.article_url or args.image_url or args.local):
        error("FBTEST", "give at least one of --article-url, --image-url, --local")
        return 2

    fetcher = PoliteFetcher(cfg.user_agent, cfg.request_timeout, cfg.per_host_delay_seconds)
    article = make_article(args, fetcher)

    # ---- 1. collect photos
    photos = load_local(args.local)
    if article.main_image_url or article.additional_image_urls:
        photos += fetch_photos(article, fetcher, limit=max(1, min(args.limit, 10)))
    photos = dedupe(photos)

    log("FBTEST", f"{len(photos)} usable distinct photo(s)")
    if not photos:
        error("FBTEST", "no usable photo (min side 400px, ratio between 1:3 and 3:1)")
        return 1

    for i, p in enumerate(photos):
        save_image(out / f"source_{i}.jpg", p)

    # ---- 2. choose the best photos + focal boxes
    if args.no_ai:
        selected = select_without_ai(photos)
    else:
        gem = GeminiClient(cfg)
        llm = LLMRouter(gem, GroqClient(cfg))
        selected = analyze_source_photos(llm, article, photos, limit=3)

    if not selected:
        error("FBTEST", "analysis selected no photo")
        return 1

    for i, s in enumerate(selected):
        log(
            "FBTEST",
            f"selected[{i}] relevance={s.relevance_score} impact={s.visual_impact_score} "
            f"focus={s.focus_box} note={s.reason!r}",
        )

    # ---- 3. compose
    layouts = sorted(SQUARE_LAYOUTS - {"auto"}) if args.layout == "all" else [args.layout]
    if args.layout == "all":
        layouts = ["auto", *layouts]

    sheet_items: list[tuple[str, Image.Image]] = []
    failures = 0

    for layout in layouts:
        try:
            canvas, used = compose_original_square(selected, layout)
            log("FBTEST", f"layout={layout}: used {used} photo(s) -> {canvas.size[0]}x{canvas.size[1]}")
            save_image(out / f"facebook_{layout}.jpg", canvas)
            sheet_items.append((f"{layout} ({used})", canvas))
        except Exception as exc:  # noqa: BLE001
            failures += 1
            traceback.print_exc()
            error("FBTEST", f"layout={layout} failed: {type(exc).__name__}: {exc}")

    if sheet_items:
        save_image(out / "contact_sheet.jpg", contact_sheet(sheet_items))

    print("\n========== RESULT ==========")
    print(f"photos downloaded : {len(photos)}")
    print(f"photos selected   : {len(selected)}")
    print(f"layouts rendered  : {len(sheet_items)} / {len(layouts)}")
    print(f"output folder     : {out}")

    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
