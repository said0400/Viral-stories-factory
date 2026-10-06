"""Facebook photo layouts built from the story's own source photos.

No AI generation, no filters, no text: photos are only cropped and placed.
A photo is used only when a vision check finds no text/logo/watermark, no minors and no collage.
"""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageOps
from pydantic import BaseModel

from . import logger
from .gemini_client import GeminiClient, GeminiError
from .models import SourceArticle
from .utils import FetchError, PoliteFetcher
from .visual_analyzer import ahash, hamming

CANVAS_SIDE = (1200, 1000)       # two tall photos side by side
CANVAS_STACK = (1080, 1350)      # two wide photos stacked
GAP = 8                          # divider between two panels
BACKGROUND = (12, 14, 18)
RING = (255, 196, 0)             # yellow ring of the circular inset
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
MIN_SIDE = 400
MAX_CANDIDATES = 6
NATURAL_MAX_SIDE = 1200
RATIO_MIN = 0.8                  # 4:5
RATIO_MAX = 1.91                 # 1.91:1
LAYOUTS = {"auto", "single", "split", "inset"}


# ------------------------------------------------------------------ download
def _load(raw: bytes) -> Image.Image | None:
    if not raw or len(raw) > MAX_DOWNLOAD_BYTES:
        return None

    try:
        img = Image.open(io.BytesIO(raw))
        img.load()
        img = ImageOps.exif_transpose(img).convert("RGB")
    except Exception:
        return None

    w, h = img.size

    if min(w, h) < MIN_SIDE:
        return None

    ratio = w / h

    if ratio > 3.0 or ratio < 0.33:
        return None

    return img


def fetch_photos(
    article: SourceArticle,
    fetcher: PoliteFetcher,
    limit: int = MAX_CANDIDATES,
) -> list[Image.Image]:
    """Download up to `limit` distinct, usable candidate photos (robots.txt respected by the fetcher)."""
    urls: list[str] = []

    for u in [article.main_image_url, *article.additional_image_urls]:
        u = str(u or "").strip()
        if u and u not in urls:
            urls.append(u)

    photos: list[Image.Image] = []
    hashes: list[str] = []

    for url in urls[:10]:
        if len(photos) >= limit:
            break

        try:
            r = fetcher.get(url)
        except FetchError as exc:
            logger.warn("PHOTO", f"image not fetched ({exc})")
            continue

        try:
            ctype = r.headers.get("Content-Type", "").split(";", 1)[0].strip().lower()

            if not ctype.startswith("image/"):
                continue

            img = _load(r.content)
        finally:
            r.close()

        if img is None:
            continue

        h = ahash(img)

        if any(hamming(h, other) <= 6 for other in hashes):
            continue

        hashes.append(h)
        photos.append(img)

    return photos


# ------------------------------------------------------------------ screening
class PhotoVerdict(BaseModel):
    index: int
    has_text_or_logo: bool
    has_minors: bool
    is_collage_or_screenshot: bool
    reason: str


class PhotoScreening(BaseModel):
    items: list[PhotoVerdict]


def _jpeg(img: Image.Image, max_side: int = 1024) -> bytes:
    copy = img.copy()
    copy.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    copy.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def screen_photos(gem: GeminiClient, photos: list[Image.Image]) -> list[int]:
    """Indexes of photos that are safe to publish as-is. Fails CLOSED: on any error nothing is approved."""
    if not photos:
        return []

    prompt = (
        f"You receive {len(photos)} photographs numbered from 0 in the order supplied.\n"
        "For EACH photograph return one item with its index and these judgements:\n"
        "- has_text_or_logo: true if the photograph shows ANY text overlay, caption, subtitle, watermark, "
        "website or channel logo, social-media interface, or clearly readable lettering or brand logo anywhere in it.\n"
        "- has_minors: true if any person who is or may be a child or teenager is visible.\n"
        "- is_collage_or_screenshot: true if it is a collage, a multi-panel image, a screenshot or a meme "
        "rather than one single photograph.\n"
        "- reason: one short sentence.\n"
        "Judge only what is visible. Do not identify anyone."
    )

    try:
        res = gem.generate_json(
            prompt,
            PhotoScreening,
            images=[(_jpeg(p), "image/jpeg") for p in photos],
            temperature=0.0,
            tag="IMGCHECK",
        )
    except GeminiError as exc:
        logger.warn("PHOTO", f"photo screening unavailable ({str(exc)[:120]}); real photos will not be used")
        return []
    except Exception as exc:
        logger.warn("PHOTO", f"photo screening error ({type(exc).__name__}); real photos will not be used")
        return []

    verdicts = {v.index: v for v in res.items if 0 <= v.index < len(photos)}
    clean: list[int] = []

    for i in range(len(photos)):
        v = verdicts.get(i)

        if v is None:
            logger.warn("PHOTO", f"photo {i}: no verdict; skipped")
            continue

        if v.has_text_or_logo or v.has_minors or v.is_collage_or_screenshot:
            flags = [
                name
                for name, on in (
                    ("text/logo", v.has_text_or_logo),
                    ("minors", v.has_minors),
                    ("collage/screenshot", v.is_collage_or_screenshot),
                )
                if on
            ]
            logger.log("PHOTO", f"photo {i} rejected ({', '.join(flags)}): {v.reason[:100]}")
            continue

        clean.append(i)

    return clean


# ------------------------------------------------------------------ layout helpers
def _is_tall(img: Image.Image) -> bool:
    w, h = img.size
    return w / h <= 1.1


def _natural(img: Image.Image) -> Image.Image:
    """Keep the photo as it is: crop only when its ratio is outside 4:5 .. 1.91:1, then cap the size."""
    img = img.convert("RGB")
    w, h = img.size
    ratio = w / h

    if ratio < RATIO_MIN:
        new_h = int(round(w / RATIO_MIN))
        top = int((h - new_h) * 0.15)
        img = img.crop((0, top, w, top + new_h))
    elif ratio > RATIO_MAX:
        new_w = int(round(h * RATIO_MAX))
        left = (w - new_w) // 2
        img = img.crop((left, 0, left + new_w, h))

    longest = max(img.size)

    if longest > NATURAL_MAX_SIDE:
        scale = NATURAL_MAX_SIDE / longest
        img = img.resize(
            (max(1, int(round(img.size[0] * scale))), max(1, int(round(img.size[1] * scale)))),
            Image.LANCZOS,
        )

    return img


def _cover(img: Image.Image, size: tuple[int, int], bias: float = 0.25) -> Image.Image:
    """Fill `size` completely (crops). `bias` keeps the upper part, where faces usually are."""
    w, h = size
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = max(w, int(round(iw * scale))), max(h, int(round(ih * scale)))

    img = img.resize((nw, nh), Image.LANCZOS)

    left = (nw - w) // 2
    top = int((nh - h) * bias)

    return img.crop((left, top, left + w, top + h))


def _circle_mask(size: int) -> Image.Image:
    big = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(big).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    return big.resize((size, size), Image.LANCZOS)


def _paste_circle(canvas: Image.Image, img: Image.Image, xy: tuple[int, int], diameter: int, ring: int = 10) -> None:
    outer = diameter + 2 * ring
    x, y = xy

    canvas.paste(Image.new("RGB", (outer, outer), RING), (x, y), _circle_mask(outer))
    canvas.paste(_cover(img, (diameter, diameter), 0.2), (x + ring, y + ring), _circle_mask(diameter))


def compose(photos: list[Image.Image], layout: str = "auto") -> tuple[Image.Image, int]:
    """Build the Facebook image from approved photos. Returns (image, number_of_photos_used). Never draws text."""
    if not photos:
        raise ValueError("no photos to compose")

    layout = layout if layout in LAYOUTS else "auto"
    first = photos[0]

    if layout == "single" or len(photos) == 1:
        return _natural(first), 1

    if layout == "inset":
        base = _natural(first)
        diameter = int(min(base.size) * 0.36)
        margin = int(min(base.size) * 0.04)
        _paste_circle(base, photos[1], (margin, margin), diameter)
        return base, 2

    partner = next((p for p in photos[1:] if _is_tall(p) == _is_tall(first)), None)

    if partner is None:
        return _natural(first), 1

    if _is_tall(first):
        width, height = CANVAS_SIDE
        left_w = (width - GAP) // 2
        right_w = width - GAP - left_w

        canvas = Image.new("RGB", CANVAS_SIDE, BACKGROUND)
        canvas.paste(_cover(first, (left_w, height)), (0, 0))
        canvas.paste(_cover(partner, (right_w, height)), (left_w + GAP, 0))

        return canvas, 2

    width, height = CANVAS_STACK
    top_h = (height - GAP) // 2
    bottom_h = height - GAP - top_h

    canvas = Image.new("RGB", CANVAS_STACK, BACKGROUND)
    canvas.paste(_cover(first, (width, top_h)), (0, 0))
    canvas.paste(_cover(partner, (width, bottom_h)), (0, top_h + GAP))

    return canvas, 2


# ------------------------------------------------------------------ square AI-image layouts
SQUARE_SIDE = 1080
SQUARE_LAYOUTS = {
    "auto",
    "single_hero",
    "inset_circle_right",
    "inset_circle_left",
    "inset_square_right",
    "inset_square_left",
    "diptych_split",
    "diptych_stack",
    "triptych",
    "triptych_bottom",
}


def to_reference(img: Image.Image, max_side: int = 1280) -> tuple[bytes, str]:
    """Encode a downloaded source photo for an image model reference input."""
    return _jpeg(img, max_side=max_side), "image/jpeg"


def _square_cover(img: Image.Image, size: tuple[int, int], bias: float = 0.22) -> Image.Image:
    return _cover(img.convert("RGB"), size, bias=bias)


def _paste_square_inset(
    canvas: Image.Image,
    detail: Image.Image,
    *,
    side: str,
    shape: str,
) -> None:
    diameter = 350
    margin = 42
    x = margin if side == "left" else SQUARE_SIDE - diameter - margin
    y = margin
    detail = _square_cover(detail, (diameter, diameter), bias=0.18)

    if shape == "circle":
        _paste_circle(canvas, detail, (x, y), diameter, ring=9)
        return

    border = 9
    draw = ImageDraw.Draw(canvas)
    draw.rectangle(
        (x - border, y - border, x + diameter + border, y + diameter + border),
        fill=(245, 245, 245),
    )
    canvas.paste(detail, (x, y))


def compose_square(photos: list[Image.Image], layout: str = "auto") -> tuple[Image.Image, int]:
    """Compose two or three generated/reference-matched photos on a 1080×1080 canvas.

    No text is rendered. ``triptych`` places one portrait panel beside two stacked
    panels; ``triptych_bottom`` places two square panels above a wide lower panel.
    Unknown layouts and incomplete three-photo layouts degrade gracefully.
    """
    if not photos:
        raise ValueError("no photos to compose")

    normalized = str(layout or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if normalized not in SQUARE_LAYOUTS:
        normalized = "auto"

    side = SQUARE_SIDE
    gap = 8
    bg = Image.new("RGB", (side, side), BACKGROUND)
    count = len(photos)

    if normalized == "auto":
        normalized = "triptych" if count >= 3 else "inset_circle_right" if count >= 2 else "single_hero"

    if normalized == "single_hero" or count == 1:
        return _square_cover(photos[0], (side, side)), 1

    if normalized.startswith("inset_"):
        canvas = _square_cover(photos[0], (side, side))
        shape = "circle" if "circle" in normalized else "square"
        inset_side = "left" if normalized.endswith("_left") else "right"
        _paste_square_inset(canvas, photos[1], side=inset_side, shape=shape)
        return canvas, 2

    if normalized == "diptych_split":
        left_w = (side - gap) // 2
        right_w = side - gap - left_w
        bg.paste(_square_cover(photos[0], (left_w, side)), (0, 0))
        bg.paste(_square_cover(photos[1], (right_w, side)), (left_w + gap, 0))
        return bg, 2

    if normalized == "diptych_stack":
        top_h = (side - gap) // 2
        bottom_h = side - gap - top_h
        bg.paste(_square_cover(photos[0], (side, top_h)), (0, 0))
        bg.paste(_square_cover(photos[1], (side, bottom_h)), (0, top_h + gap))
        return bg, 2

    if count < 3:
        # A triptych needs three distinct photos; a two-photo inset is less destructive.
        fallback = "inset_circle_right"
        canvas = _square_cover(photos[0], (side, side))
        _paste_square_inset(canvas, photos[1], side="right", shape="circle")
        logger.warn("PHOTO", f"{normalized} needs three images; used {fallback}")
        return canvas, 2

    if normalized == "triptych_bottom":
        top_h = (side - gap) // 2
        lower_h = side - gap - top_h
        left_w = (side - gap) // 2
        right_w = side - gap - left_w
        bg.paste(_square_cover(photos[0], (left_w, top_h)), (0, 0))
        bg.paste(_square_cover(photos[1], (right_w, top_h)), (left_w + gap, 0))
        bg.paste(_square_cover(photos[2], (side, lower_h)), (0, top_h + gap))
        return bg, 3

    # Main vertical frame on the left; two secondary near-square frames on the right.
    main_w = 610
    secondary_w = side - main_w - gap
    secondary_h = (side - gap) // 2
    bg.paste(_square_cover(photos[0], (main_w, side)), (0, 0))
    bg.paste(_square_cover(photos[1], (secondary_w, secondary_h)), (main_w + gap, 0))
    bg.paste(
        _square_cover(photos[2], (secondary_w, side - gap - secondary_h)),
        (main_w + gap, secondary_h + gap),
    )
    return bg, 3
