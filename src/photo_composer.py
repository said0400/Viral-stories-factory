"""Facebook photo layouts built only from original images found in the source article.

Groq (with Gemini failover) ranks the images, marks focal regions, VERIFIES each crop in a feedback
loop and plans the layout after seeing the real crops. PIL only crops, enlarges, and arranges
the original pixels. It does not synthesize, repaint, anonymize, or erase image content.
"""
from __future__ import annotations

import io
import re
from dataclasses import dataclass
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field

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
MAX_CANDIDATES = 8
MAX_PAGE_URLS = 14
NATURAL_MAX_SIDE = 1200
RATIO_MIN = 0.8                  # 4:5
RATIO_MAX = 1.91                 # 1.91:1
LAYOUTS = {"auto", "single", "split", "inset"}


# ------------------------------------------------------------------ discovery of ALL article photos
_BAD_URL = re.compile(
    r"(logo|icon|sprite|avatar|favicon|advert|/ads?/|pixel|tracking|placeholder|blank|spinner|"
    r"emoji|badge|gravatar|1x1|\.svg|\.gif)",
    re.IGNORECASE,
)


def _best_from_srcset(value: str) -> str:
    best, best_w = "", -1
    for part in str(value or "").split(","):
        bits = part.strip().split()
        if not bits:
            continue
        width = 0
        if len(bits) > 1:
            m = re.match(r"(\d+)", bits[1])
            if m:
                width = int(m.group(1))
        if width >= best_w:
            best, best_w = bits[0], width
    return best


def collect_page_image_urls(
    article: SourceArticle,
    fetcher: PoliteFetcher,
    limit: int = MAX_PAGE_URLS,
) -> list[str]:
    """Scrape every story-related image URL from the article page (og:image + body images)."""
    page = str(article.original_url or "").strip()
    if not page or fetcher is None:
        return []

    try:
        r = fetcher.get(page)
    except FetchError:
        return []

    try:
        html = r.text
    except Exception:
        html = ""
    finally:
        try:
            r.close()
        except Exception:
            pass

    if not html:
        return []

    try:
        soup = BeautifulSoup(html, "lxml")
    except Exception:
        return []

    out: list[str] = []

    def add(raw: str) -> None:
        raw = str(raw or "").strip()
        if not raw or raw.startswith("data:"):
            return
        url = urljoin(page, raw)
        parsed = urlparse(url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            return
        if _BAD_URL.search(url):
            return
        if url not in out:
            out.append(url)

    for attrs in ({"property": "og:image"}, {"name": "twitter:image"}):
        for meta in soup.find_all("meta", attrs=attrs):
            add(meta.get("content", ""))

    root = soup.find("article") or soup.find("main") or soup.body or soup
    for img in root.find_all("img"):
        if img.find_parent(["header", "footer", "nav", "aside"]):
            continue
        candidate = (
            _best_from_srcset(img.get("data-srcset") or img.get("srcset") or "")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
            or img.get("src")
            or ""
        )
        add(candidate)

    return out[:limit]


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

    # Also scrape the article page itself so we get ALL related images, not only what the extractor kept.
    try:
        for u in collect_page_image_urls(article, fetcher):
            if u not in urls:
                urls.append(u)
    except Exception as exc:
        logger.warn("PHOTO", f"page image discovery failed ({type(exc).__name__})")

    photos: list[Image.Image] = []
    hashes: list[str] = []

    for url in urls[:MAX_PAGE_URLS]:
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

    logger.log("PHOTO", f"collected {len(photos)} distinct candidate photos from {len(urls)} discovered URLs")
    return photos


# ------------------------------------------------------------------ source-photo ranking
class PhotoAssessmentSchema(BaseModel):
    index: int
    relevance_score: int = Field(ge=0, le=100)
    visual_impact_score: int = Field(ge=0, le=100)
    focus_center_x: int = Field(ge=0, le=1000)
    focus_center_y: int = Field(ge=0, le=1000)
    focus_width: int = Field(ge=80, le=1000)
    focus_height: int = Field(ge=80, le=1000)
    focal_description: str


class PhotoAssessmentBatch(BaseModel):
    items: list[PhotoAssessmentSchema]


class FocusRefineSchema(BaseModel):
    crop_ok: bool
    focus_center_x: int = Field(ge=0, le=1000)
    focus_center_y: int = Field(ge=0, le=1000)
    focus_width: int = Field(ge=80, le=1000)
    focus_height: int = Field(ge=80, le=1000)
    reason: str


class LayoutPlanSchema(BaseModel):
    layout: str
    order: list[int]
    reason: str


@dataclass(frozen=True)
class SelectedPhoto:
    image: Image.Image
    focus_box: tuple[int, int, int, int]  # normalized x0,y0,x1,y1 in 0..1000
    relevance_score: int
    visual_impact_score: int
    reason: str = ""


def _jpeg(img: Image.Image, max_side: int = 1024) -> bytes:
    copy = img.copy()
    copy.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    copy.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def _box_from_center(cx: int, cy: int, fw: int, fh: int) -> tuple[int, int, int, int]:
    cx = max(0, min(1000, int(cx)))
    cy = max(0, min(1000, int(cy)))
    fw = max(80, min(1000, int(fw)))
    fh = max(80, min(1000, int(fh)))
    x0, x1 = max(0, cx - fw // 2), min(1000, cx + fw // 2)
    y0, y1 = max(0, cy - fh // 2), min(1000, cy + fh // 2)
    return x0, y0, x1, y1


def analyze_source_photos(
    gem: GeminiClient,
    article: SourceArticle,
    photos: list[Image.Image],
    *,
    limit: int = 4,
) -> list[SelectedPhoto]:
    """Rank source photos in vision-sized batches and find story-specific focal boxes."""
    if not photos:
        return []

    limit = max(1, min(int(limit), 4))
    excerpt = (article.article_text or article.description or "")[:1800]
    verdicts: dict[int, PhotoAssessmentSchema] = {}
    for start in range(0, len(photos), 3):
        batch = photos[start : start + 3]
        end = start + len(batch)
        global_indices = list(range(start, end))
        prompt = (
            f"ARTICLE TITLE: {article.original_title}\nARTICLE CONTEXT: {excerpt}\n\n"
            f"You receive {len(batch)} ORIGINAL article images in order for global indices {global_indices}.\n"
            "Return each image's global index, relevance_score (0-100), visual_impact_score (0-100), "
            "focus_center_x, focus_center_y, focus_width, focus_height (normalized 0-1000), and focal_description.\n"
            "Rank highest the genuine, distinct source photos that best show the verified event or a different important detail; avoid near-duplicate angles when better distinct photos exist. "
            "Give relevance below 45 to unrelated, stock, or decorative images; only clearly story-related photos should score 45 or higher. "
            "Place the focal rectangle tightly around the exact important person/object/evidence, but retain enough surroundings to explain the scene; coordinates are normalized to 0-1000 of the original pixels.\n"
            "Source-native captions, signs, interface elements, arrows, logos, watermarks, and collage panels "
            "are part of the original; do not erase or reject them. Analyze visible content only. "
            "Do not identify people, infer unsupported facts, or request image generation/editing."
        )
        try:
            response = gem.generate_json(
                prompt,
                PhotoAssessmentBatch,
                images=[(_jpeg(p), "image/jpeg") for p in batch],
                temperature=0.1,
                tag="PHOTO_ANALYSIS",
            )
            for verdict in response.items:
                idx = int(verdict.index)
                # Accept explicit global indices; also normalize local indices for compatible providers.
                if start <= idx < end:
                    global_index = idx
                elif 0 <= idx < len(batch):
                    global_index = start + idx
                else:
                    continue
                verdicts[global_index] = verdict
        except Exception as exc:
            logger.warn(
                "PHOTO",
                f"source-photo batch {start // 3 + 1} analysis failed ({type(exc).__name__}); using center crops for that batch",
            )

    ranked: list[tuple[float, int, SelectedPhoto]] = []
    for i, photo in enumerate(photos):
        v = verdicts.get(i)
        if v is None:
            selected = SelectedPhoto(photo, (250, 180, 750, 820), 20, 20, "AI analysis unavailable; center-crop fallback")
        else:
            cx = max(0, min(1000, int(v.focus_center_x)))
            cy = max(0, min(1000, int(v.focus_center_y)))
            fw = max(80, min(1000, int(v.focus_width)))
            fh = max(80, min(1000, int(v.focus_height)))
            x0, x1 = max(0, cx - fw // 2), min(1000, cx + fw // 2)
            y0, y1 = max(0, cy - fh // 2), min(1000, cy + fh // 2)
            selected = SelectedPhoto(
                photo,
                (x0, y0, x1, y1),
                max(0, min(100, int(v.relevance_score))),
                max(0, min(100, int(v.visual_impact_score))),
                str(v.focal_description or "")[:180],
            )
        combined = selected.relevance_score * 0.7 + selected.visual_impact_score * 0.3
        ranked.append((combined, i, selected))

    ranked.sort(key=lambda row: (row[0], -row[1]), reverse=True)
    relevant = [row for row in ranked if row[2].relevance_score >= 45]
    if not relevant and ranked:
        relevant = ranked[:1]
    chosen = [item for _, _, item in relevant[:limit]]
    logger.log("PHOTO", f"selected {len(chosen)} of {len(photos)} original article photos using visual relevance and focal detail")
    return chosen


# ------------------------------------------------------------------ crop verification loop
def _annotated(img: Image.Image, box: tuple[int, int, int, int], max_side: int = 1024) -> bytes:
    copy = img.copy().convert("RGB")
    copy.thumbnail((max_side, max_side))
    w, h = copy.size
    x0, y0, x1, y1 = box
    ImageDraw.Draw(copy).rectangle(
        (x0 * w / 1000.0, y0 * h / 1000.0, x1 * w / 1000.0, y1 * h / 1000.0),
        outline=(255, 0, 0),
        width=max(3, w // 200),
    )
    buf = io.BytesIO()
    copy.save(buf, "JPEG", quality=85)
    return buf.getvalue()


def refine_focus_boxes(
    gem: GeminiClient,
    article: SourceArticle,
    photos: list[SelectedPhoto],
    rounds: int = 2,
) -> list[SelectedPhoto]:
    """Show the AI its own crop (red box + enlarged result) and let it correct the box until it approves."""
    excerpt = (article.article_text or article.description or "")[:1200]
    refined: list[SelectedPhoto] = []

    for number, selected in enumerate(photos, 1):
        current = selected

        for round_no in range(max(1, rounds)):
            try:
                preview = _focus_crop(current, (640, 640), context=1.3)
                prompt = (
                    f"ARTICLE TITLE: {article.original_title}\nARTICLE CONTEXT: {excerpt}\n\n"
                    "Image 1 is an ORIGINAL article photo with a RED rectangle marking the proposed focal region. "
                    "Image 2 is the enlarged crop around that region.\n"
                    "The focal region must tightly contain the single most important person/animal/object/evidence for this story, "
                    "fully inside the box (no cut-off heads, hands, faces or objects) and with no wasted empty area.\n"
                    "Set crop_ok=true only if the box is already precise. Otherwise set crop_ok=false and return a corrected box "
                    "(focus_center_x, focus_center_y, focus_width, focus_height) normalized 0-1000 relative to the ORIGINAL Image 1.\n"
                    "Do not identify people or infer facts. Keep reason to one short sentence."
                )
                res = gem.generate_json(
                    prompt,
                    FocusRefineSchema,
                    images=[
                        (_annotated(current.image, current.focus_box), "image/jpeg"),
                        (_jpeg(preview, 640), "image/jpeg"),
                    ],
                    temperature=0.0,
                    tag="PHOTO_REFINE",
                )
            except Exception as exc:
                logger.warn("PHOTO", f"crop verification unavailable for photo {number} ({type(exc).__name__})")
                break

            if res.crop_ok:
                break

            new_box = _box_from_center(res.focus_center_x, res.focus_center_y, res.focus_width, res.focus_height)
            if new_box == current.focus_box:
                break

            logger.log("PHOTO", f"photo {number}: crop corrected (round {round_no + 1}): {str(res.reason)[:120]}")
            current = SelectedPhoto(
                current.image,
                new_box,
                current.relevance_score,
                current.visual_impact_score,
                current.reason,
            )

        refined.append(current)

    return refined


# ------------------------------------------------------------------ layout planning (AI sees real crops)
def plan_layout(
    gem: GeminiClient,
    article: SourceArticle,
    photos: list[SelectedPhoto],
    preferred: str = "auto",
) -> tuple[str, list[SelectedPhoto]]:
    """Let the AI choose the layout and photo order after looking at the actual selected crops."""
    fallback_layout = _layout_name(preferred)
    count = len(photos)

    if count <= 1:
        return fallback_layout, photos

    allowed = [
        "inset_circle_right", "inset_circle_left", "inset_square_right", "inset_square_left",
        "diptych_split", "diptych_stack",
    ]
    if count >= 3:
        allowed += ["triptych", "triptych_bottom"]
    if count >= 4:
        allowed += ["quad_grid"]

    excerpt = (article.article_text or article.description or "")[:1200]
    prompt = (
        f"ARTICLE TITLE: {article.original_title}\nARTICLE CONTEXT: {excerpt}\n\n"
        f"You receive {count} enlarged crops (indices 0..{count - 1}) taken from the article's ORIGINAL photos.\n"
        f"Choose ONE layout from: {', '.join(allowed)}.\n"
        "Guidance: inset_* = one dominant main photo plus a small detail photo; diptych_split = two equally strong "
        "people/perspectives; diptych_stack = before/after or sequence; triptych / triptych_bottom = one main plus two supporting "
        "details; quad_grid = four equally relevant moments.\n"
        "Return `order` as all indices sorted by importance: index first in the list becomes the main/hero panel. "
        "The combined square image must tell the story at a glance with complete, meaningful panels, not random pieces.\n"
        "Return layout, order, and one short reason sentence. Do not identify people."
    )

    try:
        res = gem.generate_json(
            prompt,
            LayoutPlanSchema,
            images=[(_jpeg(_focus_crop(p, (512, 512), context=2.4), 512), "image/jpeg") for p in photos],
            temperature=0.1,
            tag="LAYOUT",
        )
    except Exception as exc:
        logger.warn("PHOTO", f"layout planning unavailable ({type(exc).__name__}); using default layout rules")
        return fallback_layout, photos

    layout = _layout_name(res.layout)
    if layout not in allowed:
        layout = fallback_layout if fallback_layout in allowed else "auto"

    order: list[int] = []
    for raw in res.order or []:
        try:
            i = int(raw)
        except (TypeError, ValueError):
            continue
        if 0 <= i < count and i not in order:
            order.append(i)
    for i in range(count):
        if i not in order:
            order.append(i)

    logger.log("PHOTO", f"AI layout plan: {layout}, order={order} ({str(res.reason)[:120]})")
    return layout, [photos[i] for i in order]


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


# ------------------------------------------------------------------ square layouts
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
    "quad_grid",
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
    """Compose two or three photos on a 1080x1080 canvas. No text is rendered."""
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


def _focus_crop(
    selected: SelectedPhoto,
    size: tuple[int, int],
    *,
    context: float = 2.8,
) -> Image.Image:
    """Crop around the AI-marked region, then resize; all visible pixels come from the source.

    The crop never gets so small that it must be upscaled more than ~1.65x (keeps zoomed details sharp).
    """
    img = selected.image.convert("RGB")
    iw, ih = img.size
    x0, y0, x1, y1 = selected.focus_box
    cx = (x0 + x1) * 0.5 * iw / 1000.0
    cy = (y0 + y1) * 0.5 * ih / 1000.0
    focus_w = max(1.0, (x1 - x0) * iw / 1000.0)
    focus_h = max(1.0, (y1 - y0) * ih / 1000.0)
    target_ratio = size[0] / size[1]

    min_w = size[0] * 0.6
    crop_w = max(focus_w * context, focus_h * context * target_ratio, min_w)
    crop_h = crop_w / target_ratio
    if crop_w > iw:
        crop_w, crop_h = float(iw), iw / target_ratio
    if crop_h > ih:
        crop_h, crop_w = float(ih), ih * target_ratio
    crop_w = min(float(iw), max(1.0, crop_w))
    crop_h = min(float(ih), max(1.0, crop_h))

    left = min(max(0.0, cx - crop_w / 2), iw - crop_w)
    top = min(max(0.0, cy - crop_h / 2), ih - crop_h)
    crop = img.crop((int(left), int(top), int(left + crop_w), int(top + crop_h)))
    result = ImageOps.fit(crop, size, method=Image.LANCZOS, centering=(0.5, 0.5))

    if crop.size[0] < size[0]:
        result = result.filter(ImageFilter.UnsharpMask(radius=1.2, percent=60, threshold=3))

    return result


def _layout_name(value: str) -> str:
    name = str(value or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "single": "single_hero",
        "inset_circle": "inset_circle_right",
        "inset_square": "inset_square_right",
        "diptych": "diptych_split",
        "split": "diptych_split",
        "stack": "diptych_stack",
        "quad": "quad_grid",
        "grid_2x2": "quad_grid",
    }
    name = aliases.get(name, name)
    return name if name in SQUARE_LAYOUTS else "auto"


def _same_photo_detail(selected: SelectedPhoto) -> SelectedPhoto:
    return SelectedPhoto(
        selected.image,
        selected.focus_box,
        selected.relevance_score,
        selected.visual_impact_score,
        selected.reason,
    )


def compose_original_square(
    photos: list[SelectedPhoto],
    layout: str = "auto",
) -> tuple[Image.Image, int]:
    """Compose an original-photo Facebook square, enlarging AI-selected regions without inventing pixels."""
    if not photos:
        raise ValueError("no original photos to compose")

    photos = photos[:4]
    normalized = _layout_name(layout)
    count = len(photos)

    if normalized == "auto":
        if count >= 4:
            normalized = "quad_grid"
        elif count >= 3:
            normalized = "triptych"
        elif count == 2 and abs(photos[0].relevance_score - photos[1].relevance_score) <= 15:
            normalized = "diptych_split"
        else:
            normalized = "inset_circle_right"

    if normalized == "single_hero":
        normalized = "inset_circle_right"

    if normalized == "quad_grid" and count < 4:
        normalized = "triptych" if count >= 3 else "diptych_split" if count >= 2 else "inset_circle_right"

    if normalized.startswith("inset_"):
        main = _focus_crop(photos[0], (SQUARE_SIDE, SQUARE_SIDE), context=3.0)
        detail = photos[1] if count >= 2 else _same_photo_detail(photos[0])
        inset = 360
        margin = 42
        side = "left" if normalized.endswith("_left") else "right"
        shape = "square" if "square" in normalized else "circle"
        x = margin if side == "left" else SQUARE_SIDE - inset - margin
        y = margin
        detail_crop = _focus_crop(detail, (inset, inset), context=1.45)

        if shape == "circle":
            _paste_circle(main, detail_crop, (x, y), inset, ring=10)
        else:
            border = 10
            draw = ImageDraw.Draw(main)
            draw.rectangle((x - border, y - border, x + inset + border, y + inset + border), fill=(248, 248, 248))
            main.paste(detail_crop, (x, y))
        return main, count

    if normalized == "diptych_split":
        if count < 2:
            return compose_original_square(photos, "inset_circle_right")
        gap = 8
        left_w = (SQUARE_SIDE - gap) // 2
        right_w = SQUARE_SIDE - gap - left_w
        canvas = Image.new("RGB", (SQUARE_SIDE, SQUARE_SIDE), BACKGROUND)
        canvas.paste(_focus_crop(photos[0], (left_w, SQUARE_SIDE), context=2.7), (0, 0))
        canvas.paste(_focus_crop(photos[1], (right_w, SQUARE_SIDE), context=2.7), (left_w + gap, 0))
        return canvas, 2

    if normalized == "diptych_stack":
        if count < 2:
            return compose_original_square(photos, "inset_circle_right")
        gap = 8
        top_h = (SQUARE_SIDE - gap) // 2
        bottom_h = SQUARE_SIDE - gap - top_h
        canvas = Image.new("RGB", (SQUARE_SIDE, SQUARE_SIDE), BACKGROUND)
        canvas.paste(_focus_crop(photos[0], (SQUARE_SIDE, top_h), context=2.7), (0, 0))
        canvas.paste(_focus_crop(photos[1], (SQUARE_SIDE, bottom_h), context=2.7), (0, top_h + gap))
        return canvas, 2

    if normalized == "quad_grid":
        gap = 8
        cell = (SQUARE_SIDE - gap) // 2
        canvas = Image.new("RGB", (SQUARE_SIDE, SQUARE_SIDE), BACKGROUND)
        positions = [(0, 0), (cell + gap, 0), (0, cell + gap), (cell + gap, cell + gap)]
        for photo, (x, y) in zip(photos[:4], positions):
            canvas.paste(_focus_crop(photo, (cell, cell), context=2.2), (x, y))
        return canvas, 4

    if count < 3:
        return compose_original_square(photos, "inset_circle_right")

    gap = 8
    canvas = Image.new("RGB", (SQUARE_SIDE, SQUARE_SIDE), BACKGROUND)
    if normalized == "triptych_bottom":
        upper_h = (SQUARE_SIDE - gap) // 2
        lower_h = SQUARE_SIDE - gap - upper_h
        left_w = (SQUARE_SIDE - gap) // 2
        right_w = SQUARE_SIDE - gap - left_w
        canvas.paste(_focus_crop(photos[0], (left_w, upper_h), context=2.4), (0, 0))
        canvas.paste(_focus_crop(photos[1], (right_w, upper_h), context=2.4), (left_w + gap, 0))
        canvas.paste(_focus_crop(photos[2], (SQUARE_SIDE, lower_h), context=2.4), (0, upper_h + gap))
    else:
        main_w = 610
        secondary_w = SQUARE_SIDE - main_w - gap
        secondary_h = (SQUARE_SIDE - gap) // 2
        canvas.paste(_focus_crop(photos[0], (main_w, SQUARE_SIDE), context=2.6), (0, 0))
        canvas.paste(_focus_crop(photos[1], (secondary_w, secondary_h), context=2.4), (main_w + gap, 0))
        canvas.paste(
            _focus_crop(photos[2], (secondary_w, SQUARE_SIDE - gap - secondary_h), context=2.4),
            (main_w + gap, secondary_h + gap),
        )
    return canvas, 3
