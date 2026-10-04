"""Facebook photo layouts built from the story's own source photos (no AI generation, no text)."""
from __future__ import annotations

import io

from PIL import Image, ImageDraw, ImageEnhance, ImageFilter, ImageOps

from . import logger
from .models import SourceArticle
from .utils import FetchError, PoliteFetcher
from .visual_analyzer import ahash, hamming

CANVAS = (1080, 1350)            # 4:5 portrait; use (1080, 1080) for a square image
GAP = 8                          # divider between two panels
RING = (255, 196, 0)             # yellow ring of the circular inset
MAX_DOWNLOAD_BYTES = 12 * 1024 * 1024
MIN_SIDE = 400
LAYOUTS = {"auto", "single", "split", "inset"}


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


def fetch_photos(article: SourceArticle, fetcher: PoliteFetcher, limit: int = 2) -> list[Image.Image]:
    """Download up to `limit` distinct, usable photos of the article (robots.txt respected by the fetcher)."""
    urls: list[str] = []

    for u in [article.main_image_url, *article.additional_image_urls]:
        u = str(u or "").strip()
        if u and u not in urls:
            urls.append(u)

    photos: list[Image.Image] = []
    hashes: list[str] = []

    for url in urls[:8]:
        if len(photos) >= limit:
            break

        try:
            r = fetcher.get(url)
        except FetchError as exc:
            logger.warn("PHOTO", f"image not fetched ({exc})")
            continue

        try:
            ctype = (r.headers.get("Content-Type", "").split(";", 1)[0].strip().lower())

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


# ------------------------------------------------------------------ layout helpers
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


def _blur_fill(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Whole photo, never cropped: sharp copy centred over a blurred, darkened copy of itself."""
    w, h = size

    bg = _cover(img, size, 0.5).filter(ImageFilter.GaussianBlur(28))
    bg = ImageEnhance.Brightness(bg).enhance(0.6)

    iw, ih = img.size
    scale = min(w / iw, h / ih)
    fg = img.resize((max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))), Image.LANCZOS)

    bg.paste(fg, ((w - fg.size[0]) // 2, (h - fg.size[1]) // 2))

    return bg


def _panel(img: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Crop when the photo is close to the panel's shape (loses at most ~25%), otherwise keep it whole."""
    target = size[0] / size[1]
    actual = img.size[0] / img.size[1]

    if 0.75 <= actual / target <= 1.33:
        return _cover(img, size)

    return _blur_fill(img, size)


def _circle_mask(size: int) -> Image.Image:
    big = Image.new("L", (size * 4, size * 4), 0)
    ImageDraw.Draw(big).ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    return big.resize((size, size), Image.LANCZOS)


def _paste_circle(canvas: Image.Image, img: Image.Image, xy: tuple[int, int], diameter: int, ring: int = 10) -> None:
    outer = diameter + 2 * ring
    x, y = xy

    canvas.paste(Image.new("RGB", (outer, outer), RING), (x, y), _circle_mask(outer))
    canvas.paste(_cover(img, (diameter, diameter), 0.2), (x + ring, y + ring), _circle_mask(diameter))


def compose(photos: list[Image.Image], layout: str = "auto") -> Image.Image:
    """Build the Facebook image. No text is ever drawn."""
    if not photos:
        raise ValueError("no photos to compose")

    layout = layout if layout in LAYOUTS else "auto"

    if len(photos) == 1 or layout == "single":
        return _panel(photos[0], CANVAS)

    first, second = photos[0], photos[1]
    width, height = CANVAS

    if layout == "inset":
        canvas = _panel(first, CANVAS)
        _paste_circle(canvas, second, (40, 40), 340)
        return canvas

    canvas = Image.new("RGB", CANVAS, (12, 14, 18))
    both_portrait = all(p.size[0] / p.size[1] < 1.0 for p in (first, second))

    if both_portrait:
        left_w = (width - GAP) // 2
        right_w = width - GAP - left_w

        canvas.paste(_panel(first, (left_w, height)), (0, 0))
        canvas.paste(_panel(second, (right_w, height)), (left_w + GAP, 0))
    else:
        top_h = (height - GAP) // 2
        bottom_h = height - GAP - top_h

        canvas.paste(_panel(first, (width, top_h)), (0, 0))
        canvas.paste(_panel(second, (width, bottom_h)), (0, top_h + GAP))

    return canvas
