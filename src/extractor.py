"""Full-article extraction: title, body, date, images, canonical URL, metadata."""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

from . import logger
from .models import SourceArticle
from .utils import (
    FetchError,
    PoliteFetcher,
    clean_text,
    normalize_url,
    parse_datetime,
)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------
def _meta(soup: BeautifulSoup, *names: str) -> str:
    """Return the first non-empty metadata value matching the supplied names."""
    for n in names:
        tag = (
            soup.find("meta", attrs={"property": n})
            or soup.find("meta", attrs={"name": n})
        )

        if tag and tag.get("content"):
            value = clean_text(str(tag["content"]))
            if value:
                return value

    return ""


# ---------------------------------------------------------------------------
# JSON-LD
# ---------------------------------------------------------------------------
def _jsonld(soup: BeautifulSoup) -> dict:
    """Return the most useful Article/NewsArticle JSON-LD object."""
    accepted_types = {
        "Article",
        "NewsArticle",
        "ReportageNewsArticle",
        "BlogPosting",
    }

    for s in soup.find_all("script", type="application/ld+json"):
        raw = s.string or s.get_text() or ""

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

        if isinstance(data, dict):
            if isinstance(data.get("@graph"), list):
                items = data["@graph"]
            else:
                items = [data]
        elif isinstance(data, list):
            items = data
        else:
            continue

        for it in items:
            if not isinstance(it, dict):
                continue

            raw_type = it.get("@type", "")

            if isinstance(raw_type, list):
                types = {
                    str(x).strip()
                    for x in raw_type
                    if x
                }
            else:
                types = {
                    str(raw_type).strip()
                } if raw_type else set()

            if types & accepted_types:
                return it

    return {}


# ---------------------------------------------------------------------------
# URL helpers
# ---------------------------------------------------------------------------
def _safe_http_url(url: str) -> str:
    """Return a valid absolute HTTP(S) URL or an empty string."""
    if not isinstance(url, str):
        return ""

    value = url.strip()

    if not value:
        return ""

    if value.startswith(("//",)):
        value = f"https:{value}"

    try:
        parsed = urlparse(value)
    except ValueError:
        return ""

    if parsed.scheme.lower() not in {"http", "https"}:
        return ""

    if not parsed.netloc:
        return ""

    if parsed.username is not None or parsed.password is not None:
        return ""

    return value


# ---------------------------------------------------------------------------
# srcset / image helpers
# ---------------------------------------------------------------------------
def _best_from_srcset(srcset: str) -> str:
    """Return the highest-resolution URL from a srcset attribute."""
    if not srcset:
        return ""

    best = ""
    best_w = -1
    best_index = -1

    for index, part in enumerate(srcset.split(",")):
        bits = part.strip().split()

        if not bits:
            continue

        candidate = bits[0]

        if not candidate:
            continue

        m = (
            re.match(r"(\d+)w$", bits[1])
            if len(bits) > 1
            else None
        )

        w = int(m.group(1)) if m else 0

        if w > best_w or (w == best_w and index > best_index):
            best = candidate
            best_w = w
            best_index = index

    return best


def _looks_like_bad_image_url(url: str) -> bool:
    """Reject obvious UI/decorative image URLs."""
    if not url:
        return True

    low = url.lower().strip()

    if low.startswith(
        (
            "data:",
            "blob:",
            "javascript:",
            "mailto:",
        )
    ):
        return True

    if re.search(
        r"(logo|avatar|icon|sprite|pixel|tracking|favicon|placeholder|"
        r"social[-_]?share|share[-_]?image|badge|button)",
        low,
        re.I,
    ):
        return True

    return False


def _image_dimensions(tag) -> tuple[int, int]:
    """Read numeric width/height attributes when available."""
    width = tag.get("width", "")
    height = tag.get("height", "")

    w = int(width) if str(width).isdigit() else 0
    h = int(height) if str(height).isdigit() else 0

    return w, h


def _image_candidate_from_tag(tag) -> str:
    """Return the best image URL from an <img> tag (lazy-load aware)."""
    # Prefer responsive srcset because it often contains a substantially
    # higher-resolution editorial image than the fallback src.
    for attr in ("srcset", "data-srcset"):
        v = tag.get(attr)

        if isinstance(v, str) and v.strip():
            best = _best_from_srcset(v)

            if best and not _looks_like_bad_image_url(best):
                return best

    # Fallback to ordinary and lazy-loaded image attributes.
    for attr in (
        "src",
        "data-src",
        "data-lazy-src",
        "data-original",
        "data-lazy",
    ):
        v = tag.get(attr)

        if isinstance(v, str):
            v = v.strip()

            if v and not _looks_like_bad_image_url(v):
                return v

    return ""


def _images(
    soup: BeautifulSoup,
    base: str,
    ld: dict,
) -> list[str]:
    """Collect likely editorial images while filtering UI/decorative assets."""
    urls: list[str] = []

    # OpenGraph / Twitter image.
    og = _meta(
        soup,
        "og:image",
        "twitter:image",
        "twitter:image:src",
    )

    if og and not _looks_like_bad_image_url(og):
        joined = urljoin(base, og)
        safe = _safe_http_url(joined)

        if safe:
            urls.append(safe)

    # JSON-LD image(s).
    img = ld.get("image")

    if isinstance(img, dict):
        img_items = [img]
    elif isinstance(img, list):
        img_items = img
    else:
        img_items = [img]

    for it in img_items:
        if isinstance(it, dict):
            u = (
                it.get("url")
                or it.get("contentUrl")
                or it.get("thumbnailUrl")
                or ""
            )
        else:
            u = it

        if isinstance(u, str) and u.strip():
            u = u.strip()

            if not _looks_like_bad_image_url(u):
                joined = urljoin(base, u)
                safe = _safe_http_url(joined)

                if safe:
                    urls.append(safe)

    # Prefer article/main content for ordinary <img> extraction.
    root = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    if not root:
        return urls[:8]

    for tag in root.find_all("img"):
        src = _image_candidate_from_tag(tag)

        if _looks_like_bad_image_url(src):
            continue

        width, height = _image_dimensions(tag)

        # Ignore explicitly tiny images.
        if width and width < 300:
            continue

        if height and height < 200:
            continue

        # Ignore obvious tracking/decorative dimensions.
        if width and height:
            ratio = width / max(height, 1)

            if width < 400 and height < 400:
                continue

            if ratio > 8 or ratio < 0.12:
                continue

        joined = urljoin(base, src)
        safe = _safe_http_url(joined)

        if safe:
            urls.append(safe)

    # Also inspect <source srcset> elements inside picture tags.
    for source in root.find_all("source"):
        srcset = (
            source.get("srcset")
            or source.get("data-srcset")
            or ""
        )

        src = _best_from_srcset(str(srcset))

        if _looks_like_bad_image_url(src):
            continue

        joined = urljoin(base, src)
        safe = _safe_http_url(joined)

        if safe:
            urls.append(safe)

    # Deduplicate while preserving priority order.
    seen: set[str] = set()
    out: list[str] = []

    for u in urls:
        if not isinstance(u, str):
            continue

        u = u.strip()

        if not u:
            continue

        key = u.split("#", 1)[0]

        if key in seen:
            continue

        seen.add(key)
        out.append(u)

    return out[:8]


# ---------------------------------------------------------------------------
# Article body extraction
# ---------------------------------------------------------------------------
def _body(soup: BeautifulSoup) -> str:
    for t in soup(
        [
            "script",
            "style",
            "nav",
            "footer",
            "aside",
            "form",
            "noscript",
            "template",
            "svg",
        ]
    ):
        t.decompose()

    root = (
        soup.find("article")
        or soup.find("main")
        or soup.body
        or soup
    )

    if not root:
        return ""

    paras = []

    for p in root.find_all(
        [
            "p",
            "h2",
            "h3",
            "li",
        ]
    ):
        text = clean_text(
            p.get_text(" ", strip=True)
        )

        if not text:
            continue

        if len(text) <= 30:
            continue

        if len(text) < 200 and re.search(
            r"("
            r"subscribe|"
            r"sign[\s-]*up|"
            r"newsletter|"
            r"cookie|"
            r"accept cookies|"
            r"all rights reserved|"
            r"follow us|"
            r"follow on|"
            r"share this|"
            r"read more|"
            r"related stories|"
            r"you may also like|"
            r"advertisement|"
            r"advertising"
            r")",
            text,
            re.I,
        ):
            continue

        paras.append(text)

    seen: set[str] = set()
    unique: list[str] = []

    for p in paras:
        key = re.sub(
            r"\s+",
            " ",
            p,
        ).strip().lower()

        if key in seen:
            continue

        seen.add(key)
        unique.append(p)

    return "\n".join(unique)[:12000]


# ---------------------------------------------------------------------------
# Canonical URL
# ---------------------------------------------------------------------------
def _canonical_url(
    soup: BeautifulSoup,
    base_url: str,
) -> str:
    canon_tag = soup.find(
        "link",
        rel=lambda value: (
            "canonical" in value
            if isinstance(value, list)
            else value == "canonical"
        ),
    )

    if not canon_tag:
        return ""

    href = canon_tag.get("href")

    if not isinstance(href, str) or not href.strip():
        return ""

    href = href.strip()

    if href.startswith(
        (
            "javascript:",
            "data:",
            "mailto:",
            "#",
        )
    ):
        return ""

    try:
        resolved = urljoin(
            base_url,
            href,
        )
    except Exception:
        return ""

    return _safe_http_url(resolved)


# ---------------------------------------------------------------------------
# Author
# ---------------------------------------------------------------------------
def _author(soup: BeautifulSoup, ld: dict) -> str:
    author = ld.get("author")

    if isinstance(author, list):
        for item in author:
            if isinstance(item, dict):
                name = item.get("name")

                if isinstance(name, str) and name.strip():
                    return clean_text(name)

            elif isinstance(item, str) and item.strip():
                return clean_text(item)

    elif isinstance(author, dict):
        name = author.get("name")

        if isinstance(name, str) and name.strip():
            return clean_text(name)

    elif isinstance(author, str) and author.strip():
        return clean_text(author)

    return _meta(
        soup,
        "author",
        "article:author",
        "byl",
    )


# ---------------------------------------------------------------------------
# Publication date
# ---------------------------------------------------------------------------
def _publication_date(
    soup: BeautifulSoup,
    ld: dict,
    candidate: SourceArticle,
):
    candidates = [
        ld.get("datePublished"),
        ld.get("dateCreated"),
        _meta(
            soup,
            "article:published_time",
            "og:published_time",
            "datePublished",
            "date",
        ),
    ]

    for value in candidates:
        if not value:
            continue

        try:
            parsed = parse_datetime(str(value))

            if parsed:
                return parsed

        except Exception:
            continue

    return candidate.publication_date


# ---------------------------------------------------------------------------
# Full article extraction
# ---------------------------------------------------------------------------
def extract_article(
    candidate: SourceArticle,
    fetcher: PoliteFetcher,
) -> SourceArticle | None:
    try:
        r = fetcher.get(
            candidate.original_url
        )
    except FetchError as exc:
        logger.warn(
            "EXTRACT",
            f"{candidate.source_name}: {exc}",
        )
        return None

    try:
        r.encoding = r.apparent_encoding or "utf-8"

        soup = BeautifulSoup(
            r.text,
            "lxml",
        )

    except Exception as exc:
        logger.warn(
            "EXTRACT",
            f"{candidate.source_name}: failed to parse HTML: {exc}",
        )
        return None

    ld = _jsonld(soup)
    canonical = _canonical_url(
        soup,
        candidate.original_url,
    )
    author = _author(
        soup,
        ld,
    )
    date = _publication_date(
        soup,
        ld,
        candidate,
    )
    images = _images(
        soup,
        candidate.original_url,
        ld,
    )

    title = (
        _meta(
            soup,
            "og:title",
            "twitter:title",
        )
        or clean_text(
            str(ld.get("headline", ""))
        )
        or candidate.original_title
    )

    description = (
        _meta(
            soup,
            "og:description",
            "twitter:description",
            "description",
        )
        or candidate.description
    )

    text = _body(soup)

    normalized_source_url = normalize_url(
        canonical or candidate.original_url
    )

    if not normalized_source_url:
        normalized_source_url = normalize_url(
            candidate.original_url
        )

    upd = candidate.model_copy(
        update={
            "original_title": clean_text(title),
            "canonical_url": canonical,
            "normalized_url": normalized_source_url,
            "publication_date": date,
            "author": clean_text(author),
            "description": clean_text(description),
            "article_text": text,
            "main_image_url": (
                images[0]
                if images
                else candidate.main_image_url
            ),
            "additional_image_urls": images[1:],
        }
    )

    return upd


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def is_valid(
    article: SourceArticle,
    min_chars: int,
) -> tuple[bool, str]:
    if not article.original_title or not article.original_url:
        return False, "missing title/url"

    article_text_len = len(
        clean_text(article.article_text or "")
    )

    description_len = len(
        clean_text(article.description or "")
    )

    if article_text_len < min_chars:
        if description_len < 200:
            return False, "not enough text to verify the story"

    if article_text_len == 0 and description_len == 0:
        return False, "empty article content"

    if not article.normalized_url:
        return False, "missing normalized URL"

    return True, ""
