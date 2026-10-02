"""Full-article extraction: title, body, date, images, canonical URL, metadata."""
from __future__ import annotations

import json
import re
from urllib.parse import urljoin

from bs4 import BeautifulSoup

from . import logger
from .models import SourceArticle
from .utils import FetchError, PoliteFetcher, clean_text, normalize_url, parse_datetime


def _meta(soup: BeautifulSoup, *names: str) -> str:
    for n in names:
        tag = soup.find("meta", attrs={"property": n}) or soup.find("meta", attrs={"name": n})
        if tag and tag.get("content"):
            return clean_text(tag["content"])
    return ""


def _jsonld(soup: BeautifulSoup) -> dict:
    for s in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(s.string or "")
        except (json.JSONDecodeError, TypeError):
            continue
        items = data if isinstance(data, list) else data.get("@graph", [data])
        for it in items:
            if isinstance(it, dict) and str(it.get("@type", "")).endswith(
                    ("Article", "NewsArticle", "ReportageNewsArticle", "BlogPosting")):
                return it
    return {}


def _best_from_srcset(srcset: str) -> str:
    best, best_w = "", -1
    for part in srcset.split(","):
        bits = part.strip().split()
        if not bits:
            continue
        m = re.match(r"(\d+)w", bits[1]) if len(bits) > 1 else None
        w = int(m.group(1)) if m else 0
        if w > best_w:
            best, best_w = bits[0], w
    return best


def _images(soup: BeautifulSoup, base: str, ld: dict) -> list[str]:
    urls: list[str] = []
    og = _meta(soup, "og:image", "twitter:image")
    if og:
        urls.append(urljoin(base, og))
    img = ld.get("image")
    for it in (img if isinstance(img, list) else [img]):
        u = it.get("url") if isinstance(it, dict) else it
        if isinstance(u, str) and u:
            urls.append(urljoin(base, u))
    root = soup.find("article") or soup.find("main") or soup
    for tag in root.find_all("img"):
        src = _best_from_srcset(tag.get("srcset", "")) or tag.get("data-src") or tag.get("src") or ""
        if not src or src.startswith("data:") or re.search(r"(logo|avatar|icon|sprite|pixel)", src, re.I):
            continue
        w = tag.get("width", "")
        if w.isdigit() and int(w) < 300:
            continue
        urls.append(urljoin(base, src))
    seen, out = set(), []
    for u in urls:
        k = u.split("?")[0]
        if k not in seen:
            seen.add(k)
            out.append(u)
    return out[:6]


def _body(soup: BeautifulSoup) -> str:
    for t in soup(["script", "style", "nav", "footer", "aside", "form", "noscript", "figure"]):
        t.decompose()
    root = soup.find("article") or soup.find("main") or soup.body or soup
    paras = [clean_text(p.get_text(" ")) for p in root.find_all(["p", "h2", "h3", "li"])]
    paras = [p for p in paras if len(p) > 40 and not re.search(
        r"(subscribe|sign up|newsletter|cookie|all rights reserved|follow us)", p, re.I)]
    return "\n".join(paras)[:12000]


def extract_article(candidate: SourceArticle, fetcher: PoliteFetcher) -> SourceArticle | None:
    """Fill a discovery candidate with full details. Returns None if unusable."""
    try:
        r = fetcher.get(candidate.original_url)
    except FetchError as exc:
        logger.warn("EXTRACT", f"{candidate.source_name}: {exc}")
        return None
    soup = BeautifulSoup(r.text, "lxml")
    ld = _jsonld(soup)
    canon_tag = soup.find("link", rel="canonical")
    canonical = urljoin(candidate.original_url, canon_tag["href"]) if canon_tag and canon_tag.get("href") else ""
    author = ""
    a = ld.get("author")
    if isinstance(a, list) and a:
        a = a[0]
    if isinstance(a, dict):
        author = a.get("name", "")
    elif isinstance(a, str):
        author = a
    author = author or _meta(soup, "author", "article:author")

    date = parse_datetime(ld.get("datePublished") or _meta(soup, "article:published_time", "og:published_time")) \
        or candidate.publication_date
    images = _images(soup, candidate.original_url, ld)
    title = _meta(soup, "og:title") or clean_text(ld.get("headline", "")) or candidate.original_title
    text = _body(soup)

    upd = candidate.model_copy(update={
        "original_title": title,
        "canonical_url": canonical,
        "normalized_url": normalize_url(canonical or candidate.original_url),
        "publication_date": date,
        "author": clean_text(author),
        "description": _meta(soup, "og:description", "description") or candidate.description,
        "article_text": text,
        "main_image_url": images[0] if images else candidate.main_image_url,
        "additional_image_urls": images[1:],
    })
    return upd


def is_valid(article: SourceArticle, min_chars: int) -> tuple[bool, str]:
    """Reject articles whose basic data prevents verification of the story."""
    if not article.original_title or not article.original_url:
        return False, "missing title/url"
    if len(article.article_text) < min_chars and len(article.description) < 200:
        return False, "not enough text to verify the story"
    if not article.publication_date:
        return True, "no publication date (allowed, flagged)"
    return True, ""
