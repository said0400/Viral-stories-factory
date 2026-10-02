"""Source definitions + discovery: RSS -> sitemap -> section pages (web extraction last).

NOTE: the endpoint URLs below are best-effort defaults and some sites change or block them.
Every method is robots.txt-aware and failure-isolated; override with SOURCES_OVERRIDE_FILE.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urljoin, urlparse

import feedparser
from bs4 import BeautifulSoup

from . import logger
from .models import SourceArticle
from .utils import FetchError, PoliteFetcher, clean_text, normalize_url, parse_datetime, utcnow


@dataclass
class SourceDef:
    name: str
    domain: str
    rss: list[str] = field(default_factory=list)
    sitemaps: list[str] = field(default_factory=list)
    sections: list[str] = field(default_factory=list)
    article_path_regex: str = r"/.+"   # path filter for sitemap / section links
    enabled: bool = True
    max_items: int = 25


DEFAULT_SOURCES: list[SourceDef] = [
    SourceDef(
        "Bored Panda",
        "boredpanda.com",
        rss=["https://www.boredpanda.com/feed/"],
        sections=["https://www.boredpanda.com/"],
        article_path_regex=r"^/[a-z0-9\-]{12,}/?$",
    ),
    SourceDef(
        "BuzzFeed",
        "buzzfeed.com",
        rss=["https://www.buzzfeed.com/index.xml"],
        sections=["https://www.buzzfeed.com/"],
        article_path_regex=r"^/[a-z0-9\-]+/[a-z0-9\-]{12,}",
    ),
    SourceDef(
        "Bright Side",
        "brightside.me",
        sitemaps=["https://brightside.me/sitemap.xml"],
        sections=["https://brightside.me/"],
        article_path_regex=r"^/(articles|inspiration|wonder-animals|wonder-curiosities)/",
    ),
    SourceDef(
        "Reuters",
        "reuters.com",
        sitemaps=[
            "https://www.reuters.com/arc/outboundfeeds/news-sitemap-index/?outputType=xml"
        ],
        sections=["https://www.reuters.com/offbeat/"],
        article_path_regex=r"^/.+-\d{4}-\d{2}-\d{2}/?$",
    ),
    SourceDef(
        "Associated Press",
        "apnews.com",
        sitemaps=["https://apnews.com/news-sitemap-content.xml"],
        sections=["https://apnews.com/hub/oddities"],
        article_path_regex=r"^/article/",
    ),
    SourceDef(
        "Daily Mail",
        "dailymail.co.uk",
        rss=["https://www.dailymail.co.uk/home/index.rss"],
        sections=["https://www.dailymail.co.uk/news/offbeat/index.html"],
        article_path_regex=r"/article-\d+/",
    ),
    SourceDef(
        "The Sun",
        "thesun.co.uk",
        rss=["https://www.thesun.co.uk/feed/"],
        sitemaps=["https://www.thesun.co.uk/news-sitemap.xml"],
        sections=["https://www.thesun.co.uk/news/"],
        article_path_regex=r"^/.+/\d{6,}/?$",
    ),
]


def load_sources(override_file: str = "") -> list[SourceDef]:
    if override_file and Path(override_file).exists():
        try:
            data = json.loads(Path(override_file).read_text(encoding="utf-8"))

            if not isinstance(data, list):
                raise ValueError("sources override must contain a JSON list")

            sources: list[SourceDef] = []

            for item in data:
                if not isinstance(item, dict):
                    logger.warn("DISCOVERY", "ignoring invalid source override entry")
                    continue

                try:
                    sources.append(SourceDef(**item))
                except TypeError as exc:
                    logger.warn(
                        "DISCOVERY",
                        f"ignoring invalid source override entry ({exc})",
                    )

            if sources:
                return sources

            logger.warn(
                "DISCOVERY",
                "sources override file contained no valid sources; using defaults",
            )

        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warn(
                "DISCOVERY",
                f"could not load sources override ({exc}); using defaults",
            )

    return list(DEFAULT_SOURCES)


def _same_domain(src: SourceDef, url: str) -> bool:
    """Return True only when URL belongs to the configured source domain."""
    try:
        host = (urlparse(url).hostname or "").lower().rstrip(".")
    except ValueError:
        return False

    domain = src.domain.lower().strip().lower().rstrip(".")

    if not host or not domain:
        return False

    return host == domain or host.endswith("." + domain)


def _candidate(
    src: SourceDef,
    url: str,
    title: str,
    section: str,
    date=None,
    desc: str = "",
    image: str = "",
) -> SourceArticle | None:
    url = (url or "").strip()

    if not url.startswith(("http://", "https://")):
        return None

    if not _same_domain(src, url):
        return None

    try:
        parsed = urlparse(url)
    except ValueError:
        return None

    if not re.search(src.article_path_regex, parsed.path or "/"):
        return None

    normalized = normalize_url(url)

    if not normalized:
        return None

    return SourceArticle(
        source_name=src.name,
        original_title=clean_text(title) or normalized,
        original_url=url,
        normalized_url=normalized,
        publication_date=date,
        description=clean_text(desc, 500),
        main_image_url=(image or "").strip(),
        source_section=section,
        discovered_at=utcnow(),
    )


def _dedupe(items: list[SourceArticle]) -> list[SourceArticle]:
    """Deduplicate discovered candidates by normalized URL."""
    out: list[SourceArticle] = []
    seen: set[str] = set()

    for item in items:
        key = item.normalized_url or item.original_url

        if not key or key in seen:
            continue

        seen.add(key)
        out.append(item)

    return out


# ------------------------------------------------------------------ discovery methods
def from_rss(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    out: list[SourceArticle] = []

    for feed_url in src.rss:
        try:
            r = fetcher.get(feed_url)
        except FetchError as exc:
            logger.warn("DISCOVERY", f"{src.name}: RSS unavailable ({exc})")
            continue

        try:
            feed = feedparser.parse(r.content)
        except Exception as exc:
            logger.warn("DISCOVERY", f"{src.name}: RSS parse failed ({exc})")
            continue

        for e in feed.entries[: src.max_items * 2]:
            img = ""

            for m in (e.get("media_content") or []) + (e.get("media_thumbnail") or []):
                if not isinstance(m, dict):
                    continue

                candidate = m.get("url") or m.get("href") or ""

                if candidate:
                    img = candidate
                    break

            if not img:
                for enc in e.get("enclosures") or []:
                    if not isinstance(enc, dict):
                        continue

                    enc_type = str(enc.get("type", "")).lower()

                    if enc_type.startswith("image"):
                        img = enc.get("href") or enc.get("url") or ""
                        if img:
                            break

            date = parse_datetime(
                e.get("published_parsed")
                or e.get("updated_parsed")
                or e.get("published")
            )

            desc = BeautifulSoup(
                e.get("summary", "") or e.get("description", ""),
                "lxml",
            ).get_text(" ")

            c = _candidate(
                src,
                e.get("link", ""),
                e.get("title", ""),
                "rss",
                date,
                desc,
                img,
            )

            if c:
                out.append(c)

    out = _dedupe(out)
    out.sort(
        key=lambda a: a.publication_date or a.discovered_at,
        reverse=True,
    )

    return out[: src.max_items * 2]


def _parse_sitemap(xml: bytes) -> tuple[list[dict], list[str]]:
    """Parse both sitemap indexes and regular URL sitemaps."""
    soup = BeautifulSoup(xml, "xml")

    children: list[str] = []

    for sitemap in soup.find_all("sitemap"):
        loc = sitemap.find("loc")

        if loc:
            value = loc.get_text(strip=True)

            if value:
                children.append(value)

    urls: list[dict] = []

    for u in soup.find_all("url"):
        loc = u.find("loc")

        if not loc:
            continue

        loc_text = loc.get_text(strip=True)

        if not loc_text:
            continue

        date_node = (
            u.find("news:publication_date")
            or u.find("publication_date")
            or u.find("lastmod")
        )

        title_node = u.find("news:title") or u.find("title")

        image_url = ""

        image_node = u.find("image:image")

        if image_node:
            image_loc = image_node.find("image:loc") or image_node.find("loc")

            if image_loc:
                image_url = image_loc.get_text(strip=True)

        if not image_url:
            image_loc = u.find("image:loc")

            if image_loc:
                image_url = image_loc.get_text(strip=True)

        urls.append(
            {
                "loc": loc_text,
                "date": (
                    parse_datetime(date_node.get_text(strip=True))
                    if date_node
                    else None
                ),
                "title": (
                    title_node.get_text(strip=True)
                    if title_node
                    else ""
                ),
                "image": image_url,
            }
        )

    return urls, children


def from_sitemap(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    out: list[SourceArticle] = []
    queue = list(src.sitemaps)
    visited_urls: set[str] = set()

    while queue and len(visited_urls) < 4:
        sm = queue.pop(0)

        if sm in visited_urls:
            continue

        visited_urls.add(sm)

        try:
            r = fetcher.get(sm)
        except FetchError as exc:
            logger.warn("DISCOVERY", f"{src.name}: sitemap unavailable ({exc})")
            continue

        try:
            urls, children = _parse_sitemap(r.content)
        except Exception as exc:
            logger.warn("DISCOVERY", f"{src.name}: sitemap parse failed ({exc})")
            continue

        for child in children[:2]:
            if child not in visited_urls and child not in queue:
                queue.append(child)

        for u in urls:
            title = (
                u["title"]
                or u["loc"].rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
            )

            c = _candidate(
                src,
                u["loc"],
                title,
                "sitemap",
                u["date"],
                "",
                u["image"],
            )

            if c:
                out.append(c)

    out = _dedupe(out)

    out.sort(
        key=lambda a: a.publication_date or a.discovered_at,
        reverse=True,
    )

    return out[: src.max_items * 2]


def from_sections(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    out: list[SourceArticle] = []

    for sec in src.sections:
        try:
            r = fetcher.get(sec)
        except FetchError as exc:
            logger.warn("DISCOVERY", f"{src.name}: section unavailable ({exc})")
            continue

        try:
            soup = BeautifulSoup(r.text, "lxml")
        except Exception as exc:
            logger.warn("DISCOVERY", f"{src.name}: section parse failed ({exc})")
            continue

        seen: set[str] = set()

        for a in soup.find_all("a", href=True):
            href = str(a.get("href") or "").strip()

            if not href:
                continue

            url = urljoin(sec, href)

            title = clean_text(a.get_text(" "))

            if len(title) < 20:
                continue

            normalized = normalize_url(url)

            if not normalized or normalized in seen:
                continue

            seen.add(normalized)

            c = _candidate(
                src,
                url,
                title,
                sec,
                None,
                "",
                "",
            )

            if c:
                out.append(c)

    out = _dedupe(out)

    return out[: src.max_items]


def discover(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    """RSS -> sitemap -> section pages. Uses the first method that yields results."""
    if not src.enabled:
        logger.log("DISCOVERY", f"{src.name}: disabled")
        return []

    methods = (
        ("RSS", from_rss, bool(src.rss)),
        ("sitemap", from_sitemap, bool(src.sitemaps)),
        ("section", from_sections, bool(src.sections)),
    )

    for name, fn, has in methods:
        if not has:
            continue

        found = fn(src, fetcher)

        if found:
            logger.log(
                "DISCOVERY",
                f"{src.name}: {len(found)} candidates via {name}",
            )
            return found[: src.max_items]

    logger.warn("DISCOVERY", f"{src.name}: no candidates found")
    return []
