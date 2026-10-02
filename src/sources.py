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
    SourceDef("Bored Panda", "boredpanda.com",
              rss=["https://www.boredpanda.com/feed/"],
              sections=["https://www.boredpanda.com/"], article_path_regex=r"^/[a-z0-9\-]{12,}/?$"),
    SourceDef("BuzzFeed", "buzzfeed.com",
              rss=["https://www.buzzfeed.com/index.xml"],
              sections=["https://www.buzzfeed.com/"], article_path_regex=r"^/[a-z0-9\-]+/[a-z0-9\-]{12,}"),
    SourceDef("Bright Side", "brightside.me",
              sitemaps=["https://brightside.me/sitemap.xml"],
              sections=["https://brightside.me/"], article_path_regex=r"^/(articles|inspiration|wonder-animals|wonder-curiosities)/"),
    SourceDef("Reuters", "reuters.com",
              sitemaps=["https://www.reuters.com/arc/outboundfeeds/news-sitemap-index/?outputType=xml"],
              sections=["https://www.reuters.com/offbeat/"], article_path_regex=r"^/.+-\d{4}-\d{2}-\d{2}/?$"),
    SourceDef("Associated Press", "apnews.com",
              sitemaps=["https://apnews.com/news-sitemap-content.xml"],
              sections=["https://apnews.com/hub/oddities"], article_path_regex=r"^/article/"),
    SourceDef("Daily Mail", "dailymail.co.uk",
              rss=["https://www.dailymail.co.uk/home/index.rss"],
              sections=["https://www.dailymail.co.uk/news/offbeat/index.html"],
              article_path_regex=r"/article-\d+/"),
    SourceDef("The Sun", "thesun.co.uk",
              rss=["https://www.thesun.co.uk/feed/"],
              sitemaps=["https://www.thesun.co.uk/news-sitemap.xml"],
              sections=["https://www.thesun.co.uk/news/"], article_path_regex=r"^/.+/\d{6,}/?$"),
]


def load_sources(override_file: str = "") -> list[SourceDef]:
    if override_file and Path(override_file).exists():
        data = json.loads(Path(override_file).read_text(encoding="utf-8"))
        return [SourceDef(**d) for d in data]
    return list(DEFAULT_SOURCES)


def _candidate(src: SourceDef, url: str, title: str, section: str, date=None,
               desc: str = "", image: str = "") -> SourceArticle | None:
    url = (url or "").strip()
    if not url.startswith("http") or src.domain not in urlparse(url).netloc:
        return None
    if not re.search(src.article_path_regex, urlparse(url).path):
        return None
    return SourceArticle(
        source_name=src.name, original_title=clean_text(title) or url, original_url=url,
        normalized_url=normalize_url(url), publication_date=date, description=clean_text(desc, 500),
        main_image_url=image, source_section=section, discovered_at=utcnow())


# ------------------------------------------------------------------ discovery methods
def from_rss(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    out: list[SourceArticle] = []
    for feed_url in src.rss:
        try:
            r = fetcher.get(feed_url)
        except FetchError as exc:
            logger.warn("DISCOVERY", f"{src.name}: RSS unavailable ({exc})")
            continue
        feed = feedparser.parse(r.content)
        for e in feed.entries[: src.max_items * 2]:
            img = ""
            for m in (e.get("media_content") or []) + (e.get("media_thumbnail") or []):
                img = m.get("url", img)
            for enc in e.get("enclosures") or []:
                if str(enc.get("type", "")).startswith("image"):
                    img = enc.get("href", img)
            date = parse_datetime(e.get("published_parsed") or e.get("updated_parsed")
                                  or e.get("published"))
            desc = BeautifulSoup(e.get("summary", ""), "lxml").get_text(" ")
            c = _candidate(src, e.get("link", ""), e.get("title", ""), "rss", date, desc, img)
            if c:
                out.append(c)
    return out


def _parse_sitemap(xml: bytes) -> tuple[list[dict], list[str]]:
    soup = BeautifulSoup(xml, "xml")
    children = [s.find("loc").get_text(strip=True) for s in soup.find_all("sitemap") if s.find("loc")]
    urls = []
    for u in soup.find_all("url"):
        loc = u.find("loc")
        if not loc:
            continue
        date = u.find("news:publication_date") or u.find("publication_date") or u.find("lastmod")
        title = u.find("news:title") or u.find("title")
        img = u.find("image:loc") or u.find("loc", recursive=False)
        urls.append({"loc": loc.get_text(strip=True),
                     "date": parse_datetime(date.get_text(strip=True)) if date else None,
                     "title": title.get_text(strip=True) if title else "",
                     "image": img.get_text(strip=True) if img and img is not loc else ""})
    return urls, children


def from_sitemap(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    out: list[SourceArticle] = []
    queue = list(src.sitemaps)
    visited = 0
    while queue and visited < 4:       # limit requests to the same site
        sm = queue.pop(0)
        visited += 1
        try:
            r = fetcher.get(sm)
        except FetchError as exc:
            logger.warn("DISCOVERY", f"{src.name}: sitemap unavailable ({exc})")
            continue
        urls, children = _parse_sitemap(r.content)
        queue = children[:2] + queue
        for u in urls:
            title = u["title"] or u["loc"].rstrip("/").rsplit("/", 1)[-1].replace("-", " ")
            c = _candidate(src, u["loc"], title, "sitemap", u["date"], "", u["image"])
            if c:
                out.append(c)
    out.sort(key=lambda a: a.publication_date or a.discovered_at, reverse=True)
    return out[: src.max_items * 2]


def from_sections(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    out: list[SourceArticle] = []
    for sec in src.sections:
        try:
            r = fetcher.get(sec)
        except FetchError as exc:
            logger.warn("DISCOVERY", f"{src.name}: section unavailable ({exc})")
            continue
        soup = BeautifulSoup(r.text, "lxml")
        seen: set[str] = set()
        for a in soup.find_all("a", href=True):
            url = urljoin(sec, a["href"])
            title = clean_text(a.get_text(" "))
            if len(title) < 20 or url in seen:
                continue
            seen.add(url)
            c = _candidate(src, url, title, sec, None, "", "")
            if c:
                out.append(c)
    return out[: src.max_items]


def discover(src: SourceDef, fetcher: PoliteFetcher) -> list[SourceArticle]:
    """RSS -> sitemap -> section pages. Stops at the first method that yields results."""
    for name, fn, has in (("RSS", from_rss, bool(src.rss)),
                          ("sitemap", from_sitemap, bool(src.sitemaps)),
                          ("section", from_sections, bool(src.sections))):
        if not has:
            continue
        found = fn(src, fetcher)
        if found:
            logger.log("DISCOVERY", f"{src.name}: {len(found)} candidates via {name}")
            return found[: src.max_items]
    logger.warn("DISCOVERY", f"{src.name}: no candidates found")
    return []
