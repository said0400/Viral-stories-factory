"""Shared helpers: URL normalisation, IDs, retry, polite fetching, git."""
from __future__ import annotations

import functools
import hashlib
import json
import os
import re
import subprocess
import tempfile
import time
from datetime import datetime, timezone
from difflib import SequenceMatcher
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Iterable
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse
from urllib.robotparser import RobotFileParser

import requests

from . import logger

TRACKING_PARAMS = {
    "fbclid", "gclid", "dclid", "msclkid", "mc_cid", "mc_eid", "igshid", "ref", "ref_src",
    "ref_url", "cmpid", "cid", "ito", "ns_campaign", "ns_mchannel", "ns_source", "ns_linkname",
    "ncid", "taid", "_ga", "yclid", "campaign_id", "source", "src", "share",
}


# --------------------------------------------------------------------------- URLs
def normalize_url(url: str) -> str:
    """Lower-case host, drop fragments, tracking params and trailing slash."""
    url = (url or "").strip()
    if not url:
        return ""
    p = urlparse(url)
    scheme = "https" if p.scheme in ("http", "https", "") else p.scheme
    host = p.netloc.lower()
    if host.startswith("www."):
        host = host[4:]
    q = [(k, v) for k, v in parse_qsl(p.query, keep_blank_values=False)
         if not k.lower().startswith("utm_") and k.lower() not in TRACKING_PARAMS]
    q.sort()
    path = re.sub(r"/{2,}", "/", p.path)
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunparse((scheme, host, path, "", urlencode(q), ""))


# --------------------------------------------------------------------------- IDs / hashes
def sha256_hex(data: str | bytes) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def make_story_id(source: str, normalized_url: str) -> str:
    """Stable ID: SHA-256(source|normalized_url), first 20 hex chars."""
    return sha256_hex(f"{source.strip().lower()}|{normalized_url}")[:20]


# --------------------------------------------------------------------------- text
def normalize_title(t: str) -> str:
    t = (t or "").lower()
    t = re.sub(r"[^\w\s]", " ", t, flags=re.UNICODE)
    return re.sub(r"\s+", " ", t).strip()


def title_similarity(a: str, b: str) -> float:
    na, nb = normalize_title(a), normalize_title(b)
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def clean_text(s: str, limit: int | None = None) -> str:
    s = re.sub(r"\s+", " ", s or "").strip()
    return s[:limit] if limit else s


# --------------------------------------------------------------------------- time
def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: datetime | None = None) -> str:
    return (dt or utcnow()).astimezone(timezone.utc).isoformat(timespec="seconds")


def parse_datetime(value: Any) -> datetime | None:
    """Parse RFC-2822 / ISO-8601 / struct_time into an aware UTC datetime."""
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        dt = value
    elif isinstance(value, time.struct_time):
        dt = datetime(*value[:6], tzinfo=timezone.utc)
    else:
        s = str(value).strip()
        dt = None
        try:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
        except ValueError:
            try:
                dt = parsedate_to_datetime(s)
            except (TypeError, ValueError):
                return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# --------------------------------------------------------------------------- files
def atomic_write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


# --------------------------------------------------------------------------- retry
def retry(max_retries: int = 3, base_delay: float = 2.0, max_delay: float = 60.0,
          exceptions: tuple[type[BaseException], ...] = (Exception,),
          tag: str = "RETRY") -> Callable:
    """Exponential backoff decorator. Never retries forever."""
    def deco(fn: Callable) -> Callable:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            attempt = 0
            while True:
                try:
                    return fn(*args, **kwargs)
                except exceptions as exc:  # noqa: PERF203
                    attempt += 1
                    if attempt > max_retries:
                        raise
                    delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
                    logger.warn(tag, f"{fn.__name__} failed ({type(exc).__name__}); "
                                     f"retry {attempt}/{max_retries} in {delay:.0f}s")
                    time.sleep(delay)
        return wrapper
    return deco


# --------------------------------------------------------------------------- polite fetching
class FetchError(Exception):
    pass


class RobotsDisallowed(FetchError):
    pass


class PoliteFetcher:
    """requests wrapper: robots.txt, per-host delay, timeouts, status handling."""

    def __init__(self, user_agent: str, timeout: int, per_host_delay: float = 2.0,
                 respect_robots: bool = True) -> None:
        self.ua = user_agent
        self.timeout = timeout
        self.delay = per_host_delay
        self.respect_robots = respect_robots
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept-Language": "en,*;q=0.5"})
        self._robots: dict[str, RobotFileParser | None] = {}
        self._last: dict[str, float] = {}

    def _robot_for(self, url: str) -> RobotFileParser | None:
        p = urlparse(url)
        base = f"{p.scheme}://{p.netloc}"
        if base in self._robots:
            return self._robots[base]
        rp: RobotFileParser | None = RobotFileParser()
        try:
            r = self.session.get(base + "/robots.txt", timeout=self.timeout)
            if r.status_code == 200:
                rp.parse(r.text.splitlines())
            elif 400 <= r.status_code < 500:
                rp = None  # no robots file -> allowed
            else:
                rp.parse(["User-agent: *", "Disallow: /"])  # server trouble -> be conservative
        except requests.RequestException:
            rp.parse(["User-agent: *", "Disallow: /"])
        self._robots[base] = rp
        return rp

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        rp = self._robot_for(url)
        return True if rp is None else rp.can_fetch(self.ua, url)

    def crawl_delay(self, url: str) -> float:
        rp = self._robot_for(url) if self.respect_robots else None
        d = rp.crawl_delay(self.ua) if rp else None
        return max(self.delay, float(d)) if d else self.delay

    def get(self, url: str, *, binary: bool = False, check_robots: bool = True) -> requests.Response:
        if check_robots and not self.allowed(url):
            raise RobotsDisallowed(f"robots.txt disallows {url}")
        host = urlparse(url).netloc
        wait = self.crawl_delay(url) - (time.monotonic() - self._last.get(host, 0))
        if wait > 0:
            time.sleep(wait)
        for attempt in range(2):
            try:
                r = self.session.get(url, timeout=self.timeout)
            except requests.RequestException as exc:
                self._last[host] = time.monotonic()
                raise FetchError(f"{type(exc).__name__}: {url}") from exc
            self._last[host] = time.monotonic()
            if r.status_code in (429, 503) and attempt == 0:
                ra = r.headers.get("Retry-After", "")
                time.sleep(min(30, int(ra)) if ra.isdigit() else 10)
                continue
            if r.status_code >= 400:
                raise FetchError(f"HTTP {r.status_code}: {url}")
            return r
        raise FetchError(f"rate limited: {url}")


# --------------------------------------------------------------------------- git
def run_git(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], capture_output=True, text=True, check=check)


def git_commit_and_push(paths: Iterable[str | Path], message: str, retries: int = 3) -> bool:
    """Stage paths, commit only when something changed, push with rebase-retry."""
    try:
        run_git(["add", "--", *[str(p) for p in paths]])
        if run_git(["diff", "--cached", "--quiet"], check=False).returncode == 0:
            return False  # nothing changed -> no empty commit
        run_git(["commit", "-m", message])
        for i in range(1, retries + 1):
            r = run_git(["push"], check=False)
            if r.returncode == 0:
                return True
            logger.warn("GIT", f"push failed (attempt {i}/{retries}); rebasing")
            rb = run_git(["pull", "--rebase", "--autostash"], check=False)
            if rb.returncode != 0:
                run_git(["rebase", "--abort"], check=False)
                logger.error("GIT", "rebase conflict; aborting push")
                return False
            time.sleep(2 * i)
        return False
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("GIT", f"git operation failed: {type(exc).__name__}")
        return False
