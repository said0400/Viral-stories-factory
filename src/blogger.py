"""Blogger API v3 (REST) with OAuth refresh token. Idempotent publishing."""
from __future__ import annotations

import time

import requests
from google.auth.transport.requests import AuthorizedSession
from google.oauth2.credentials import Credentials

from . import logger
from .config import Settings
from .models import BloggerResult
from .utils import iso

API = "https://www.googleapis.com/blogger/v3"
SCOPE = "https://www.googleapis.com/auth/blogger"


class BloggerError(Exception):
    pass


class BloggerClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        creds = Credentials(None, refresh_token=cfg.google_refresh_token,
                            token_uri="https://oauth2.googleapis.com/token",
                            client_id=cfg.google_client_id, client_secret=cfg.google_client_secret,
                            scopes=[SCOPE])
        self.s = AuthorizedSession(creds)
        self.blog = cfg.blogger_blog_id

    # ---- reads -----------------------------------------------------------
    def get_post(self, post_id: str) -> dict | None:
        r = self.s.get(f"{API}/blogs/{self.blog}/posts/{post_id}", timeout=self.cfg.request_timeout)
        if r.status_code == 404:
            return None
        if r.status_code >= 400:
            raise BloggerError(f"posts.get HTTP {r.status_code}")
        return r.json()

    def find_by_story_id(self, story_id: str, title: str) -> BloggerResult | None:
        """Idempotency probe: scan recent posts for our hidden story_id marker (or exact title)."""
        marker = f"story_id:{story_id}"
        token = ""
        for _ in range(3):  # up to ~60 recent posts
            params = {"maxResults": 20, "orderBy": "published", "fetchBodies": "true",
                      "status": ["live", "draft"]}
            if token:
                params["pageToken"] = token
            r = self.s.get(f"{API}/blogs/{self.blog}/posts", params=params, timeout=self.cfg.request_timeout)
            if r.status_code >= 400:
                raise BloggerError(f"posts.list HTTP {r.status_code}")
            data = r.json()
            for p in data.get("items", []):
                if marker in (p.get("content") or "") or (p.get("title") or "").strip() == title.strip():
                    if p.get("status", "LIVE") == "LIVE" and p.get("url"):
                        return BloggerResult(post_id=p["id"], url=p["url"],
                                             published_at=p.get("published", iso()))
            token = data.get("nextPageToken", "")
            if not token:
                break
        return None

    # ---- publish ---------------------------------------------------------
    def verify(self, post_id: str) -> BloggerResult:
        p = self.get_post(post_id)
        if not p or not p.get("url") or p.get("status", "LIVE") != "LIVE":
            raise BloggerError("post not live / URL missing after publish")
        return BloggerResult(post_id=p["id"], url=p["url"], published_at=p.get("published", iso()))

    def publish(self, *, story_id: str, title: str, html: str, labels: list[str],
                description: str) -> BloggerResult:
        existing = self.find_by_story_id(story_id, title)
        if existing:
            logger.log("BLOGGER", "post already exists; reusing (no duplicate)")
            return existing
        body = {"kind": "blogger#post", "title": title, "content": html, "labels": labels[:5],
                "customMetaData": description[:155]}
        last = ""
        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.s.post(f"{API}/blogs/{self.blog}/posts", params={"isDraft": "false"},
                                json=body, timeout=self.cfg.request_timeout)
                if r.status_code < 300:
                    return self.verify(r.json()["id"])
                last = f"HTTP {r.status_code}"
                if r.status_code in (400, 401, 403, 404):  # not retryable
                    raise BloggerError(f"posts.insert {last}")
            except (requests.Timeout, requests.ConnectionError) as exc:
                last = type(exc).__name__
            if attempt < self.cfg.max_retries:
                # A timeout does not mean failure: look for the post BEFORE retrying.
                time.sleep(min(30, 3 * 2 ** attempt))
                found = self.find_by_story_id(story_id, title)
                if found:
                    logger.log("BLOGGER", "post found after uncertain response; not re-posting")
                    return found
                logger.warn("BLOGGER", f"publish attempt {attempt + 1} failed ({last}); retrying")
        raise BloggerError(f"publish failed: {last}")
