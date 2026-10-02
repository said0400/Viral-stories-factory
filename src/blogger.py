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


def _https(u: str) -> str:
    """Ensure Blogger URL is https:// for Facebook's strict requirements."""
    u = (u or "").strip()
    if u.startswith("http://"):
        return "https://" + u[7:]
    return u


class BloggerClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

        if not cfg.google_refresh_token:
            raise BloggerError("GOOGLE_REFRESH_TOKEN is missing")

        if not cfg.google_client_id:
            raise BloggerError("GOOGLE_CLIENT_ID is missing")

        if not cfg.google_client_secret:
            raise BloggerError("GOOGLE_CLIENT_SECRET is missing")

        if not cfg.blogger_blog_id:
            raise BloggerError("BLOGGER_BLOG_ID is missing")

        creds = Credentials(
            None,
            refresh_token=cfg.google_refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=cfg.google_client_id,
            client_secret=cfg.google_client_secret,
            scopes=[SCOPE],
        )

        self.s = AuthorizedSession(creds)
        self.blog = cfg.blogger_blog_id

    # ---- reads -----------------------------------------------------------
    def get_post(self, post_id: str) -> dict | None:
        r = self.s.get(
            f"{API}/blogs/{self.blog}/posts/{post_id}",
            timeout=self.cfg.request_timeout,
        )

        if r.status_code == 404:
            return None

        if r.status_code >= 400:
            detail = ""
            try:
                detail = r.text[:500]
            except Exception:
                pass
            raise BloggerError(
                f"posts.get HTTP {r.status_code}"
                + (f": {detail}" if detail else "")
            )

        try:
            return r.json()
        except ValueError as exc:
            raise BloggerError("posts.get returned invalid JSON") from exc

    def find_by_story_id(self, story_id: str) -> BloggerResult | None:
        """
        Idempotency probe.

        Scan recent posts for our hidden story_id marker.
        Capped at 3 pages to avoid excessive network strain on busy blogs.
        """
        marker = f"story_id:{story_id}"
        token = ""

        for _ in range(3):
            params = {
                "maxResults": 50,
                "orderBy": "published",
                "fetchBodies": "true",
                "status": ["live", "draft"],
            }

            if token:
                params["pageToken"] = token

            try:
                r = self.s.get(
                    f"{API}/blogs/{self.blog}/posts",
                    params=params,
                    timeout=self.cfg.request_timeout,
                )
            except (requests.Timeout, requests.ConnectionError) as exc:
                raise BloggerError(
                    f"posts.list failed: {type(exc).__name__}"
                ) from exc

            if r.status_code >= 400:
                detail = ""
                try:
                    detail = r.text[:500]
                except Exception:
                    pass
                raise BloggerError(
                    f"posts.list HTTP {r.status_code}"
                    + (f": {detail}" if detail else "")
                )

            try:
                data = r.json()
            except ValueError as exc:
                raise BloggerError("posts.list returned invalid JSON") from exc

            for p in data.get("items", []):
                content = p.get("content") or ""

                # Only use the strict marker to avoid collision with similar titles.
                if marker in content:
                    if p.get("status", "LIVE").upper() == "LIVE" and p.get("url"):
                        return BloggerResult(
                            post_id=p["id"],
                            url=_https(p["url"]),
                            published_at=p.get("published", iso()),
                        )

            token = data.get("nextPageToken", "")
            if not token:
                break

        return None

    # ---- publish ---------------------------------------------------------
    def verify(self, post_id: str, retries: int = 3) -> BloggerResult:
        """
        Verify that Blogger actually exposes the post as LIVE and that a
        public URL is available. Handles Blogger consistency lag by retrying
        404s briefly.
        """
        for i in range(retries):
            p = self.get_post(post_id)

            if p and p.get("url") and p.get("status", "LIVE").upper() == "LIVE":
                return BloggerResult(
                    post_id=p["id"],
                    url=_https(p["url"]),
                    published_at=p.get("published", iso()),
                )

            if i < retries - 1:
                time.sleep(2 * (i + 1))

        raise BloggerError("post not live / URL missing after publish and lag-wait")

    def publish(
        self,
        *,
        story_id: str,
        title: str,
        html: str,
        labels: list[str],
        description: str,
    ) -> BloggerResult:
        """
        Publish one Blogger post idempotently.
        """
        existing = self.find_by_story_id(story_id)

        if existing:
            logger.log(
                "BLOGGER",
                "post already exists; reusing (no duplicate)",
            )
            return existing

        body = {
            "kind": "blogger#post",
            "title": title,
            "content": html,
            "labels": labels[:5],
            "customMetaData": description[:155],
        }

        last = ""

        for attempt in range(self.cfg.max_retries + 1):
            try:
                r = self.s.post(
                    f"{API}/blogs/{self.blog}/posts",
                    params={"isDraft": "false"},
                    json=body,
                    timeout=self.cfg.request_timeout,
                )

                if r.status_code < 300:
                    try:
                        data = r.json()
                    except ValueError as exc:
                        raise BloggerError(
                            "posts.insert succeeded but returned invalid JSON"
                        ) from exc

                    post_id = data.get("id")

                    if not post_id:
                        raise BloggerError(
                            "posts.insert succeeded but no post ID was returned"
                        )
                    
                    url = data.get("url")
                    if url and data.get("status", "LIVE").upper() == "LIVE":
                        return BloggerResult(
                            post_id=post_id,
                            url=_https(url),
                            published_at=data.get("published", iso()),
                        )
                    
                    # Fallback to verification if the insert response lacked the URL
                    return self.verify(post_id)

                last = f"HTTP {r.status_code}"

                if r.status_code in (400, 401, 403, 404):
                    detail = ""
                    try:
                        detail = r.text[:500]
                    except Exception:
                        pass

                    raise BloggerError(
                        f"posts.insert {last}"
                        + (f": {detail}" if detail else "")
                    )

                # Honor Retry-After if provided
                if r.status_code in (429, 503):
                    ra = r.headers.get("Retry-After", "").strip()
                    if ra.isdigit():
                        time.sleep(min(60, max(5, int(ra))))
                        continue

            except (requests.Timeout, requests.ConnectionError) as exc:
                last = type(exc).__name__

            if attempt < self.cfg.max_retries:
                time.sleep(min(30, 3 * 2 ** attempt))

                found = self.find_by_story_id(story_id)

                if found:
                    logger.log(
                        "BLOGGER",
                        "post found after uncertain response; not re-posting",
                    )
                    return found

                logger.warn(
                    "BLOGGER",
                    f"publish attempt {attempt + 1} failed ({last}); retrying",
                )

        found = self.find_by_story_id(story_id)

        if found:
            logger.log(
                "BLOGGER",
                "post found during final idempotency check; not re-posting",
            )
            return found

        raise BloggerError(f"publish failed: {last}")
