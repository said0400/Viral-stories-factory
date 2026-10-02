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

    def find_by_story_id(self, story_id: str, title: str) -> BloggerResult | None:
        """
        Idempotency probe.

        Scan recent posts for our hidden story_id marker first, while also
        retaining exact-title matching as a backward-compatible fallback.
        """
        marker = f"story_id:{story_id}"
        token = ""

        # Scan more than the original 60 posts so a delayed run or a busy
        # Blogger account is less likely to create a duplicate.
        for _ in range(10):
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
                post_title = (p.get("title") or "").strip()

                if marker in content or post_title == title.strip():
                    if p.get("status", "LIVE").upper() == "LIVE" and p.get("url"):
                        return BloggerResult(
                            post_id=p["id"],
                            url=p["url"],
                            published_at=p.get("published", iso()),
                        )

            token = data.get("nextPageToken", "")
            if not token:
                break

        return None

    # ---- publish ---------------------------------------------------------
    def verify(self, post_id: str) -> BloggerResult:
        """
        Verify that Blogger actually exposes the post as LIVE and that a
        public URL is available.
        """
        p = self.get_post(post_id)

        if (
            not p
            or not p.get("url")
            or p.get("status", "LIVE").upper() != "LIVE"
        ):
            raise BloggerError("post not live / URL missing after publish")

        return BloggerResult(
            post_id=p["id"],
            url=p["url"],
            published_at=p.get("published", iso()),
        )

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

        The story marker is expected to already exist inside the generated
        HTML. If an earlier request succeeded but the response was lost,
        the idempotency probe finds the existing post before another insert.
        """
        existing = self.find_by_story_id(story_id, title)

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

                    return self.verify(post_id)

                last = f"HTTP {r.status_code}"

                # Authentication, permission, malformed-request and
                # not-found errors are not useful to retry immediately.
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

            except (requests.Timeout, requests.ConnectionError) as exc:
                last = type(exc).__name__

            if attempt < self.cfg.max_retries:
                # A timeout/connection failure does NOT prove that Blogger
                # rejected the request. Wait briefly, then perform another
                # idempotency probe before sending another insert.
                time.sleep(min(30, 3 * 2 ** attempt))

                found = self.find_by_story_id(story_id, title)

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

        # Final safety probe: the last request may have reached Blogger even
        # if the client did not receive a successful response.
        found = self.find_by_story_id(story_id, title)

        if found:
            logger.log(
                "BLOGGER",
                "post found during final idempotency check; not re-posting",
            )
            return found

        raise BloggerError(f"publish failed: {last}")
