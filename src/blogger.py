"""Blogger API v3 (REST) with OAuth refresh token. Idempotent publishing."""
from __future__ import annotations

import time
from urllib.parse import urlparse

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


def _valid_public_url(u: str) -> bool:
    """
    Validate that a Blogger URL is a usable public HTTPS URL.

    Blogger itself is the source of this URL, but validating it here prevents
    malformed or incomplete API responses from being treated as successful
    publication results.
    """
    u = _https(u)

    if not u:
        return False

    try:
        parsed = urlparse(u)
    except ValueError:
        return False

    if parsed.scheme != "https":
        return False

    if not parsed.netloc:
        return False

    if parsed.username or parsed.password:
        return False

    return True


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
        post_id = str(post_id or "").strip()

        if not post_id:
            raise BloggerError("post_id is missing")

        try:
            r = self.s.get(
                f"{API}/blogs/{self.blog}/posts/{post_id}",
                timeout=self.cfg.request_timeout,
            )
        except (requests.Timeout, requests.ConnectionError) as exc:
            raise BloggerError(
                f"posts.get failed: {type(exc).__name__}"
            ) from exc

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

        Only a strict story_id marker is accepted. Title matching is
        intentionally not used because two independent stories can have
        similar or identical titles.
        """
        story_id = str(story_id or "").strip()

        if not story_id:
            raise BloggerError("story_id is missing")

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
                raise BloggerError(
                    "posts.list returned invalid JSON"
                ) from exc

            items = data.get("items", [])

            if not isinstance(items, list):
                items = []

            for p in items:
                if not isinstance(p, dict):
                    continue

                content = p.get("content") or ""

                if marker not in content:
                    continue

                status = str(p.get("status", "LIVE")).upper()
                post_url = _https(p.get("url", ""))

                # A draft with our marker is deliberately not reused as a
                # successful publication. We only return a real LIVE post.
                if status != "LIVE":
                    continue

                if not _valid_public_url(post_url):
                    continue

                post_id = str(p.get("id") or "").strip()

                if not post_id:
                    continue

                return BloggerResult(
                    post_id=post_id,
                    url=post_url,
                    published_at=p.get("published", iso()),
                )

            token = str(data.get("nextPageToken") or "").strip()

            if not token:
                break

        return None

    # ---- publish verification ------------------------------------------

    def verify(self, post_id: str, retries: int = 4) -> BloggerResult:
        """
        Verify that Blogger actually exposes the post as LIVE and that a
        public HTTPS URL is available.

        Blogger can briefly return incomplete information immediately after
        posts.insert, so this method performs a bounded consistency wait.
        """
        post_id = str(post_id or "").strip()

        if not post_id:
            raise BloggerError("post_id is missing during verification")

        retries = max(1, int(retries))

        last_error = ""

        for i in range(retries):
            try:
                p = self.get_post(post_id)
            except BloggerError as exc:
                last_error = str(exc)

                if i >= retries - 1:
                    break

                time.sleep(min(10, 2 * (i + 1)))
                continue

            if p:
                status = str(p.get("status", "LIVE")).upper()
                url = _https(p.get("url", ""))

                if (
                    status == "LIVE"
                    and _valid_public_url(url)
                    and p.get("id")
                ):
                    return BloggerResult(
                        post_id=str(p["id"]),
                        url=url,
                        published_at=p.get("published", iso()),
                    )

                last_error = (
                    f"status={status or 'missing'}, "
                    f"url={'present' if url else 'missing'}"
                )
            else:
                last_error = "post not found yet"

            if i < retries - 1:
                time.sleep(min(10, 2 * (i + 1)))

        raise BloggerError(
            "post not live / valid HTTPS URL missing after "
            f"publish verification retries"
            + (f" ({last_error})" if last_error else "")
        )

    # ---- publish ---------------------------------------------------------

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

        The story_id marker must already be present in `html`. It is the
        authoritative idempotency key used to prevent duplicate posts after
        uncertain network responses or process interruptions.
        """
        story_id = str(story_id or "").strip()
        title = str(title or "").strip()
        html = str(html or "")
        description = str(description or "")

        if not story_id:
            raise BloggerError("story_id is missing")

        if not title:
            raise BloggerError("title is empty")

        if not html.strip():
            raise BloggerError("html is empty")

        marker = f"story_id:{story_id}"

        if marker not in html:
            raise BloggerError(
                "Blogger HTML is missing the required story_id marker"
            )

        existing = self.find_by_story_id(story_id)

        if existing:
            logger.log(
                "BLOGGER",
                "post already exists; reusing (no duplicate)",
            )
            return existing

        clean_labels = [
            str(label).strip()
            for label in (labels or [])
            if str(label).strip()
        ]

        body = {
            "kind": "blogger#post",
            "title": title,
            "content": html,
            "labels": clean_labels[:5],
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

                    post_id = str(data.get("id") or "").strip()

                    if not post_id:
                        raise BloggerError(
                            "posts.insert succeeded but no post ID was returned"
                        )

                    url = _https(data.get("url", ""))
                    status = str(
                        data.get("status", "LIVE")
                    ).upper()

                    if (
                        status == "LIVE"
                        and _valid_public_url(url)
                    ):
                        return BloggerResult(
                            post_id=post_id,
                            url=url,
                            published_at=data.get("published", iso()),
                        )

                    # Blogger may have successfully created the post while
                    # the immediate response still lacks a final public URL.
                    # Verify it before deciding that publication failed.
                    try:
                        return self.verify(post_id)
                    except BloggerError as verify_error:
                        last = str(verify_error)

                        # Critical idempotency protection:
                        # the insert may have succeeded even though verify()
                        # temporarily failed. Search by story_id before ever
                        # attempting another insert.
                        found = self.find_by_story_id(story_id)

                        if found:
                            logger.log(
                                "BLOGGER",
                                "post found after verification failure; "
                                "reusing existing post",
                            )
                            return found

                        # No post was discoverable yet. Continue through the
                        # normal retry path rather than blindly inserting.
                        if attempt >= self.cfg.max_retries:
                            break

                else:
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

                    # Retry-After is particularly useful for 429/503.
                    if r.status_code in (429, 503):
                        retry_after = (
                            r.headers.get("Retry-After", "")
                            .strip()
                        )

                        if retry_after.isdigit():
                            delay = min(
                                60,
                                max(5, int(retry_after)),
                            )
                            time.sleep(delay)

                            # Before retrying the INSERT, always check whether
                            # Blogger actually created the post.
                            found = self.find_by_story_id(story_id)

                            if found:
                                logger.log(
                                    "BLOGGER",
                                    "post found after rate/server "
                                    "response; not re-posting",
                                )
                                return found

                            continue

            except (requests.Timeout, requests.ConnectionError) as exc:
                last = type(exc).__name__

            except BloggerError:
                # Non-transient API errors and validation failures should not
                # be swallowed and retried as if they were network failures.
                raise

            if attempt < self.cfg.max_retries:
                # Before every retry, check idempotency again. This protects
                # against the case where Blogger accepted the request but the
                # client did not receive the successful response.
                try:
                    found = self.find_by_story_id(story_id)
                except BloggerError as probe_error:
                    logger.warn(
                        "BLOGGER",
                        f"idempotency probe failed before retry: "
                        f"{probe_error}",
                    )
                    found = None

                if found:
                    logger.log(
                        "BLOGGER",
                        "post found after uncertain response; "
                        "not re-posting",
                    )
                    return found

                delay = min(30, 3 * 2 ** attempt)

                logger.warn(
                    "BLOGGER",
                    f"publish attempt {attempt + 1} failed "
                    f"({last or 'unknown error'}); retrying in {delay}s",
                )

                time.sleep(delay)

        # Final idempotency check before declaring publication failure.
        try:
            found = self.find_by_story_id(story_id)
        except BloggerError as probe_error:
            if last:
                last = f"{last}; final probe failed: {probe_error}"
            else:
                last = f"final probe failed: {probe_error}"
            found = None

        if found:
            logger.log(
                "BLOGGER",
                "post found during final idempotency check; "
                "not re-posting",
            )
            return found

        raise BloggerError(
            f"publish failed: {last or 'unknown error'}"
        )
