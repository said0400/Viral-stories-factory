"""Make generated images reachable by Blogger/Twilio (public URL, correct Content-Type)."""
from __future__ import annotations

import time
from pathlib import Path

import requests

from . import logger
from .config import Settings
from .utils import git_commit_and_push

MAX_WHATSAPP_IMAGE_BYTES = 5 * 1024 * 1024


def public_url_for(cfg: Settings, path: str | Path) -> str:
    """
    Build a public URL from the basename only.

    Filenames MUST already be unique per story (image_generator writes
    ``{story_id}_generated.jpg`` / ``{story_id}_facebook.jpg``).
    """
    base = (cfg.public_images_base or "").rstrip("/")
    return f"{base}/{Path(path).name}" if base else ""


def probe(url: str, timeout: int = 20) -> tuple[bool, str]:
    """GET check: 200, image/* Content-Type, size < 5MB.

    Uses GET (not HEAD) because some CDNs/raw hosts mishandle HEAD.
    Streams the body only when Content-Length is missing so large files
    never fully load into RAM.
    """
    if not url:
        return False, "empty public URL"

    try:
        with requests.get(
            url,
            timeout=timeout,
            stream=True,
            allow_redirects=True,
        ) as r:
            ctype = (
                (r.headers.get("Content-Type") or "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )

            content_length = r.headers.get("Content-Length")
            try:
                size = int(content_length) if content_length else 0
            except (TypeError, ValueError):
                size = 0

            if r.status_code != 200:
                return False, f"HTTP {r.status_code}"

            if not ctype.startswith("image/"):
                return False, f"Content-Type {ctype!r}"

            if size > MAX_WHATSAPP_IMAGE_BYTES:
                return False, "image larger than 5MB"

            if size == 0:
                total = 0
                for chunk in r.iter_content(chunk_size=64 * 1024):
                    if not chunk:
                        continue
                    total += len(chunk)
                    if total > MAX_WHATSAPP_IMAGE_BYTES:
                        return False, "image larger than 5MB"

            return True, ""

    except requests.RequestException as exc:
        return False, type(exc).__name__


def publish_images(
    cfg: Settings,
    paths: list[str],
    story_id: str,
    wait_seconds: int = 120,
) -> dict[str, str]:
    """
    Optionally push image files, then wait until public URLs respond.

    Returns ``{local_path: public_url}`` only for URLs that probe OK.

    Design notes
    ------------
    * Missing ``public_images_base`` → empty dict (local Export/ZIP still works).
    * Git push failure is logged and does NOT raise; we still probe in case
      files were already reachable from a previous push.
    * Workflow may also commit ``data/images`` at the end — possible extra
      commits / races are accepted when raw.githubusercontent.com must serve
      images *before* Blogger publish in the same run.
    """
    unique = list(dict.fromkeys(p for p in paths if p))

    if not unique:
        return {}

    if not cfg.public_images_base:
        logger.warn(
            "IMAGE",
            "no public image base URL "
            "(set IMAGE_PUBLIC_BASE_URL or run in GitHub Actions)",
        )
        return {}

    if cfg.git_push_enabled:
        try:
            ok = git_commit_and_push(
                unique,
                f"images: {story_id}",
            )
            if not ok:
                logger.warn(
                    "IMAGE",
                    "git push for images returned false "
                    "(conflict, auth, or nothing to commit); "
                    "will still probe public URLs",
                )
        except Exception as exc:
            # Never let Git kill the story pipeline here.
            logger.warn(
                "IMAGE",
                f"git push for images failed "
                f"({type(exc).__name__}); will still probe public URLs",
            )

    out: dict[str, str] = {}
    deadline = time.monotonic() + max(0, wait_seconds)

    for p in unique:
        url = public_url_for(cfg, p)

        if not url:
            logger.warn(
                "IMAGE",
                f"could not build public URL: {Path(p).name}",
            )
            continue

        why = ""

        # wait_seconds=0 still performs exactly one probe (good).
        while True:
            ok, why = probe(url)

            if ok:
                out[p] = url
                break

            if time.monotonic() >= deadline:
                logger.warn(
                    "IMAGE",
                    f"public URL not reachable ({why}): {Path(p).name}",
                )
                break

            time.sleep(8)

    return out
