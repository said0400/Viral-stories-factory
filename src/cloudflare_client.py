# src/cloudflare_client.py
"""Cloudflare Workers AI image generation (FLUX.2 klein by default; reference images supported)."""
from __future__ import annotations

import base64
import io
import re
import time
from typing import Any

import requests
from PIL import Image

from . import logger
from .config import Settings
from .gemini_client import ImageGenError, ImageQuotaError

API_BASE = "https://api.cloudflare.com/client/v4/accounts"

MAX_REFERENCES = 3          # FLUX.2 supports input_image_0 .. input_image_3
REFERENCE_MAX_SIDE = 480    # Cloudflare documents reference images below 512x512

_RETRYABLE_STATUS = {408, 500, 502, 503, 504}
_ACCOUNT_RE = re.compile(r"^[A-Za-z0-9]+$")
_MODEL_RE = re.compile(r"^@(cf|hf)/[A-Za-z0-9._\-]+/[A-Za-z0-9._\-]+$")


def _size_for(aspect_ratio: str, long_side: int = 1536) -> tuple[int, int]:
    """Return API-valid dimensions (multiples of 16), preserving aspect ratio."""
    try:
        a, b = str(aspect_ratio).split(":", 1)
        wr, hr = float(a), float(b)
        if wr <= 0 or hr <= 0:
            raise ValueError
    except ValueError:
        return 1024, 1024

    long_side = max(1024, min(1920, int(long_side)))
    long_side = int(round(long_side / 16.0)) * 16
    long_side = max(1024, min(1920, long_side))

    if wr >= hr:
        width = long_side
        height = int(round(long_side * hr / wr / 16.0)) * 16
    else:
        height = long_side
        width = int(round(long_side * wr / hr / 16.0)) * 16

    return max(256, width), max(256, height)


def _prepare_reference(data: bytes, mime: str) -> tuple[bytes, str]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")
        img.thumbnail((REFERENCE_MAX_SIDE, REFERENCE_MAX_SIDE))

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=88)

        return buf.getvalue(), "image/jpeg"
    except Exception:
        return data, mime or "image/jpeg"


def _short(text: str, limit: int = 300) -> str:
    return " ".join(str(text or "").split())[:limit]


def _is_quota(status: int, text: str) -> bool:
    """Heuristic: Cloudflare daily-neuron exhaustion (exact wording is not guaranteed)."""
    low = str(text or "").lower()

    if "neuron" in low or "4006" in low or ("daily" in low and "allocation" in low):
        return True

    return status == 429 and any(word in low for word in ("quota", "allocation", "exceeded the"))


def _sniff_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return "image/jpeg"


class CloudflareClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._warned_refs = False

    # ------------------------------------------------------------------
    def _url_and_headers(self) -> tuple[str, dict[str, str]]:
        account = self.cfg.cloudflare_account_id
        token = self.cfg.cloudflare_api_token
        model = (self.cfg.cloudflare_image_model or "").strip()

        if not account or not token:
            raise ImageGenError("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN is missing")

        if not _ACCOUNT_RE.fullmatch(account):
            raise ImageGenError("CLOUDFLARE_ACCOUNT_ID has an invalid format")

        if not _MODEL_RE.fullmatch(model):
            raise ImageGenError("CLOUDFLARE_IMAGE_MODEL has an invalid format")

        return f"{API_BASE}/{account}/ai/run/{model}", {"Authorization": f"Bearer {token}"}

    @staticmethod
    def _extract_image(resp: requests.Response) -> tuple[bytes, str]:
        ctype = (resp.headers.get("Content-Type") or "").split(";", 1)[0].strip().lower()

        if ctype.startswith("image/"):
            return resp.content, ctype

        try:
            body: Any = resp.json()
        except ValueError as exc:
            raise ImageGenError("Cloudflare returned a response that is neither an image nor JSON") from exc

        if isinstance(body, dict) and body.get("success") is False:
            raise ImageGenError(f"Cloudflare reported failure: {_short(body.get('errors'))}")

        result = body.get("result", body) if isinstance(body, dict) else None
        raw = result.get("image") if isinstance(result, dict) else None

        if not raw:
            raise ImageGenError("Cloudflare response contained no image")

        try:
            data = base64.b64decode(raw)
        except Exception as exc:  # noqa: BLE001
            raise ImageGenError("Cloudflare image was not valid base64") from exc

        return data, _sniff_mime(data)

    # ------------------------------------------------------------------
    def generate_image(
        self,
        prompt: str,
        *,
        references: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "16:9",
        tag: str = "CF_IMAGE",
    ) -> tuple[bytes, str]:
        """Return (image_bytes, mime_type). Raises ImageQuotaError / ImageGenError."""
        url, headers = self._url_and_headers()

        model = self.cfg.cloudflare_image_model
        multipart_model = "flux-2" in model.lower()
        width, height = _size_for(aspect_ratio, self.cfg.image_long_side)

        refs: list[tuple[bytes, str]] = []

        if references:
            if multipart_model:
                refs = [_prepare_reference(d, m) for d, m in references[:MAX_REFERENCES]]
            elif not self._warned_refs:
                logger.warn(tag, f"{model} does not accept reference images; ignoring them")
                self._warned_refs = True

        delays = [5.0, 15.0][: min(max(0, int(self.cfg.max_retries)), 2)]
        attempts = len(delays) + 1
        timeout = (15, self.cfg.cloudflare_timeout)
        last = "unknown error"

        for attempt in range(1, attempts + 1):
            logger.log(
                tag,
                f"Calling Cloudflare {model.rsplit('/', 1)[-1]} "
                f"({width}x{height}, refs={len(refs)}, attempt {attempt}/{attempts})",
            )

            started = time.time()

            try:
                if multipart_model:
                    files: dict[str, Any] = {
                        "prompt": (None, prompt),
                        "width": (None, str(width)),
                        "height": (None, str(height)),
                    }
                    if "flux-2-dev" in model.lower():
                        files["steps"] = (None, str(self.cfg.cloudflare_image_steps))
                    for i, (data, mime) in enumerate(refs):
                        files[f"input_image_{i}"] = (f"reference_{i}.jpg", data, mime)

                    resp = requests.post(url, headers=headers, files=files, timeout=timeout)
                else:
                    resp = requests.post(
                        url, headers=headers, json={"prompt": prompt, "steps": 4}, timeout=timeout
                    )

            except requests.Timeout:
                last = f"timed out after {self.cfg.cloudflare_timeout}s"
            except requests.RequestException as exc:
                last = type(exc).__name__
            else:
                status = resp.status_code

                if status < 400:
                    data, mime = self._extract_image(resp)
                    logger.log(tag, f"Cloudflare image received in {time.time() - started:.1f}s")
                    return data, mime

                text = resp.text

                if _is_quota(status, text):
                    raise ImageQuotaError(
                        f"Cloudflare daily free allocation appears exhausted (HTTP {status}): {_short(text, 200)}"
                    )

                if status in (401, 403):
                    raise ImageGenError("Cloudflare authentication failed (check CLOUDFLARE_API_TOKEN permissions)")

                if status == 404:
                    raise ImageGenError("Cloudflare model or account not found (check CLOUDFLARE_IMAGE_MODEL / ACCOUNT_ID)")

                last = f"HTTP {status}: {_short(text, 200)}"

                if status not in _RETRYABLE_STATUS and status != 429:
                    raise ImageGenError(f"Cloudflare image generation failed ({last})")

            if attempt < attempts:
                delay = delays[attempt - 1]
                logger.warn(tag, f"Cloudflare attempt {attempt}/{attempts} failed ({last}); retrying in {delay}s")
                time.sleep(delay)

        raise ImageGenError(f"Cloudflare image generation failed: {last}")
