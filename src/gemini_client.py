# src/gemini_client.py
"""Wrapper around the Google GenAI SDK (`google-genai`): JSON + image generation."""
from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel

from . import logger
from .config import Settings

T = TypeVar("T", bound=BaseModel)

_RETRYABLE_CODES = {408, 429, 500, 502, 503, 504}
_RETRYABLE_WORDS = (
    "resource_exhausted",
    "rate limit",
    "rate_limit",
    "too many requests",
    "temporarily unavailable",
    "service unavailable",
    "overloaded",
    "deadline exceeded",
    "timeout",
    "timed out",
    "connection reset",
    "connection error",
)


class GeminiError(Exception):
    """Base exception for Gemini failures."""


class ImageGenError(GeminiError):
    """Image generation failed."""


class ImageQuotaError(ImageGenError):
    """The image provider has no quota left (retrying is pointless)."""


class GeminiClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._image_client: Any = None

        if cfg.gemini_api_key:
            try:
                self._client = self._make_client(cfg.gemini_api_key, cfg.llm_timeout)
            except Exception as exc:
                logger.error("GEMINI", f"Failed to initialise google-genai client ({type(exc).__name__})")
                self._client = None

    # ------------------------------------------------------------------
    @staticmethod
    def _make_client(api_key: str, timeout_seconds: int) -> Any:
        from google import genai
        from google.genai import types

        try:
            options = types.HttpOptions(timeout=int(timeout_seconds) * 1000)  # milliseconds
            return genai.Client(api_key=api_key, http_options=options)
        except Exception:
            return genai.Client(api_key=api_key)

    def is_configured(self) -> bool:
        return self._client is not None

    @staticmethod
    def _is_quota_exhausted(exc: BaseException) -> bool:
        """Zero quota (limit: 0) or an exhausted daily quota: retrying cannot help."""
        low = str(exc).lower()
        return "limit: 0" in low or "perday" in low

    @classmethod
    def _retryable(cls, exc: BaseException) -> bool:
        if cls._is_quota_exhausted(exc):
            return False

        for attr in ("code", "status_code"):
            raw = getattr(exc, attr, None)
            try:
                code = int(raw)
            except (TypeError, ValueError):
                continue
            return code in _RETRYABLE_CODES

        name = type(exc).__name__.lower()
        if "timeout" in name or "connect" in name:
            return True

        msg = str(exc).lower()
        return any(word in msg for word in _RETRYABLE_WORDS)

    def _delays(self) -> list[float]:
        n = max(0, int(self.cfg.max_retries))
        base = [2.0, 5.0, 10.0, 20.0] + [20.0] * max(0, n - 4)
        return base[:n]

    # ------------------------------------------------------------------
    # JSON generation
    @staticmethod
    def _parse(response: Any, schema: type[T], model: str) -> T:
        parsed = getattr(response, "parsed", None)

        if parsed is not None:
            if isinstance(parsed, schema):
                return parsed
            try:
                return schema.model_validate(parsed)
            except Exception as exc:
                raise GeminiError(
                    f"Structured response could not be validated against {schema.__name__}: {exc}"
                ) from exc

        text = (getattr(response, "text", "") or "").strip()

        if not text:
            raise GeminiError(f"Empty response from Gemini model {model}")

        try:
            return schema.model_validate_json(text)
        except Exception as exc:
            raise GeminiError(f"Invalid JSON for {schema.__name__}: {exc}") from exc

    def _generate_json_model(
        self,
        model: str,
        prompt: str,
        schema: type[T],
        *,
        images: list[tuple[bytes, str]] | None,
        system: str | None,
        temperature: float,
        tag: str,
    ) -> T:
        if not self._client:
            raise GeminiError("Gemini client is not configured (missing GEMINI_API_KEY)")

        from google.genai import types

        contents: list[Any] = []

        for img_bytes, mime in images or []:
            try:
                contents.append(types.Part.from_bytes(data=img_bytes, mime_type=mime))
            except Exception as exc:
                raise GeminiError(f"Failed to convert image for Gemini prompt: {exc}") from exc

        contents.append(prompt)

        kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }
        if system:
            kwargs["system_instruction"] = system

        try:
            config = types.GenerateContentConfig(**kwargs)
        except Exception as exc:
            raise GeminiError(f"Failed to build Gemini config: {exc}") from exc

        delays = self._delays()
        attempts = len(delays) + 1

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as exc:
                if attempt < attempts and self._retryable(exc):
                    delay = delays[attempt - 1]
                    logger.warn(
                        tag,
                        f"Transient API error on {model} (attempt {attempt}/{attempts}), "
                        f"retrying in {delay}s: {exc}",
                    )
                    time.sleep(delay)
                    continue

                logger.error(tag, f"API error on {model} (attempt {attempt}/{attempts}): {exc}")
                raise GeminiError(f"API error on {model}: {exc}") from exc

            try:
                return self._parse(response, schema, model)
            except GeminiError as exc:
                # An empty / malformed model answer is worth exactly one more try.
                if attempt < min(attempts, 2):
                    delay = delays[attempt - 1]
                    logger.warn(tag, f"Bad output from {model} (attempt {attempt}): {exc}; retrying in {delay}s")
                    time.sleep(delay)
                    continue

                logger.error(tag, f"Generation failed on {model}: {exc}")
                raise

        raise GeminiError(f"Failed to generate valid JSON from model {model}")

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        images: list[tuple[bytes, str]] | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        tag: str = "GEMINI",
    ) -> T:
        """Structured output validated against a Pydantic schema (primary model, then fallback)."""
        models = [self.cfg.gemini_model]
        fallback = (self.cfg.gemini_fallback_model or "").strip()

        if fallback and fallback not in models:
            models.append(fallback)

        last: GeminiError | None = None

        for index, name in enumerate(models):
            try:
                return self._generate_json_model(
                    name,
                    prompt,
                    schema,
                    images=images,
                    system=system,
                    temperature=temperature,
                    tag=tag,
                )
            except GeminiError as exc:
                last = exc
                if index < len(models) - 1:
                    logger.warn(tag, f"Model '{name}' failed ({exc}). Falling back to '{models[index + 1]}'")
                    continue
                raise

        raise last or GeminiError("Gemini JSON generation failed across all models")

    # ------------------------------------------------------------------
    # Image generation (IMAGE_PROVIDER=gemini only)
    def _get_image_client(self) -> Any:
        if self._image_client is None:
            key = self.cfg.image_api_key or self.cfg.gemini_api_key
            try:
                self._image_client = self._make_client(key, self.cfg.image_request_timeout)
            except Exception as exc:
                logger.warn("GEMINI_IMAGE", f"image client init failed ({type(exc).__name__}); using primary client")
                self._image_client = self._client
        return self._image_client

    @staticmethod
    def _gemini_image(
        client: Any,
        model: str,
        prompt: str,
        references: list[tuple[bytes, str]] | None,
        aspect_ratio: str,
    ) -> tuple[bytes, str]:
        from google.genai import types

        contents: Any = prompt

        if references:
            parts: list[Any] = [
                types.Part.from_bytes(data=data, mime_type=mime) for data, mime in references
            ]
            parts.append(prompt)
            contents = parts

        kwargs: dict[str, Any] = {"response_modalities": ["IMAGE"]}

        try:
            kwargs["image_config"] = types.ImageConfig(aspect_ratio=aspect_ratio)
            config = types.GenerateContentConfig(**kwargs)
        except Exception:
            # Older SDK: the aspect ratio is already stated inside the prompt.
            kwargs.pop("image_config", None)
            config = types.GenerateContentConfig(**kwargs)

        response = client.models.generate_content(model=model, contents=contents, config=config)

        parts_out = getattr(response, "parts", None)

        if not parts_out:
            for cand in getattr(response, "candidates", None) or []:
                parts_out = getattr(getattr(cand, "content", None), "parts", None)
                if parts_out:
                    break

        for part in parts_out or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None

            if data:
                mime = getattr(inline, "mime_type", None) or "image/png"
                return bytes(data), str(mime)

        raise ImageGenError(f"No image bytes returned by {model}")

    def generate_image(
        self,
        prompt: str,
        *,
        references: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        tag: str = "GEMINI_IMAGE",
    ) -> tuple[bytes, str]:
        """Return (image_bytes, mime_type). Raises ImageQuotaError / ImageGenError."""
        if not self._client:
            raise ImageGenError("Gemini client is not configured")

        model = (self.cfg.gemini_image_model or "").strip()

        if not model:
            raise ImageGenError("GEMINI_IMAGE_MODEL is empty")

        if "imagen" in model.lower():
            raise ImageGenError("Imagen models are not supported by the Gemini Developer API; use a gemini-*-image model")

        client = self._get_image_client()

        delays = self._delays()
        attempts = len(delays) + 1
        last: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                logger.log(tag, f"Calling generate_content on {model} (attempt {attempt}/{attempts})")
                return self._gemini_image(client, model, prompt, references, aspect_ratio)

            except Exception as exc:
                last = exc

                if self._is_quota_exhausted(exc):
                    raise ImageQuotaError(
                        f"No usable quota for {model} (limit 0 or daily quota exhausted); billing may be required"
                    ) from exc

                if attempt < attempts and self._retryable(exc):
                    delay = delays[attempt - 1]
                    logger.warn(
                        tag,
                        f"Transient image error on {model} (attempt {attempt}/{attempts}), "
                        f"retrying in {delay}s: {exc}",
                    )
                    time.sleep(delay)
                    continue

                break

        raise ImageGenError(f"Image generation failed on {model}: {last}") from last
