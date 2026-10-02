"""All Gemini traffic lives here. Swap models via GEMINI_MODEL / GEMINI_IMAGE_MODEL."""
from __future__ import annotations

import time
from typing import TypeVar

from google import genai
from google.genai import errors as genai_errors
from google.genai import types
from pydantic import BaseModel, ValidationError

from . import logger
from .config import Settings

T = TypeVar("T", bound=BaseModel)

STRICT_SUFFIX = ("\n\nIMPORTANT: Reply with ONE valid JSON object that exactly matches the schema. "
                 "No markdown fences, no commentary, no trailing text.")


class GeminiError(Exception):
    pass


class ImageGenError(GeminiError):
    pass


class GeminiClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.client = genai.Client(api_key=cfg.gemini_api_key)
        self.image_client = (self.client if cfg.image_api_key == cfg.gemini_api_key
                             else genai.Client(api_key=cfg.image_api_key))

    # ------------------------------------------------------------------ helpers
    def _sleep(self, attempt: int) -> None:
        time.sleep(min(60, 2 * (2 ** attempt)))

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        code = getattr(exc, "code", None)
        return isinstance(exc, (genai_errors.ServerError,)) or code in (408, 429, 500, 502, 503, 504)

    # ------------------------------------------------------------------ JSON
    def generate_json(self, prompt: str, schema: type[T], *,
                      images: list[tuple[bytes, str]] | None = None,
                      system: str | None = None, temperature: float = 0.7,
                      tag: str = "GEMINI") -> T:
        """Structured output with validation. retry -> validate -> stricter retry -> fail."""
        parts: list = [types.Part.from_bytes(data=b, mime_type=m) for b, m in (images or [])]
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            text_prompt = prompt + (STRICT_SUFFIX if attempt >= 1 else "")
            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
                temperature=max(0.0, temperature - 0.2 * attempt),
                http_options=types.HttpOptions(timeout=self.cfg.request_timeout * 1000),
            )
            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.gemini_model, contents=[*parts, text_prompt], config=config)
                raw = (resp.text or "").strip()
                if not raw:
                    raise GeminiError("empty response (possible safety block or token limit)")
                if raw.startswith("```"):
                    raw = raw.strip("`").removeprefix("json").strip()
                return schema.model_validate_json(raw)
            except (ValidationError, GeminiError, ValueError) as exc:
                last = exc
                logger.warn(tag, f"invalid JSON/response (attempt {attempt + 1}): {type(exc).__name__}")
            except genai_errors.APIError as exc:
                last = exc
                if not self._retryable(exc):
                    raise GeminiError(f"non-retryable API error {getattr(exc, 'code', '?')}") from exc
                logger.warn(tag, f"API error {getattr(exc, 'code', '?')} (attempt {attempt + 1})")
            except Exception as exc:  # network / timeout
                last = exc
                logger.warn(tag, f"{type(exc).__name__} (attempt {attempt + 1})")
            if attempt < self.cfg.max_retries:
                self._sleep(attempt)
        raise GeminiError(f"Gemini JSON call failed after retries: {type(last).__name__}")

    # ------------------------------------------------------------------ images
    def generate_image(self, prompt: str, *, references: list[tuple[bytes, str]] | None = None,
                       aspect_ratio: str = "16:9") -> tuple[bytes, str]:
        """Return (bytes, mime). Uses reference images when provided (multimodal input)."""
        contents: list = [prompt]
        for b, m in (references or []):
            contents.append(types.Part.from_bytes(data=b, mime_type=m))
        base = dict(response_modalities=["IMAGE"],
                    http_options=types.HttpOptions(timeout=self.cfg.image_request_timeout * 1000))
        last: Exception | None = None
        for use_ratio in (True, False):  # some models reject image_config
            try:
                cfg = types.GenerateContentConfig(
                    **base, **({"image_config": types.ImageConfig(aspect_ratio=aspect_ratio)}
                               if use_ratio else {}))
                resp = self.image_client.models.generate_content(
                    model=self.cfg.gemini_image_model, contents=contents, config=cfg)
                for cand in resp.candidates or []:
                    for part in (cand.content.parts if cand.content and cand.content.parts else []):
                        inline = getattr(part, "inline_data", None)
                        if inline and inline.data:
                            return inline.data, inline.mime_type or "image/png"
                raise ImageGenError("no image in response (refused or filtered)")
            except ImageGenError as exc:
                raise exc
            except genai_errors.APIError as exc:
                last = exc
                if use_ratio and getattr(exc, "code", None) == 400:
                    continue  # retry once without image_config
                raise ImageGenError(f"image API error {getattr(exc, 'code', '?')}") from exc
            except Exception as exc:
                last = exc
                raise ImageGenError(f"image call failed: {type(exc).__name__}") from exc
        raise ImageGenError(f"image generation failed: {type(last).__name__}")
