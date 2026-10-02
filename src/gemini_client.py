# src/gemini_client.py
"""
Wrapper around Google GenAI SDK (`google-genai`).
Encapsulates initialization, retry logic, timeout handling, fallback models, and type-safe JSON extraction.
"""

from __future__ import annotations

import io
import time
from typing import Any, TypeVar

from PIL import Image
from pydantic import BaseModel

from src.config import Settings
from src.logger import logger

T = TypeVar("T", bound=BaseModel)


class GeminiError(Exception):
    """Base exception for Gemini client failures."""


class GeminiClient:

    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._init_client()

    def _init_client(self) -> None:
        if not self.cfg.gemini_api_key:
            return
        try:
            from google import genai

            self._client = genai.Client(api_key=self.cfg.gemini_api_key)
        except Exception as e:
            logger.error("GEMINI", f"Failed to initialize google-genai client: {e}")
            self._client = None

    def is_configured(self) -> bool:
        return self._client is not None

    def _retryable(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        code = getattr(exc, "code", None)
        status = getattr(exc, "status_code", None)
        if code in (429, 500, 502, 503, 504) or status in (429, 500, 502, 503, 504):
            return True
        for keyword in ("429", "500", "502", "503", "504", "quota", "resource_exhausted", "unavailable", "overloaded", "rate limit"):
            if keyword in msg:
                return True
        return False

    def _generate_json_model(
        self,
        model: str,
        prompt: str,
        schema: type[T],
        *,
        images: list[tuple[bytes, str]] | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        tag: str = "GEMINI",
    ) -> T:
        if not self._client:
            raise GeminiError("Gemini client is not configured (missing GEMINI_API_KEY)")

        from google.genai import types

        contents: list[Any] = []

        if images:
            for img_bytes, mime_type in images:
                try:
                    part = types.Part.from_bytes(data=img_bytes, mime_type=mime_type)
                    contents.append(part)
                except Exception as e:
                    logger.warn(tag, f"Failed to convert image bytes for Gemini prompt: {e}")

        contents.append(prompt)

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }
        if system:
            config_kwargs["system_instruction"] = system

        config = types.GenerateContentConfig(**config_kwargs)

        backoffs = [2.0, 5.0, 10.0, 20.0]
        max_attempts = len(backoffs) + 1
        last_exc: Exception | None = None

        for attempt in range(1, max_attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                parsed = getattr(response, "parsed", None)
                if parsed is not None and isinstance(parsed, schema):
                    return parsed

                text = getattr(response, "text", "") or ""
                if not text:
                    raise GeminiError("Empty response text from Gemini API")

                return schema.model_validate_json(text)

            except Exception as e:
                last_exc = e
                if not self._retryable(e) or attempt == max_attempts:
                    logger.error(tag, f"API error on {model} (attempt {attempt}/{max_attempts}): {e}")
                    raise GeminiError(f"API error on {model}: {e}") from e

                delay = backoffs[attempt - 1]
                logger.warn(
                    tag,
                    f"API error {e} on {model} (attempt {attempt}/{max_attempts}), retrying in {delay}s...",
                )
                time.sleep(delay)

        raise GeminiError(f"Failed to generate valid JSON from model {model}") from last_exc

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
        """
        Generates structured JSON output validated against a Pydantic schema.
        Falls back to fallback model if primary model is overloaded/unavailable (503/429/500).
        """
        models = [self.cfg.gemini_model]
        fb = (self.cfg.gemini_fallback_model or "").strip()
        if fb and fb not in models:
            models.append(fb)

        last_err: GeminiError | None = None

        for i, m in enumerate(models):
            try:
                return self._generate_json_model(
                    model=m,
                    prompt=prompt,
                    schema=schema,
                    images=images,
                    system=system,
                    temperature=temperature,
                    tag=tag,
                )
            except GeminiError as exc:
                last_err = exc
                cause = exc.__cause__ or exc
                if i < len(models) - 1 and self._retryable(cause):
                    logger.warn(
                        tag,
                        f"Model '{m}' unavailable after retries ({exc}). Falling back to '{models[i + 1]}'",
                    )
                    continue
                raise

        raise last_err or GeminiError("Gemini JSON generation failed across all models")

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        tag: str = "GEMINI_IMAGE",
    ) -> bytes:
        """Generates an image bytes payload using Imagen or Gemini image generation model."""
        if not self._client:
            raise GeminiError("Gemini client is not configured")

        from google.genai import types

        api_key_override = self.cfg.image_api_key or self.cfg.gemini_api_key
        client_to_use = self._client
        if api_key_override and api_key_override != self.cfg.gemini_api_key:
            try:
                from google import genai

                client_to_use = genai.Client(api_key=api_key_override)
            except Exception as e:
                logger.warn(tag, f"Failed to initialize image client override, falling back: {e}")

        models_to_try = [self.cfg.gemini_image_model, "imagen-3.0-generate-002"]

        backoffs = [3.0, 7.0, 15.0]
        max_attempts = len(backoffs) + 1

        for model_name in models_to_try:
            for attempt in range(1, max_attempts + 1):
                try:
                    img_config = None
                    try:
                        img_config = types.ImageConfig(aspect_ratio=aspect_ratio)
                    except Exception:
                        pass

                    config_args: dict[str, Any] = {
                        "number_of_images": 1,
                        "output_mime_type": "image/jpeg",
                    }
                    if img_config is not None:
                        config_args["image_config"] = img_config

                    config = types.GenerateImagesConfig(**config_args)

                    response = client_to_use.models.generate_images(
                        model=model_name,
                        prompt=prompt,
                        config=config,
                    )

                    generated_images = getattr(response, "generated_images", None)
                    if generated_images and len(generated_images) > 0:
                        first = generated_images[0]
                        image_obj = getattr(first, "image", None)
                        if image_obj:
                            image_bytes = getattr(image_obj, "image_bytes", None)
                            if image_bytes:
                                return bytes(image_bytes)

                    raise GeminiError(f"No image bytes returned in response from {model_name}")

                except Exception as e:
                    if not self._retryable(e) or attempt == max_attempts:
                        logger.warn(
                            tag,
                            f"Image model {model_name} attempt {attempt} failed: {e}",
                        )
                        break

                    delay = backoffs[attempt - 1]
                    logger.warn(
                        tag,
                        f"Image error on {model_name} (attempt {attempt}), retrying in {delay}s...",
                    )
                    time.sleep(delay)

        raise GeminiError("All image generation attempts and models failed")
