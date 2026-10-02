# src/gemini_client.py
"""
Wrapper around Google GenAI SDK (`google-genai`).
Encapsulates initialization, retry logic, timeout handling, fallback models,
and type-safe JSON/image extraction.
"""

from __future__ import annotations

import time
from typing import Any, TypeVar

from pydantic import BaseModel

from .config import Settings
from . import logger

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

            self._client = genai.Client(
                api_key=self.cfg.gemini_api_key,
            )

        except Exception as exc:
            logger.error(
                "GEMINI",
                f"Failed to initialize google-genai client: {exc}",
            )
            self._client = None

    def is_configured(self) -> bool:
        return self._client is not None

    def _retryable(self, exc: Exception) -> bool:
        """
        Return True only for errors that are reasonably safe to retry.

        Primary signals are explicit HTTP/status codes. Message matching is
        retained as a fallback because google-genai exceptions can expose
        status information differently across SDK versions.
        """
        code = getattr(exc, "code", None)
        status = getattr(exc, "status_code", None)

        try:
            code_int = int(code) if code is not None else None
        except (TypeError, ValueError):
            code_int = None

        try:
            status_int = int(status) if status is not None else None
        except (TypeError, ValueError):
            status_int = None

        retryable_codes = {
            408,
            429,
            500,
            502,
            503,
            504,
        }

        if code_int in retryable_codes:
            return True

        if status_int in retryable_codes:
            return True

        msg = str(exc).lower()

        retryable_keywords = (
            "429",
            "500",
            "502",
            "503",
            "504",
            "resource_exhausted",
            "rate limit",
            "rate_limit",
            "too many requests",
            "temporarily unavailable",
            "service unavailable",
            "unavailable",
            "overloaded",
            "deadline exceeded",
            "timeout",
            "timed out",
        )

        return any(
            keyword in msg
            for keyword in retryable_keywords
        )

    def _retry_delays(self) -> list[float]:
        """
        Build the retry schedule from Settings.max_retries.

        max_retries=3 means at most 4 total attempts.
        """
        max_retries = max(
            0,
            int(self.cfg.max_retries),
        )

        base_delays = [
            2.0,
            5.0,
            10.0,
            20.0,
        ]

        if max_retries <= 0:
            return []

        return base_delays[:max_retries]

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
            raise GeminiError(
                "Gemini client is not configured "
                "(missing GEMINI_API_KEY)"
            )

        from google.genai import types

        contents: list[Any] = []

        if images:
            for img_bytes, mime_type in images:
                try:
                    part = types.Part.from_bytes(
                        data=img_bytes,
                        mime_type=mime_type,
                    )
                    contents.append(part)

                except Exception as exc:
                    raise GeminiError(
                        f"Failed to convert image bytes for Gemini prompt: {exc}"
                    ) from exc

        contents.append(prompt)

        config_kwargs: dict[str, Any] = {
            "response_mime_type": "application/json",
            "response_schema": schema,
            "temperature": temperature,
        }

        if system:
            config_kwargs["system_instruction"] = system

        try:
            config = types.GenerateContentConfig(
                **config_kwargs,
            )
        except Exception as exc:
            raise GeminiError(
                f"Failed to build Gemini generation config: {exc}"
            ) from exc

        backoffs = self._retry_delays()
        max_attempts = len(backoffs) + 1

        last_exc: Exception | None = None

        for attempt in range(
            1,
            max_attempts + 1,
        ):
            try:
                response = self._client.models.generate_content(
                    model=model,
                    contents=contents,
                    config=config,
                )

                parsed = getattr(
                    response,
                    "parsed",
                    None,
                )

                if parsed is not None:
                    if isinstance(parsed, schema):
                        return parsed

                    try:
                        return schema.model_validate(parsed)
                    except Exception as exc:
                        raise GeminiError(
                            "Gemini returned a structured response, "
                            "but it could not be validated against the "
                            f"{schema.__name__} schema: {exc}"
                        ) from exc

                text = getattr(
                    response,
                    "text",
                    "",
                ) or ""

                text = text.strip()

                if not text:
                    raise GeminiError(
                        f"Empty response from Gemini model {model}"
                    )

                try:
                    return schema.model_validate_json(text)
                except Exception as exc:
                    raise GeminiError(
                        f"Gemini returned invalid JSON for "
                        f"{schema.__name__}: {exc}"
                    ) from exc

            except GeminiError as exc:
                last_exc = exc

                # Validation / empty-response errors are not treated as
                # transient API failures.
                if attempt == max_attempts:
                    logger.error(
                        tag,
                        f"Gemini generation failed on {model} "
                        f"(attempt {attempt}/{max_attempts}): {exc}",
                    )
                    raise

                # Only retry if the underlying error is actually transient.
                if not self._retryable(exc):
                    logger.error(
                        tag,
                        f"Gemini generation failed on {model} "
                        f"(attempt {attempt}/{max_attempts}): {exc}",
                    )
                    raise

                delay = backoffs[attempt - 1]

                logger.warn(
                    tag,
                    f"Transient Gemini error on {model} "
                    f"(attempt {attempt}/{max_attempts}), "
                    f"retrying in {delay}s: {exc}",
                )

                time.sleep(delay)

            except Exception as exc:
                last_exc = exc

                if not self._retryable(exc):
                    logger.error(
                        tag,
                        f"API error on {model} "
                        f"(attempt {attempt}/{max_attempts}): {exc}",
                    )
                    raise GeminiError(
                        f"API error on {model}: {exc}"
                    ) from exc

                if attempt == max_attempts:
                    logger.error(
                        tag,
                        f"API error on {model} "
                        f"(attempt {attempt}/{max_attempts}): {exc}",
                    )
                    raise GeminiError(
                        f"API error on {model} after retries: {exc}"
                    ) from exc

                delay = backoffs[attempt - 1]

                logger.warn(
                    tag,
                    f"Transient API error on {model} "
                    f"(attempt {attempt}/{max_attempts}), "
                    f"retrying in {delay}s: {exc}",
                )

                time.sleep(delay)

        raise GeminiError(
            f"Failed to generate valid JSON from model {model}"
        ) from last_exc

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
        Generate structured JSON output validated against a Pydantic schema.

        The primary Gemini model is attempted first. If it fails, the
        configured fallback model is attempted.
        """
        models = [
            self.cfg.gemini_model,
        ]

        fallback = (
            self.cfg.gemini_fallback_model or ""
        ).strip()

        if fallback and fallback not in models:
            models.append(fallback)

        last_err: GeminiError | None = None

        for index, model_name in enumerate(models):
            try:
                return self._generate_json_model(
                    model=model_name,
                    prompt=prompt,
                    schema=schema,
                    images=images,
                    system=system,
                    temperature=temperature,
                    tag=tag,
                )

            except GeminiError as exc:
                last_err = exc

                if index < len(models) - 1:
                    next_model = models[index + 1]

                    logger.warn(
                        tag,
                        f"Model '{model_name}' failed "
                        f"({exc}). Falling back to '{next_model}'",
                    )

                    continue

                raise

        raise last_err or GeminiError(
            "Gemini JSON generation failed across all models"
        )

    def _generate_gemini_image(
        self,
        client: Any,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        tag: str,
    ) -> bytes:
        """
        Generate an image through Gemini's content-generation API.

        Gemini image-capable models use generate_content() with IMAGE
        response modality rather than the Imagen generate_images() endpoint.
        """
        from google.genai import types

        config_kwargs: dict[str, Any] = {
            "response_modalities": [
                "IMAGE",
            ],
        }

        try:
            config_kwargs["image_config"] = types.ImageConfig(
                aspect_ratio=aspect_ratio,
            )
        except Exception:
            pass

        config = types.GenerateContentConfig(
            **config_kwargs,
        )

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )

        parts = getattr(
            response,
            "parts",
            None,
        ) or []

        for part in parts:
            inline_data = getattr(
                part,
                "inline_data",
                None,
            )

            if inline_data is not None:
                image_bytes = getattr(
                    inline_data,
                    "data",
                    None,
                )

                if image_bytes:
                    return bytes(image_bytes)

                try:
                    image = part.as_image()

                    image_bytes = getattr(
                        image,
                        "image_bytes",
                        None,
                    )

                    if image_bytes:
                        return bytes(image_bytes)

                except Exception:
                    pass

        raise GeminiError(
            f"No image bytes returned in response from {model_name}"
        )

    def _generate_imagen_image(
        self,
        client: Any,
        model_name: str,
        prompt: str,
        aspect_ratio: str,
        tag: str,
    ) -> bytes:
        """
        Generate an image through the Imagen generate_images() API.
        """
        from google.genai import types

        config_args: dict[str, Any] = {
            "number_of_images": 1,
            "output_mime_type": "image/jpeg",
        }

        try:
            config_args["aspect_ratio"] = aspect_ratio
        except Exception:
            pass

        try:
            config = types.GenerateImagesConfig(
                **config_args,
            )
        except Exception:
            config_args.pop(
                "aspect_ratio",
                None,
            )

            config = types.GenerateImagesConfig(
                **config_args,
            )

        response = client.models.generate_images(
            model=model_name,
            prompt=prompt,
            config=config,
        )

        generated_images = getattr(
            response,
            "generated_images",
            None,
        )

        if generated_images:
            first = generated_images[0]

            image_obj = getattr(
                first,
                "image",
                None,
            )

            if image_obj:
                image_bytes = getattr(
                    image_obj,
                    "image_bytes",
                    None,
                )

                if image_bytes:
                    return bytes(image_bytes)

        raise GeminiError(
            f"No image bytes returned in response from {model_name}"
        )

    def generate_image(
        self,
        prompt: str,
        aspect_ratio: str = "1:1",
        tag: str = "GEMINI_IMAGE",
    ) -> bytes:
        """
        Generate an image using the configured Gemini image model.

        Gemini image models use generate_content(). If that fails, an
        Imagen model is attempted through generate_images().
        """
        if not self._client:
            raise GeminiError(
                "Gemini client is not configured"
            )

        api_key_override = (
            self.cfg.image_api_key
            or self.cfg.gemini_api_key
        )

        client_to_use = self._client

        if (
            api_key_override
            and api_key_override != self.cfg.gemini_api_key
        ):
            try:
                from google import genai

                client_to_use = genai.Client(
                    api_key=api_key_override,
                )

            except Exception as exc:
                logger.warn(
                    tag,
                    "Failed to initialize image client override, "
                    f"falling back to primary client: {exc}",
                )

        primary_image_model = (
            self.cfg.gemini_image_model or ""
        ).strip()

        models_to_try: list[tuple[str, str]] = []

        if primary_image_model:
            if "imagen" in primary_image_model.lower():
                models_to_try.append(
                    (
                        primary_image_model,
                        "imagen",
                    )
                )
            else:
                models_to_try.append(
                    (
                        primary_image_model,
                        "gemini",
                    )
                )

        models_to_try.append(
            (
                "imagen-3.0-generate-002",
                "imagen",
            )
        )

        # Remove duplicate model/backend pairs while preserving order.
        seen_models: set[tuple[str, str]] = set()
        unique_models: list[tuple[str, str]] = []

        for item in models_to_try:
            if item in seen_models:
                continue

            seen_models.add(item)
            unique_models.append(item)

        models_to_try = unique_models

        backoffs = self._retry_delays()
        max_attempts = len(backoffs) + 1

        last_exc: Exception | None = None

        for model_name, backend in models_to_try:
            for attempt in range(
                1,
                max_attempts + 1,
            ):
                try:
                    if backend == "gemini":
                        return self._generate_gemini_image(
                            client=client_to_use,
                            model_name=model_name,
                            prompt=prompt,
                            aspect_ratio=aspect_ratio,
                            tag=tag,
                        )

                    return self._generate_imagen_image(
                        client=client_to_use,
                        model_name=model_name,
                        prompt=prompt,
                        aspect_ratio=aspect_ratio,
                        tag=tag,
                    )

                except Exception as exc:
                    last_exc = exc

                    if not self._retryable(exc):
                        logger.warn(
                            tag,
                            f"Image model {model_name} "
                            f"attempt {attempt}/{max_attempts} "
                            f"failed with non-retryable error: {exc}",
                        )
                        break

                    if attempt == max_attempts:
                        logger.warn(
                            tag,
                            f"Image model {model_name} "
                            f"attempt {attempt}/{max_attempts} "
                            f"failed after retries: {exc}",
                        )
                        break

                    delay = backoffs[attempt - 1]

                    logger.warn(
                        tag,
                        f"Transient image error on {model_name} "
                        f"(attempt {attempt}/{max_attempts}), "
                        f"retrying in {delay}s: {exc}",
                    )

                    time.sleep(delay)

        raise GeminiError(
            "All image generation attempts and models failed"
        ) from last_exc
