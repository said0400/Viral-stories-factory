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


STRICT_SUFFIX = (
    "\n\nIMPORTANT: Reply with ONE valid JSON object that exactly matches the schema. "
    "No markdown fences, no commentary, no trailing text."
)


class GeminiError(Exception):
    pass


class ImageGenError(GeminiError):
    pass


class GeminiClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg

        if not cfg.gemini_api_key:
            raise GeminiError(
                "GEMINI_API_KEY is missing"
            )

        self.client = genai.Client(
            api_key=cfg.gemini_api_key
        )

        if cfg.image_api_key:
            self.image_client = (
                self.client
                if cfg.image_api_key == cfg.gemini_api_key
                else genai.Client(
                    api_key=cfg.image_api_key
                )
            )
        else:
            self.image_client = self.client

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------
    def _sleep(self, attempt: int) -> None:
        """
        Exponential backoff.

        attempt=0 -> 2 seconds
        attempt=1 -> 4 seconds
        attempt=2 -> 8 seconds
        ...
        capped at 60 seconds.
        """
        time.sleep(
            min(
                60,
                2 * (2 ** attempt),
            )
        )

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """Return True for transient Gemini/API/network failures."""
        code = getattr(
            exc,
            "code",
            None,
        )

        return (
            isinstance(
                exc,
                genai_errors.ServerError,
            )
            or code in (
                408,
                429,
                500,
                502,
                503,
                504,
            )
        )

    @staticmethod
    def _clean_json_response(raw: str) -> str:
        """Remove accidental markdown fences without changing the JSON itself."""
        raw = (raw or "").strip()

        if not raw:
            return ""

        if raw.startswith("```"):
            lines = raw.splitlines()

            if lines:
                lines = lines[1:]

            if lines and lines[-1].strip() == "```":
                lines = lines[:-1]

            raw = "\n".join(lines).strip()

            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()

        return raw

    @staticmethod
    def _extract_image_from_response(resp) -> tuple[bytes, str] | None:
        """
        Extract the first inline image from a GenerateContent response.

        Supports the normal candidates/content.parts structure and also
        response.parts when exposed by the installed google-genai version.
        """

        # Preferred/current GenerateContent response path.
        for cand in getattr(
            resp,
            "candidates",
            None,
        ) or []:
            content = getattr(
                cand,
                "content",
                None,
            )

            parts = (
                getattr(
                    content,
                    "parts",
                    None,
                )
                if content
                else None
            ) or []

            for part in parts:
                inline = getattr(
                    part,
                    "inline_data",
                    None,
                )

                if inline is None:
                    continue

                data = getattr(
                    inline,
                    "data",
                    None,
                )

                if data:
                    mime = (
                        getattr(
                            inline,
                            "mime_type",
                            None,
                        )
                        or "image/png"
                    )

                    return data, mime

        # Some google-genai response versions expose response.parts directly.
        for part in getattr(
            resp,
            "parts",
            None,
        ) or []:
            inline = getattr(
                part,
                "inline_data",
                None,
            )

            if inline is None:
                continue

            data = getattr(
                inline,
                "data",
                None,
            )

            if data:
                mime = (
                    getattr(
                        inline,
                        "mime_type",
                        None,
                    )
                    or "image/png"
                )

                return data, mime

        return None

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
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
        Structured output with validation.

        Flow:
        1. Call Gemini.
        2. Parse JSON.
        3. Validate against the Pydantic schema.
        4. Retry transient API failures.
        5. Retry invalid output with a stricter JSON instruction.
        6. Fail only after all retries are exhausted.

        Google supports structured output with Pydantic JSON schemas. 
        """

        parts: list = [
            types.Part.from_bytes(
                data=b,
                mime_type=m,
            )
            for b, m in (images or [])
        ]

        last: Exception | None = None

        for attempt in range(
            self.cfg.max_retries + 1
        ):
            text_prompt = (
                prompt
                + (
                    STRICT_SUFFIX
                    if attempt >= 1
                    else ""
                )
            )

            config = types.GenerateContentConfig(
                system_instruction=system,
                response_mime_type="application/json",
                response_json_schema=schema.model_json_schema(),
                temperature=max(
                    0.0,
                    temperature - (0.2 * attempt),
                ),
                http_options=types.HttpOptions(
                    timeout=self.cfg.request_timeout * 1000
                ),
            )

            try:
                resp = self.client.models.generate_content(
                    model=self.cfg.gemini_model,
                    contents=[
                        *parts,
                        text_prompt,
                    ],
                    config=config,
                )

                raw = self._clean_json_response(
                    getattr(
                        resp,
                        "text",
                        "",
                    )
                )

                if not raw:
                    raise GeminiError(
                        "empty response "
                        "(possible safety block or token limit)"
                    )

                try:
                    return schema.model_validate_json(
                        raw
                    )
                except ValidationError as exc:
                    raise exc

            except ValidationError as exc:
                last = exc

                logger.warn(
                    tag,
                    "invalid structured output "
                    f"(attempt {attempt + 1}): "
                    f"{type(exc).__name__}",
                )

            except (GeminiError, ValueError) as exc:
                last = exc

                logger.warn(
                    tag,
                    "invalid JSON/response "
                    f"(attempt {attempt + 1}): "
                    f"{type(exc).__name__}",
                )

            except genai_errors.APIError as exc:
                last = exc

                code = getattr(
                    exc,
                    "code",
                    "?",
                )

                if not self._retryable(exc):
                    raise GeminiError(
                        f"non-retryable API error {code}"
                    ) from exc

                logger.warn(
                    tag,
                    f"API error {code} "
                    f"(attempt {attempt + 1})",
                )

            except Exception as exc:
                last = exc

                logger.warn(
                    tag,
                    f"{type(exc).__name__} "
                    f"(attempt {attempt + 1})",
                )

            if attempt < self.cfg.max_retries:
                self._sleep(
                    attempt
                )

        raise GeminiError(
            "Gemini JSON call failed after retries: "
            f"{type(last).__name__ if last else 'unknown'}"
        )

    # ------------------------------------------------------------------
    # images
    # ------------------------------------------------------------------
    def generate_image(
        self,
        prompt: str,
        *,
        references: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "16:9",
    ) -> tuple[bytes, str]:
        """
        Generate a new image and optionally use supplied reference images.

        Returns:
            (image_bytes, mime_type)

        The source/reference image is supplied as multimodal input.
        It is never returned directly as the generated image.

        Gemini 3.1 Flash Image supports image references and configurable
        aspect ratios through the image response format.
        """

        contents: list = [
            prompt
        ]

        for b, m in (
            references or []
        ):
            contents.append(
                types.Part.from_bytes(
                    data=b,
                    mime_type=m,
                )
            )

        last: Exception | None = None

        # First try with the requested aspect ratio.
        # If a model/version rejects response_format, retry once without it.
        use_ratio_options = (
            True,
            False,
        )

        for use_ratio in use_ratio_options:
            for attempt in range(
                self.cfg.max_retries + 1
            ):
                try:
                    if use_ratio:
                        image_format = {
                            "image": {
                                "aspect_ratio": aspect_ratio,
                            }
                        }

                        config = types.GenerateContentConfig(
                            response_modalities=[
                                "IMAGE"
                            ],
                            response_format=image_format,
                            http_options=types.HttpOptions(
                                timeout=(
                                    self.cfg.image_request_timeout
                                    * 1000
                                )
                            ),
                        )
                    else:
                        config = types.GenerateContentConfig(
                            response_modalities=[
                                "IMAGE"
                            ],
                            http_options=types.HttpOptions(
                                timeout=(
                                    self.cfg.image_request_timeout
                                    * 1000
                                )
                            ),
                        )

                    resp = self.image_client.models.generate_content(
                        model=self.cfg.gemini_image_model,
                        contents=contents,
                        config=config,
                    )

                    image = self._extract_image_from_response(
                        resp
                    )

                    if image:
                        return image

                    raise ImageGenError(
                        "no image in response "
                        "(refused, filtered, or empty response)"
                    )

                except ImageGenError as exc:
                    last = exc

                    # A refusal/filtering response is generally not fixed
                    # by blindly repeating the same request many times.
                    # However, retrying can still help with transient
                    # response failures, so follow the configured retry count.
                    if attempt < self.cfg.max_retries:
                        logger.warn(
                            "IMAGE",
                            "image response contained no usable image "
                            f"(attempt {attempt + 1})",
                        )
                        self._sleep(
                            attempt
                        )
                        continue

                    # If the aspect-ratio configuration itself might be the
                    # problem, move to the no-ratio fallback.
                    if use_ratio:
                        logger.warn(
                            "IMAGE",
                            "image generation with aspect ratio failed; "
                            "retrying without response_format",
                        )
                        break

                    raise ImageGenError(
                        "image generation failed: "
                        f"{type(exc).__name__}"
                    ) from exc

                except genai_errors.APIError as exc:
                    last = exc

                    code = getattr(
                        exc,
                        "code",
                        None,
                    )

                    if (
                        use_ratio
                        and code == 400
                    ):
                        logger.warn(
                            "IMAGE",
                            "model rejected image response_format; "
                            "retrying without aspect ratio",
                        )
                        break

                    if not self._retryable(exc):
                        raise ImageGenError(
                            f"image API error "
                            f"{getattr(exc, 'code', '?')}"
                        ) from exc

                    logger.warn(
                        "IMAGE",
                        f"image API error "
                        f"{getattr(exc, 'code', '?')} "
                        f"(attempt {attempt + 1})",
                    )

                    if attempt < self.cfg.max_retries:
                        self._sleep(
                            attempt
                        )
                        continue

                    raise ImageGenError(
                        f"image API error "
                        f"{getattr(exc, 'code', '?')}"
                    ) from exc

                except Exception as exc:
                    last = exc

                    logger.warn(
                        "IMAGE",
                        f"{type(exc).__name__} "
                        f"(attempt {attempt + 1})",
                    )

                    if (
                        self._retryable(exc)
                        and attempt < self.cfg.max_retries
                    ):
                        self._sleep(
                            attempt
                        )
                        continue

                    # If response_format is rejected by the installed SDK
                    # or model, allow the no-ratio fallback.
                    if use_ratio and isinstance(
                        exc,
                        (TypeError, ValueError),
                    ):
                        logger.warn(
                            "IMAGE",
                            "image response_format was not accepted; "
                            "retrying without aspect ratio",
                        )
                        break

                    raise ImageGenError(
                        f"image call failed: "
                        f"{type(exc).__name__}"
                    ) from exc

        raise ImageGenError(
            "image generation failed: "
            f"{type(last).__name__ if last else 'unknown'}"
        )
