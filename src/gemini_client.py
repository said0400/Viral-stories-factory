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
    def _sleep(self, attempt: int, *, minimum: float = 2.0) -> None:
        """
        Exponential backoff.

        attempt=0 -> max(minimum, 2) seconds
        attempt=1 -> max(minimum, 4) seconds
        ...
        capped at 60 seconds.

        For rate limits (429), callers may pass minimum=15.
        """
        delay = min(
            60.0,
            max(
                float(minimum),
                2.0 * (2 ** attempt),
            ),
        )
        time.sleep(delay)

    @staticmethod
    def _retryable(exc: Exception) -> bool:
        """Return True only for transient Gemini/API/network failures."""
        if isinstance(exc, genai_errors.ServerError):
            return True

        code = getattr(exc, "code", None)
        if code in (408, 429, 500, 502, 503, 504):
            return True

        # Network-ish failures sometimes wrap as generic exceptions.
        name = type(exc).__name__
        if name in {
            "TimeoutError",
            "ConnectError",
            "ReadTimeout",
            "ConnectTimeout",
            "ConnectionError",
            "RemoteProtocolError",
        }:
            return True

        return False

    @staticmethod
    def _is_rate_limited(exc: Exception) -> bool:
        return getattr(exc, "code", None) == 429

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

        Supports candidates/content.parts and response.parts.
        MIME validation is intentionally left to image_generator.
        """
        for cand in getattr(resp, "candidates", None) or []:
            content = getattr(cand, "content", None)
            parts = (
                getattr(content, "parts", None)
                if content
                else None
            ) or []

            for part in parts:
                inline = getattr(part, "inline_data", None)
                if inline is None:
                    continue

                data = getattr(inline, "data", None)
                if data:
                    mime = (
                        getattr(inline, "mime_type", None)
                        or "image/png"
                    )
                    return data, mime

        for part in getattr(resp, "parts", None) or []:
            inline = getattr(part, "inline_data", None)
            if inline is None:
                continue

            data = getattr(inline, "data", None)
            if data:
                mime = (
                    getattr(inline, "mime_type", None)
                    or "image/png"
                )
                return data, mime

        return None

    def _image_config(
        self,
        aspect_ratio: str | None,
        *,
        use_aspect: bool,
    ) -> types.GenerateContentConfig:
        """
        Build image generation config.

        Both shapes are kept because different google-genai / model
        combinations accept different knobs:

        1) image_config=types.ImageConfig(aspect_ratio=...)
        2) response_format={"image": {"aspect_ratio": ...}}

        If aspect ratio cannot be applied, fall back to IMAGE-only config.
        """
        timeout_ms = self.cfg.image_request_timeout * 1000

        if use_aspect and aspect_ratio:
            try:
                image_cfg = types.ImageConfig(
                    aspect_ratio=aspect_ratio,
                )
                return types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    image_config=image_cfg,
                    http_options=types.HttpOptions(
                        timeout=timeout_ms,
                    ),
                )
            except (TypeError, ValueError, AttributeError):
                pass

            try:
                return types.GenerateContentConfig(
                    response_modalities=["IMAGE"],
                    response_format={
                        "image": {
                            "aspect_ratio": aspect_ratio,
                        }
                    },
                    http_options=types.HttpOptions(
                        timeout=timeout_ms,
                    ),
                )
            except (TypeError, ValueError, AttributeError):
                pass

        return types.GenerateContentConfig(
            response_modalities=["IMAGE"],
            http_options=types.HttpOptions(
                timeout=timeout_ms,
            ),
        )

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

        Retry policy:
        - ValidationError / invalid JSON / empty body: retry with stricter
          instruction and lower temperature (may recover).
        - Empty response is retried at most ONCE (often safety/token; hammering
          rarely helps).
        - Transient API/network errors: retry with backoff (429 minimum 15s).
        - Non-retryable API errors and unexpected programming errors: fail fast.
        """
        parts: list = [
            types.Part.from_bytes(
                data=b,
                mime_type=m,
            )
            for b, m in (images or [])
        ]

        last: Exception | None = None
        timeout_ms = self.cfg.llm_timeout * 1000
        empty_retries_used = 0

        for attempt in range(self.cfg.max_retries + 1):
            text_prompt = prompt + (
                STRICT_SUFFIX if attempt >= 1 else ""
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
                    timeout=timeout_ms,
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
                    getattr(resp, "text", "")
                )

                if not raw:
                    raise GeminiError(
                        "empty response "
                        "(possible safety block or token limit)"
                    )

                return schema.model_validate_json(raw)

            except ValidationError as exc:
                last = exc
                logger.warn(
                    tag,
                    "invalid structured output "
                    f"(attempt {attempt + 1}): "
                    f"{type(exc).__name__}",
                )
                # Recoverable: stricter JSON instruction on next loop.

            except GeminiError as exc:
                last = exc
                msg = str(exc).lower()
                is_empty = "empty response" in msg

                logger.warn(
                    tag,
                    "invalid JSON/response "
                    f"(attempt {attempt + 1}): "
                    f"{type(exc).__name__}",
                )

                if is_empty:
                    empty_retries_used += 1
                    # Safety/empty rarely heals by repeating the same call.
                    if empty_retries_used > 1:
                        raise GeminiError(
                            "empty/safety response persisted after retry"
                        ) from exc

            except genai_errors.APIError as exc:
                last = exc
                code = getattr(exc, "code", "?")

                if not self._retryable(exc):
                    raise GeminiError(
                        f"non-retryable API error {code}"
                    ) from exc

                logger.warn(
                    tag,
                    f"API error {code} "
                    f"(attempt {attempt + 1})",
                )

                if attempt >= self.cfg.max_retries:
                    raise GeminiError(
                        f"API error {code} after retries"
                    ) from exc

                self._sleep(
                    attempt,
                    minimum=15.0 if self._is_rate_limited(exc) else 2.0,
                )
                continue

            except Exception as exc:
                # Do NOT blindly retry programming / SDK shape errors.
                last = exc

                if self._retryable(exc):
                    logger.warn(
                        tag,
                        f"transient {type(exc).__name__} "
                        f"(attempt {attempt + 1})",
                    )
                    if attempt >= self.cfg.max_retries:
                        break
                    self._sleep(attempt, minimum=2.0)
                    continue

                logger.error(
                    tag,
                    f"non-retryable {type(exc).__name__}: failing fast",
                )
                raise GeminiError(
                    f"Gemini JSON call failed: {type(exc).__name__}"
                ) from exc

            if attempt < self.cfg.max_retries:
                self._sleep(attempt, minimum=2.0)
                continue

            break

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

        Retry budget (cost control):
        - At most `max_total_calls` API calls across BOTH aspect-ratio modes.
        - Empty/filtered IMAGE responses: at most 2 empty results total, then stop
          (image_generator may change prompt/strategy afterward).
        - HTTP 400 on aspect config: drop aspect and continue within budget.
        - Transient API errors: backoff; 429 uses minimum 15s.
        - Unexpected non-retryable exceptions: fail fast.
        """
        contents: list = [prompt]

        for b, m in references or []:
            contents.append(
                types.Part.from_bytes(
                    data=b,
                    mime_type=m,
                )
            )

        last: Exception | None = None

        # Hard cap across aspect-on + aspect-off paths.
        # Example: max_retries=3 -> max_total_calls=4 (not 3*2=6).
        max_total_calls = max(
            1,
            min(self.cfg.max_retries + 1, 4),
        )
        calls_used = 0
        empty_image_count = 0

        for use_ratio in (True, False):
            if calls_used >= max_total_calls:
                break

            while calls_used < max_total_calls:
                attempt = calls_used
                calls_used += 1

                try:
                    config = self._image_config(
                        aspect_ratio,
                        use_aspect=use_ratio,
                    )

                    resp = self.image_client.models.generate_content(
                        model=self.cfg.gemini_image_model,
                        contents=contents,
                        config=config,
                    )

                    image = self._extract_image_from_response(resp)
                    if image:
                        return image

                    empty_image_count += 1
                    last = ImageGenError(
                        "no image in response "
                        "(refused, filtered, or empty response)"
                    )

                    logger.warn(
                        "IMAGE",
                        "image response contained no usable image "
                        f"(call {calls_used}/{max_total_calls}, "
                        f"aspect={'on' if use_ratio else 'off'})",
                    )

                    # Same prompt + empty IMAGE usually will not heal.
                    if empty_image_count >= 2:
                        raise ImageGenError(
                            "no usable image after repeated empty/filtered responses"
                        )

                    self._sleep(attempt, minimum=2.0)
                    continue

                except ImageGenError:
                    raise

                except genai_errors.APIError as exc:
                    last = exc
                    code = getattr(exc, "code", None)

                    if use_ratio and code == 400:
                        logger.warn(
                            "IMAGE",
                            "model rejected image aspect/config; "
                            "switching to no-aspect path",
                        )
                        break  # next use_ratio=False

                    if not self._retryable(exc):
                        raise ImageGenError(
                            f"image API error {getattr(exc, 'code', '?')}"
                        ) from exc

                    logger.warn(
                        "IMAGE",
                        f"image API error {getattr(exc, 'code', '?')} "
                        f"(call {calls_used}/{max_total_calls})",
                    )

                    if calls_used >= max_total_calls:
                        raise ImageGenError(
                            f"image API error {getattr(exc, 'code', '?')}"
                        ) from exc

                    self._sleep(
                        attempt,
                        minimum=15.0 if self._is_rate_limited(exc) else 2.0,
                    )
                    continue

                except Exception as exc:
                    last = exc

                    # Aspect/config shape issues -> try without aspect.
                    if use_ratio and isinstance(
                        exc,
                        (TypeError, ValueError, AttributeError),
                    ):
                        logger.warn(
                            "IMAGE",
                            "image config was not accepted; "
                            "switching to no-aspect path",
                        )
                        break

                    if self._retryable(exc):
                        logger.warn(
                            "IMAGE",
                            f"transient {type(exc).__name__} "
                            f"(call {calls_used}/{max_total_calls})",
                        )
                        if calls_used >= max_total_calls:
                            break
                        self._sleep(attempt, minimum=2.0)
                        continue

                    raise ImageGenError(
                        f"image call failed: {type(exc).__name__}"
                    ) from exc

        raise ImageGenError(
            "image generation failed: "
            f"{type(last).__name__ if last else 'unknown'}"
        )
