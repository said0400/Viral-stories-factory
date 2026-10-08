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

# Calls with these tags use the strong model chain (GEMINI_CONTENT_MODEL + fallbacks).
# Everything else (triage, visual analysis, image check) uses the cheaper GEMINI_MODEL.
_STRONG_TAGS = {"CONTENT", "ARTICLE", "METADATA", "FACTCHECK"}

# While another model is still available in the chain, give up on a struggling model after this many attempts.
_ATTEMPTS_BEFORE_FALLBACK = 2


class GeminiError(Exception):
    """Base exception for Gemini failures."""


class ImageGenError(GeminiError):
    """Image generation failed."""


class ImageQuotaError(ImageGenError):
    """The image provider has no quota left (retrying is pointless)."""


class GeminiKeyError(GeminiError):
    """The active Gemini API key is rate-limited, out of quota, or rejected."""


class GeminiClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self._client: Any = None
        self._image_client: Any = None
        self._image_client_api_key = ""
        self._api_keys: list[tuple[str, str]] = []
        self._exhausted_keys: set[int] = set()
        self._exhausted_image_keys: set[str] = set()
        self._active_key_index: int | None = None

        for label, value in (
            ("GEMINI_API_KEY_1", cfg.gemini_api_key),
            ("GEMINI_API_KEY_2", cfg.gemini_api_key_2),
            ("GEMINI_API_KEY_3", cfg.gemini_api_key_3),
        ):
            value = str(value or "").strip()
            if value and all(existing != value for _, existing in self._api_keys):
                self._api_keys.append((label, value))

        for index in range(len(self._api_keys)):
            try:
                self._activate_key(index)
                break
            except Exception as exc:
                self._exhausted_keys.add(index)
                logger.error("GEMINI", f"Failed to initialise {self._api_keys[index][0]} client ({type(exc).__name__})")

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
        return self._client is not None and bool(self._api_keys)

    def _activate_key(self, index: int) -> None:
        if not 0 <= index < len(self._api_keys):
            raise GeminiError("Gemini API key index is invalid")
        if self._active_key_index == index and self._client is not None:
            return

        _label, value = self._api_keys[index]
        self._client = self._make_client(value, self.cfg.llm_timeout)
        self._active_key_index = index
        # If IMAGE_API_KEY is absent, image calls use the active Gemini key too.
        self._image_client = None
        self._image_client_api_key = ""

    def _available_key_indices(self) -> list[int]:
        return [i for i in range(len(self._api_keys)) if i not in self._exhausted_keys]

    def _active_key_label(self) -> str:
        if self._active_key_index is None:
            return "Gemini key"
        return self._api_keys[self._active_key_index][0]

    @staticmethod
    def _is_quota_exhausted(exc: BaseException) -> bool:
        """Recognize quota/resource-exhaustion responses that should move to another key."""
        low = str(exc).lower()
        return any(word in low for word in ("limit: 0", "perday", "quota exceeded", "resource_exhausted"))

    @classmethod
    def _key_unavailable(cls, exc: BaseException) -> bool:
        if cls._is_quota_exhausted(exc):
            return True
        for attr in ("code", "status_code"):
            try:
                if int(getattr(exc, attr, 0)) in {401, 403, 429}:
                    return True
            except (TypeError, ValueError):
                pass
        low = str(exc).lower()
        return any(
            phrase in low
            for phrase in (
                "invalid api key",
                "api key not valid",
                "api_key_invalid",
                "rate limit",
                "too many requests",
                "resource exhausted",
                "quota exceeded",
            )
        )

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

    def _model_chain(self, tag: str) -> list[str]:
        if tag in _STRONG_TAGS:
            chain = list(self.cfg.content_models)
        else:
            chain = [self.cfg.gemini_model, self.cfg.gemini_fallback_model]

        out: list[str] = []
        retired = {"gemini-2.5-flash"}

        for name in chain:
            name = (name or "").strip()
            if name and name.lower() not in retired and name not in out:
                out.append(name)

        return out or ["gemini-3.5-flash"]

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
        max_attempts: int | None = None,
    ) -> T:
        if not self._client:
            raise GeminiError("Gemini client is not configured (missing Gemini API keys)")

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

        if max_attempts is not None:
            delays = delays[: max(0, int(max_attempts) - 1)]

        attempts = len(delays) + 1

        for attempt in range(1, attempts + 1):
            try:
                response = self._client.models.generate_content(
                    model=model, contents=contents, config=config
                )
            except Exception as exc:
                if self._key_unavailable(exc):
                    if self._active_key_index is not None:
                        self._exhausted_keys.add(self._active_key_index)
                    logger.warn(
                        tag,
                        f"{self._active_key_label()} is unavailable for this run; moving to the next configured key",
                    )
                    raise GeminiKeyError(
                        f"{model} rejected the active key due to quota, rate limit, or authentication"
                    ) from exc

                if attempt < attempts and self._retryable(exc):
                    delay = delays[attempt - 1]
                    logger.warn(
                        tag,
                        f"Transient API error on {model} (attempt {attempt}/{attempts}), "
                        f"retrying in {delay}s: {str(exc)[:200]}",
                    )
                    time.sleep(delay)
                    continue

                logger.error(tag, f"API error on {model} (attempt {attempt}/{attempts}): {str(exc)[:300]}")
                raise GeminiError(f"API error on {model}: {str(exc)[:300]}") from exc

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
        """Structured output validated against a Pydantic schema, walking the model chain for `tag`."""
        models = self._model_chain(tag)
        key_indices = self._available_key_indices()
        if not key_indices:
            raise GeminiError("All configured Gemini API keys are unavailable or exhausted")

        last: GeminiError | None = None
        for key_index in key_indices:
            try:
                self._activate_key(key_index)
            except Exception:
                self._exhausted_keys.add(key_index)
                last = GeminiError(f"Could not initialize {self._api_keys[key_index][0]}")
                logger.warn(tag, f"{self._api_keys[key_index][0]} could not be initialized; trying the next key")
                continue

            key_limited = False
            for index, name in enumerate(models):
                try:
                    result = self._generate_json_model(
                        name,
                        prompt,
                        schema,
                        images=images,
                        system=system,
                        temperature=temperature,
                        tag=tag,
                        # Bound every model to two attempts. The final fallback must not
                        # consume the full global retry budget after earlier models failed.
                        max_attempts=_ATTEMPTS_BEFORE_FALLBACK,
                    )
                    logger.log(tag, f"model used: {name}; key slot={key_index + 1}")
                    return result
                except GeminiKeyError as exc:
                    last = exc
                    key_limited = True
                    break
                except GeminiError as exc:
                    last = exc
                    if index + 1 < len(models):
                        logger.warn(
                            tag,
                            f"Model '{name}' failed ({str(exc)[:200]}). Falling back to '{models[index + 1]}'",
                        )
                        continue
                    raise
                except Exception as exc:
                    # SDK/runtime failures must not block the configured fallback
                    # model; log only the exception type, not potentially sensitive text.
                    last = GeminiError(
                        f"Unexpected {type(exc).__name__} while calling Gemini model {name}"
                    )
                    logger.warn(
                        tag,
                        f"Unexpected {type(exc).__name__} from model '{name}'; trying the next configured model",
                    )
                    if index + 1 < len(models):
                        continue
                    raise last from exc

            if key_limited:
                continue

        raise GeminiError("Gemini generation failed across all available API keys") from last

    # ------------------------------------------------------------------
    # Image generation (IMAGE_PROVIDER=gemini only)
    def _get_image_client(self, api_key: str = "") -> Any:
        key = api_key or self.cfg.image_api_key
        if not key and self._active_key_index is not None:
            key = self._api_keys[self._active_key_index][1]
        if not key:
            raise ImageGenError("No Gemini API key is configured for image generation")
        if self._image_client is None or self._image_client_api_key != key:
            self._image_client = self._make_client(key, self.cfg.image_request_timeout)
            self._image_client_api_key = key
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
        if not self._api_keys and not self.cfg.image_api_key:
            raise ImageGenError("Gemini client is not configured (missing Gemini API keys)")

        model = (self.cfg.gemini_image_model or "").strip()

        if not model:
            raise ImageGenError("GEMINI_IMAGE_MODEL is empty")

        if "imagen" in model.lower():
            raise ImageGenError("Imagen models are not supported by the Gemini Developer API; use a gemini-*-image model")

        key_options: list[tuple[str, str]] = []
        if self.cfg.image_api_key:
            key_options.append(("IMAGE_API_KEY", self.cfg.image_api_key))
        key_options.extend(self._api_keys)
        unique: list[tuple[str, str]] = []
        for label, key in key_options:
            if key and all(existing[1] != key for existing in unique):
                unique.append((label, key))
        available = [(label, key) for label, key in unique if label not in self._exhausted_image_keys]
        if not available:
            raise ImageQuotaError("All configured Gemini image API keys are unavailable or exhausted")

        delays = self._delays()
        attempts = len(delays) + 1
        last: Exception | None = None
        key_limited = False

        for label, key in available:
            key_limited = False
            try:
                client = self._get_image_client(key)
            except Exception as exc:
                last = exc
                self._exhausted_image_keys.add(label)
                logger.warn(tag, f"{label} image client could not be initialized; trying the next key")
                continue

            for attempt in range(1, attempts + 1):
                try:
                    logger.log(tag, f"Calling generate_content on {model} using {label} (attempt {attempt}/{attempts})")
                    return self._gemini_image(client, model, prompt, references, aspect_ratio)

                except Exception as exc:
                    last = exc
                    if self._key_unavailable(exc):
                        self._exhausted_image_keys.add(label)
                        key_limited = True
                        logger.warn(tag, f"{label} is out of quota, rate-limited, or rejected; trying the next key")
                        break

                    if attempt < attempts and self._retryable(exc):
                        delay = delays[attempt - 1]
                        logger.warn(
                            tag,
                            f"Transient image error on {model} (attempt {attempt}/{attempts}), "
                            f"retrying in {delay}s: {str(exc)[:200]}",
                        )
                        time.sleep(delay)
                        continue

                    break

            if not key_limited:
                break

        if key_limited or (last is not None and self._key_unavailable(last)):
            raise ImageQuotaError("No configured Gemini image key has usable quota") from last
        raise ImageGenError(f"Image generation failed on {model}: {last}") from last
