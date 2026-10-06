"""Task-aware failover between Gemini and Groq."""
from __future__ import annotations

import re
from typing import Any, TypeVar

from pydantic import BaseModel

from . import logger
from .gemini_client import GeminiClient, GeminiError
from .groq_client import GroqClient

T = TypeVar("T", bound=BaseModel)


class LLMRouter:
    """Gemini writes article bodies; Groq leads other text/analysis tasks.

    Each task falls back to the other provider when its preferred provider is
    missing or fails. Image generation remains delegated to Gemini.
    """

    _GEMINI_PRIMARY_TAGS = {"ARTICLE"}

    def __init__(self, gemini: GeminiClient, groq: GroqClient) -> None:
        self.gemini = gemini
        self.groq = groq

    @staticmethod
    def _ready(client: Any) -> bool:
        check = getattr(client, "is_configured", None)
        return bool(check()) if callable(check) else True

    @staticmethod
    def _safe_error(client: Any, exc: Exception) -> str:
        """Return a short diagnostic while removing any credential-like values."""
        message = " ".join(str(exc or type(exc).__name__).split())
        secrets: list[str] = []
        api_key = getattr(client, "api_key", "")
        if api_key:
            secrets.append(str(api_key))
        for item in getattr(client, "_api_keys", []) or []:
            if isinstance(item, (tuple, list)) and len(item) > 1 and item[1]:
                secrets.append(str(item[1]))
        for secret in sorted(set(secrets), key=len, reverse=True):
            message = message.replace(secret, "***")
        message = re.sub(r"(?i)bearer\s+\S+", "Bearer ***", message)
        return message[:240] or type(exc).__name__

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        images: list[tuple[bytes, str]] | None = None,
        system: str | None = None,
        temperature: float = 0.7,
        tag: str = "LLM",
    ) -> T:
        if str(tag).upper() in self._GEMINI_PRIMARY_TAGS:
            preferred = [("Gemini", self.gemini), ("Groq", self.groq)]
        else:
            preferred = [("Groq", self.groq), ("Gemini", self.gemini)]

        errors: list[str] = []
        attempted = False
        for name, client in preferred:
            if not self._ready(client):
                continue
            attempted = True
            try:
                result = client.generate_json(
                    prompt,
                    schema,
                    images=images,
                    system=system,
                    temperature=temperature,
                    tag=tag,
                )
                logger.log(tag, f"provider used: {name}")
                return result
            except Exception as exc:
                reason = self._safe_error(client, exc)
                errors.append(f"{name}:{type(exc).__name__}:{reason}")
                logger.warn(
                    tag,
                    f"{name} failed ({type(exc).__name__}: {reason}); trying the other provider",
                )

        if not attempted:
            raise GeminiError("No configured Gemini or Groq API key is available")
        raise GeminiError(f"All configured text providers failed for {tag}: {' | '.join(errors)}")

    def generate_image(self, *args: Any, **kwargs: Any) -> Any:
        """Image generation is not a Groq task; keep using the configured Gemini image model."""
        return self.gemini.generate_image(*args, **kwargs)
