"""Groq chat-completions client with validated JSON and image inputs."""
from __future__ import annotations

import base64
import json
import time
from typing import Any, TypeVar

import requests
from pydantic import BaseModel, ValidationError

from . import logger
from .config import Settings

T = TypeVar("T", bound=BaseModel)
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"
RETRYABLE_HTTP_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}
MAX_ATTEMPTS = 2
MAX_COMPLETION_TOKENS_BY_TAG = {
    "ARTICLE": 4096,
    "TRIAGE": 4096,
    "METADATA": 2048,
    "FACTCHECK": 1024,
    "IMGCHECK": 512,
    "VISUAL": 1024,
}
DEFAULT_MAX_COMPLETION_TOKENS = 2048


class GroqError(Exception):
    """A Groq request failed or returned an invalid structured response."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int | None = None,
        retryable: bool = False,
        error_code: str = "",
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.retryable = retryable
        self.error_code = error_code


class GroqClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.api_key = str(cfg.groq_api_key or "").strip()
        self.base_url = GROQ_API_BASE_URL
        self.model = str(cfg.groq_model or "openai/gpt-oss-20b").strip()
        self.vision_model = str(cfg.groq_vision_model or "").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

    def _safe(self, message: str) -> str:
        """Keep provider errors useful without ever echoing the API secret."""
        value = " ".join(str(message or "").split())
        if self.api_key:
            value = value.replace(self.api_key, "***")
        return value[:240]

    def _http_error(self, response: requests.Response, tag: str) -> GroqError:
        status = int(response.status_code)
        code = ""
        detail = ""
        try:
            body = response.json()
            error = body.get("error", body) if isinstance(body, dict) else {}
            if isinstance(error, dict):
                code = str(error.get("code") or "")
                detail = str(error.get("message") or error.get("detail") or "")
        except Exception:
            detail = str(getattr(response, "text", "") or "")
        suffix = f" code={code}" if code else ""
        if detail:
            suffix += f": {self._safe(detail)}"
        lower_detail = detail.lower()
        retryable_tpm_limit = (
            status == 413
            and code.lower() == "rate_limit_exceeded"
            and "tokens per minute" in lower_detail
        )
        return GroqError(
            f"Groq returned HTTP {status}{suffix} for {tag}",
            status_code=status,
            retryable=(
                status in RETRYABLE_HTTP_STATUS
                or code.lower() == "json_validate_failed"
                or retryable_tpm_limit
            ),
            error_code=code,
        )

    @staticmethod
    def _validation_detail(exc: ValidationError) -> str:
        errors = exc.errors()
        if not errors:
            return "schema mismatch"
        first = errors[0]
        location = ".".join(str(item) for item in first.get("loc", ())) or "response"
        return f"field={location}, issue={first.get('type', 'invalid')}"

    def _parse_response(self, response: requests.Response, schema: type[T], tag: str) -> T:
        try:
            body = response.json()
        except Exception as exc:
            raise GroqError(
                f"Groq returned invalid response JSON for {tag}: {type(exc).__name__}",
                retryable=True,
            ) from exc

        choices = body.get("choices") if isinstance(body, dict) else None
        if not choices:
            raise GroqError(f"Groq returned no choices for {tag}", retryable=True)

        choice = choices[0] if isinstance(choices[0], dict) else {}
        message = choice.get("message") or {}
        raw = message.get("content") if isinstance(message, dict) else None
        if isinstance(raw, list):
            raw = "".join(
                str(part.get("text", ""))
                for part in raw
                if isinstance(part, dict) and part.get("type") == "text"
            )
        if not isinstance(raw, str) or not raw.strip():
            finish = str(choice.get("finish_reason") or "unknown")
            raise GroqError(
                f"Groq returned empty JSON content for {tag} (finish_reason={finish})",
                retryable=True,
            )

        text = raw.strip()
        if text.startswith("```"):
            text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
        if not text.startswith("{"):
            start, end = text.find("{"), text.rfind("}")
            if start >= 0 and end > start:
                text = text[start : end + 1]

        try:
            parsed = json.loads(text)
        except json.JSONDecodeError as exc:
            raise GroqError(
                f"Groq returned malformed JSON for {tag} at {exc.lineno}:{exc.colno}",
                retryable=True,
            ) from exc
        try:
            return schema.model_validate(parsed)
        except ValidationError as exc:
            raise GroqError(
                f"Groq JSON did not match {schema.__name__} for {tag}: {self._validation_detail(exc)}",
                retryable=True,
            ) from exc

    def generate_json(
        self,
        prompt: str,
        schema: type[T],
        *,
        images: list[tuple[bytes, str]] | None = None,
        system: str | None = None,
        temperature: float = 0.3,
        tag: str = "GROQ",
    ) -> T:
        if not self.api_key:
            raise GroqError("GROQ_API_KEY is not configured")

        image_inputs = images or []
        if len(image_inputs) > 3:
            raise GroqError("Groq vision accepts at most 3 images per request")
        if image_inputs and not self.vision_model:
            raise GroqError(
                "GROQ_VISION_MODEL is disabled in the Free-tier defaults; "
                "route image analysis through Gemini Flash-Lite"
            )

        model = self.vision_model if image_inputs else self.model
        try:
            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            instructions = (system or "You are a precise assistant.").strip()
            instructions += (
                "\n\nReturn only one valid JSON object matching this JSON Schema exactly. "
                "Do not wrap it in Markdown or add commentary.\n" + schema_json
            )

            content: Any = prompt
            if image_inputs:
                parts: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
                for data, mime in image_inputs:
                    encoded = base64.b64encode(data).decode("ascii")
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime or 'image/jpeg'};base64,{encoded}"},
                    })
                content = parts

            temperature_value = min(2.0, max(0.01, float(temperature)))
            reasoning_model = model.startswith("openai/gpt-oss-") or model == "qwen/qwen3.8-27b"
            if reasoning_model:
                # Groq recommends low reasoning effort for concise JSON tasks; GPT-OSS/Qwen
                # support hidden reasoning with JSON mode. Put instructions in user content
                # for these reasoning models, as recommended by Groq's API guidance.
                user_content: Any = (
                    f"{instructions}\n\nREQUEST:\n{content}"
                    if isinstance(content, str)
                    else [
                        {"type": "text", "text": f"{instructions}\n\nREQUEST:\n{prompt}"},
                        *content[1:],
                    ]
                )
                messages = [{"role": "user", "content": user_content}]
            else:
                messages = [
                    {"role": "system", "content": instructions},
                    {"role": "user", "content": content},
                ]

            payload: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "temperature": temperature_value,
                "max_completion_tokens": MAX_COMPLETION_TOKENS_BY_TAG.get(
                    tag.upper(), DEFAULT_MAX_COMPLETION_TOKENS
                ),
                "response_format": {"type": "json_object"},
                "stream": False,
            }
            if reasoning_model:
                payload["reasoning_effort"] = "low"
                payload["reasoning_format"] = "hidden"

            last_error: GroqError | None = None
            for attempt in range(1, MAX_ATTEMPTS + 1):
                try:
                    response = requests.post(
                        f"{self.base_url}/chat/completions",
                        headers={
                            "Authorization": f"Bearer {self.api_key}",
                            "Content-Type": "application/json",
                        },
                        json=payload,
                        timeout=int(self.cfg.llm_timeout),
                    )
                    if response.status_code >= 400:
                        raise self._http_error(response, tag)
                    return self._parse_response(response, schema, tag)
                except GroqError as exc:
                    last_error = exc
                    if attempt < MAX_ATTEMPTS and exc.retryable:
                        logger.warn(tag, f"{self._safe(str(exc))}; retrying once")
                        if exc.status_code == 413 and exc.error_code.lower() == "rate_limit_exceeded":
                            # A 413 TPM response includes the requested completion budget.
                            # Halve that budget on the single retry; never retry an
                            # authentication, input-size, or unrelated 413 response.
                            payload["max_completion_tokens"] = max(
                                256, int(payload["max_completion_tokens"]) // 2
                            )
                        if exc.status_code is None or exc.error_code.lower() == "json_validate_failed":
                            # Ask for a fresh valid object after client- or provider-side
                            # JSON validation errors, and reduce randomness for the retry.
                            payload["temperature"] = min(temperature_value, 0.1)
                            correction = (
                                "Your previous response was empty, malformed, or did not match "
                                "the required JSON schema. Try again from scratch and return "
                                "one complete valid JSON object only."
                            )
                            if messages and isinstance(messages[-1].get("content"), str):
                                messages.append({"role": "user", "content": correction})
                        time.sleep(0.8)
                        continue
                    raise
                except requests.RequestException as exc:
                    safe = self._safe(type(exc).__name__)
                    last_error = GroqError(
                        f"Groq network request failed for {tag}: {safe}",
                        retryable=True,
                    )
                    if attempt < MAX_ATTEMPTS:
                        logger.warn(tag, f"{last_error}; retrying once")
                        time.sleep(0.8)
                        continue
                    raise last_error from exc

            raise last_error or GroqError(f"Groq request failed for {tag}")
        except GroqError:
            raise
        except Exception as exc:
            raise GroqError(
                f"Groq request preparation failed for {tag}: {type(exc).__name__}"
            ) from exc
