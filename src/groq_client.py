"""Groq chat-completions client with Pydantic-validated JSON and image inputs."""
from __future__ import annotations

import base64
import json
from typing import Any, TypeVar

import requests
from pydantic import BaseModel

from .config import Settings

T = TypeVar("T", bound=BaseModel)
GROQ_API_BASE_URL = "https://api.groq.com/openai/v1"


class GroqError(Exception):
    """A Groq request failed or returned an invalid structured response."""


class GroqClient:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.api_key = str(cfg.groq_api_key or "").strip()
        self.base_url = GROQ_API_BASE_URL
        self.model = str(cfg.groq_model or "openai/gpt-oss-120b").strip()
        self.vision_model = str(cfg.groq_vision_model or "qwen/qwen3.8-27b").strip()

    def is_configured(self) -> bool:
        return bool(self.api_key)

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

        try:
            schema_json = json.dumps(schema.model_json_schema(), ensure_ascii=False)
            system_text = (system or "You are a precise assistant.").strip()
            system_text += (
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

            temp = min(2.0, max(0.01, float(temperature)))
            payload = {
                "model": self.vision_model if image_inputs else self.model,
                "messages": [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": content},
                ],
                "temperature": temp,
                "max_completion_tokens": 8192,
                "response_format": {"type": "json_object"},
                "stream": False,
            }
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
                raise GroqError(f"Groq returned HTTP {response.status_code} for {tag}")
            body = response.json()
            choices = body.get("choices") or []
            if not choices:
                raise GroqError(f"Groq returned no choices for {tag}")
            raw = choices[0].get("message", {}).get("content")
            if not isinstance(raw, str) or not raw.strip():
                raise GroqError(f"Groq returned empty JSON content for {tag}")
            text = raw.strip()
            if text.startswith("```"):
                text = text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            if not text.startswith("{"):
                start, end = text.find("{"), text.rfind("}")
                if start >= 0 and end > start:
                    text = text[start : end + 1]
            return schema.model_validate_json(text)
        except GroqError:
            raise
        except requests.RequestException as exc:
            raise GroqError(f"Groq network request failed for {tag}: {type(exc).__name__}") from exc
        except Exception as exc:
            raise GroqError(f"Groq response validation failed for {tag}: {type(exc).__name__}") from exc
