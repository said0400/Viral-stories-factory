# src/config.py
"""
Central configuration module.
Loads settings from environment variables with strong typing, safe defaults, and strict validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping


def _str(key: str, default: str, env: Mapping[str, str] | None = None) -> str:
    source = env if env is not None else os.environ
    val = source.get(key)
    if val is None:
        return default
    s = str(val).strip()
    return s if s else default


def _int(key: str, default: int, env: Mapping[str, str] | None = None) -> int:
    source = env if env is not None else os.environ
    raw = source.get(key)
    if raw is None:
        return default
    try:
        return int(str(raw).strip())
    except ValueError:
        return default


def _bool(key: str, default: bool, env: Mapping[str, str] | None = None) -> bool:
    source = env if env is not None else os.environ
    raw = source.get(key)
    if raw is None:
        return default
    cleaned = str(raw).strip().lower()
    if cleaned in ("1", "true", "yes", "y", "on"):
        return True
    if cleaned in ("0", "false", "no", "n", "off"):
        return False
    return default


@dataclass(frozen=True)
class Settings:
    dry_run: bool = True
    manual_override: bool = False
    git_push_enabled: bool = True
    number_of_stories: int = 1
    target_daily_stories: int = 6
    max_stories_per_run: int = 3
    max_source_age_hours: int = 48
    people_image_style: str = "reference"
    day_timezone: str = "Africa/Casablanca"

    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-3.1-flash-image"
    gemini_api_key: str = ""
    image_api_key: str = ""

    llm_timeout: int = 180

    export_enabled: bool = True
    exports_dir: str = "data/exports"

    blogger_blog_id: str = ""

    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""
    your_personal_number: str = ""
    whatsapp_content_sid: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "number_of_stories", max(1, min(10, self.number_of_stories)))
        object.__setattr__(self, "target_daily_stories", max(1, min(50, self.target_daily_stories)))
        object.__setattr__(self, "max_stories_per_run", max(1, min(10, self.max_stories_per_run)))
        object.__setattr__(self, "max_source_age_hours", max(1, min(168, self.max_source_age_hours)))
        object.__setattr__(self, "llm_timeout", max(30, min(600, self.llm_timeout)))

        if self.dry_run and self.exports_dir == "data/exports":
            object.__setattr__(self, "exports_dir", "data/dry_run/exports")

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> Settings:
        return cls(
            dry_run=_bool("DRY_RUN", True, env),
            manual_override=_bool("MANUAL_OVERRIDE", False, env),
            git_push_enabled=_bool("GIT_PUSH_ENABLED", True, env),
            number_of_stories=_int("NUMBER_OF_STORIES", 1, env),
            target_daily_stories=_int("TARGET_DAILY_STORIES", 6, env),
            max_stories_per_run=_int("MAX_STORIES_PER_RUN", 3, env),
            max_source_age_hours=_int("MAX_SOURCE_AGE_HOURS", 48, env),
            people_image_style=_str("PEOPLE_IMAGE_STYLE", "reference", env),
            day_timezone=_str("DAY_TIMEZONE", "Africa/Casablanca", env),
            gemini_model=_str("GEMINI_MODEL", "gemini-3.8-flash", env),
            gemini_fallback_model=_str("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash", env),
            gemini_image_model=_str("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image", env),
            gemini_api_key=_str("GEMINI_API_KEY", "", env),
            image_api_key=_str("IMAGE_API_KEY", "", env),
            llm_timeout=_int("LLM_TIMEOUT", 180, env),
            export_enabled=_bool("EXPORT_ENABLED", True, env),
            exports_dir=_str("EXPORTS_DIR", "data/exports", env),
            blogger_blog_id=_str("BLOGGER_BLOG_ID", "", env),
            google_client_id=_str("GOOGLE_CLIENT_ID", "", env),
            google_client_secret=_str("GOOGLE_CLIENT_SECRET", "", env),
            google_refresh_token=_str("GOOGLE_REFRESH_TOKEN", "", env),
            twilio_account_sid=_str("TWILIO_ACCOUNT_SID", "", env),
            twilio_auth_token=_str("TWILIO_AUTH_TOKEN", "", env),
            twilio_whatsapp_number=_str("TWILIO_WHATSAPP_NUMBER", "", env),
            your_personal_number=_str("YOUR_PERSONAL_NUMBER", "", env),
            whatsapp_content_sid=_str("WHATSAPP_CONTENT_SID", "", env),
        )

    def is_gemini_ready(self) -> bool:
        return bool(self.gemini_api_key)

    def is_blogger_ready(self) -> bool:
        return bool(
            self.blogger_blog_id
            and self.google_client_id
            and self.google_client_secret
            and self.google_refresh_token
        )

    def is_whatsapp_ready(self) -> bool:
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_whatsapp_number
            and self.your_personal_number
        )

    def secrets_to_redact(self) -> list[str]:
        secrets: list[str] = []
        for val in [
            self.gemini_api_key,
            self.image_api_key,
            self.google_client_secret,
            self.google_refresh_token,
            self.twilio_auth_token,
            self.twilio_account_sid,
        ]:
            if val and len(val) >= 4:
                secrets.append(val)
        return secrets
