# src/config.py
"""Central configuration: typed env loading, safe defaults, validation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

DEFAULT_EXPORTS_DIR = "data/exports"


def _raw(key: str, env: Mapping[str, str] | None) -> str | None:
    source = env if env is not None else os.environ
    return source.get(key)


def _str(key: str, default: str, env: Mapping[str, str] | None = None) -> str:
    v = _raw(key, env)
    if v is None:
        return default
    s = str(v).strip()
    return s if s else default


def _int(key: str, default: int, env: Mapping[str, str] | None = None) -> int:
    v = _raw(key, env)
    if v is None:
        return default
    try:
        return int(str(v).strip())
    except (TypeError, ValueError):
        return default


def _float(key: str, default: float, env: Mapping[str, str] | None = None) -> float:
    v = _raw(key, env)
    if v is None:
        return default
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return default


def _bool(key: str, default: bool, env: Mapping[str, str] | None = None) -> bool:
    v = _raw(key, env)
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("1", "true", "yes", "y", "on"):
        return True
    if s in ("0", "false", "no", "n", "off"):
        return False
    return default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


@dataclass(frozen=True)
class Settings:
    # --- run behaviour
    dry_run: bool = True
    dry_run_generate_images: bool = False
    manual_override: bool = False
    git_push_enabled: bool = False

    number_of_stories: int = 0          # 0 => use max_stories_per_run
    target_daily_stories: int = 6
    max_stories_per_run: int = 3
    max_candidates_for_triage: int = 60
    max_source_age_hours: int = 48
    min_viral_score: int = 60
    max_story_attempts: int = 3
    min_article_chars: int = 600

    # --- editorial / image
    people_image_style: str = "reference"   # reference | illustration | faceless
    facebook_separate_image: bool = True
    image_vlm_check: bool = True
    image_required: bool = True

    # --- time
    day_timezone: str = "Africa/Casablanca"

    # --- gemini
    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-3.1-flash-image"
    gemini_api_key: str = ""
    image_api_key: str = ""
    llm_timeout: int = 180

    # --- network
    request_timeout: int = 60
    image_request_timeout: int = 180
    max_retries: int = 3
    per_host_delay_seconds: float = 2.0
    user_agent: str = "ViralStoriesFactoryBot/1.0"

    # --- export
    export_enabled: bool = True
    exports_dir: str = DEFAULT_EXPORTS_DIR

    # --- public images
    public_images_base: str = ""

    # --- files
    data_dir: str = "data"
    dry_run_dir: str = "data/dry_run"
    history_file: str = "data/history.json"
    cache_dir: str = "data/cache"
    sources_override_file: str = ""

    # --- blogger
    blogger_blog_id: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    # --- twilio
    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""
    your_personal_number: str = ""
    whatsapp_content_sid: str = ""

    # ------------------------------------------------------------------
    def __post_init__(self) -> None:
        def put(name: str, value: object) -> None:
            object.__setattr__(self, name, value)

        put("number_of_stories", int(_clamp(int(self.number_of_stories), 0, 10)))
        put("target_daily_stories", int(_clamp(int(self.target_daily_stories), 1, 50)))
        put("max_stories_per_run", int(_clamp(int(self.max_stories_per_run), 1, 10)))
        put("max_candidates_for_triage", int(_clamp(int(self.max_candidates_for_triage), 10, 500)))
        put("max_source_age_hours", int(_clamp(int(self.max_source_age_hours), 1, 168)))
        put("min_viral_score", int(_clamp(int(self.min_viral_score), 0, 100)))
        put("max_story_attempts", int(_clamp(int(self.max_story_attempts), 1, 20)))
        put("min_article_chars", int(_clamp(int(self.min_article_chars), 100, 20000)))
        put("llm_timeout", int(_clamp(int(self.llm_timeout), 30, 600)))
        put("request_timeout", int(_clamp(int(self.request_timeout), 5, 300)))
        put("image_request_timeout", int(_clamp(int(self.image_request_timeout), 10, 900)))
        put("max_retries", int(_clamp(int(self.max_retries), 0, 10)))
        put("per_host_delay_seconds", float(_clamp(float(self.per_host_delay_seconds), 0.0, 60.0)))

        style = str(self.people_image_style or "").strip().lower()
        put("people_image_style", style if style in {"reference", "illustration", "faceless"} else "illustration")

        defaults = {
            "data_dir": "data",
            "dry_run_dir": "data/dry_run",
            "history_file": "data/history.json",
            "cache_dir": "data/cache",
            "exports_dir": DEFAULT_EXPORTS_DIR,
            "user_agent": "ViralStoriesFactoryBot/1.0",
            "day_timezone": "Africa/Casablanca",
        }
        for name, default in defaults.items():
            put(name, str(getattr(self, name) or "").strip() or default)

        put("public_images_base", str(self.public_images_base or "").strip().rstrip("/"))

    # ------------------------------------------------------------------
    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "Settings":
        return cls(
            dry_run=_bool("DRY_RUN", True, env),
            dry_run_generate_images=_bool("DRY_RUN_GENERATE_IMAGES", False, env),
            manual_override=_bool("MANUAL_OVERRIDE", False, env),
            git_push_enabled=_bool("GIT_PUSH_ENABLED", False, env),
            number_of_stories=_int("NUMBER_OF_STORIES", 0, env),
            target_daily_stories=_int("TARGET_DAILY_STORIES", 6, env),
            max_stories_per_run=_int("MAX_STORIES_PER_RUN", 3, env),
            max_candidates_for_triage=_int("MAX_CANDIDATES_FOR_TRIAGE", 60, env),
            max_source_age_hours=_int("MAX_SOURCE_AGE_HOURS", 48, env),
            min_viral_score=_int("MIN_VIRAL_SCORE", 60, env),
            max_story_attempts=_int("MAX_STORY_ATTEMPTS", 3, env),
            min_article_chars=_int("MIN_ARTICLE_CHARS", 600, env),
            people_image_style=_str("PEOPLE_IMAGE_STYLE", "reference", env),
            facebook_separate_image=_bool("FACEBOOK_SEPARATE_IMAGE", True, env),
            image_vlm_check=_bool("IMAGE_VLM_CHECK", True, env),
            image_required=_bool("IMAGE_REQUIRED", True, env),
            day_timezone=_str("DAY_TIMEZONE", "Africa/Casablanca", env),
            gemini_model=_str("GEMINI_MODEL", "gemini-3.8-flash", env),
            gemini_fallback_model=_str("GEMINI_FALLBACK_MODEL", "gemini-2.5-flash", env),
            gemini_image_model=_str("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-image", env),
            gemini_api_key=_str("GEMINI_API_KEY", "", env),
            image_api_key=_str("IMAGE_API_KEY", "", env),
            llm_timeout=_int("LLM_TIMEOUT", 180, env),
            request_timeout=_int("REQUEST_TIMEOUT", 60, env),
            image_request_timeout=_int("IMAGE_REQUEST_TIMEOUT", 180, env),
            max_retries=_int("MAX_RETRIES", 3, env),
            per_host_delay_seconds=_float("PER_HOST_DELAY_SECONDS", 2.0, env),
            user_agent=_str("USER_AGENT", "ViralStoriesFactoryBot/1.0", env),
            export_enabled=_bool("EXPORT_ENABLED", True, env),
            exports_dir=_str("EXPORTS_DIR", DEFAULT_EXPORTS_DIR, env),
            public_images_base=_str("IMAGE_PUBLIC_BASE_URL", "", env),
            data_dir=_str("DATA_DIR", "data", env),
            dry_run_dir=_str("DRY_RUN_DIR", "data/dry_run", env),
            history_file=_str("HISTORY_FILE", "data/history.json", env),
            cache_dir=_str("CACHE_DIR", "data/cache", env),
            sources_override_file=_str("SOURCES_OVERRIDE_FILE", "", env),
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

    # ------------------------------------------------------------------
    # Derived paths
    @property
    def images_dir(self) -> Path:
        return Path(self.data_dir) / "images"

    @property
    def export_path(self) -> Path:
        """Dry-run exports never mix with live exports (unless EXPORTS_DIR was customised)."""
        if self.dry_run and self.exports_dir == DEFAULT_EXPORTS_DIR:
            return Path(self.dry_run_dir) / "exports"
        return Path(self.exports_dir)

    # ------------------------------------------------------------------
    # Readiness
    def is_blogger_ready(self) -> bool:
        return bool(
            self.blogger_blog_id
            and self.google_client_id
            and self.google_client_secret
            and self.google_refresh_token
        )

    def twilio_configured(self) -> bool:
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_whatsapp_number
            and self.your_personal_number
        )

    # ------------------------------------------------------------------
    def validate(self, need_publish: bool = False) -> list[str]:
        problems: list[str] = []

        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is missing")

        try:
            ZoneInfo(self.day_timezone)
        except Exception:
            problems.append(f"DAY_TIMEZONE is invalid: {self.day_timezone}")

        if need_publish:
            for name, value in (
                ("BLOGGER_BLOG_ID", self.blogger_blog_id),
                ("GOOGLE_CLIENT_ID", self.google_client_id),
                ("GOOGLE_CLIENT_SECRET", self.google_client_secret),
                ("GOOGLE_REFRESH_TOKEN", self.google_refresh_token),
            ):
                if not value:
                    problems.append(f"{name} is missing (required for live publishing)")

        return problems

    def warnings(self, need_publish: bool = False) -> list[str]:
        out: list[str] = []

        if self.image_required and not (self.image_api_key or self.gemini_api_key):
            out.append("No IMAGE_API_KEY or GEMINI_API_KEY; image generation will fail.")

        if need_publish:
            if not self.twilio_configured():
                out.append("Twilio WhatsApp is not fully configured; WhatsApp delivery will be skipped.")
            if not self.public_images_base:
                out.append(
                    "IMAGE_PUBLIC_BASE_URL is empty; images will be embedded inline in Blogger."
                )
            elif not self.git_push_enabled:
                out.append(
                    "GIT_PUSH_ENABLED=false; public image URLs work only if the files "
                    "are already pushed to the repository."
                )

        if self.llm_timeout < 120:
            out.append("LLM_TIMEOUT is below 120 seconds; long Gemini requests may time out.")

        return out

    def secret_values(self) -> list[str]:
        values = [
            self.gemini_api_key,
            self.image_api_key,
            self.google_client_id,
            self.google_client_secret,
            self.google_refresh_token,
            self.twilio_account_sid,
            self.twilio_auth_token,
            self.twilio_whatsapp_number,
            self.your_personal_number,
            self.whatsapp_content_sid,
        ]
        return [v for v in values if v and len(v) >= 5]
