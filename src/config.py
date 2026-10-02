# src/config.py
"""
Central configuration module.

Loads settings from environment variables with strong typing, safe defaults,
derived paths, and strict validation.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


def _str(
    key: str,
    default: str,
    env: Mapping[str, str] | None = None,
) -> str:
    source = env if env is not None else os.environ
    val = source.get(key)

    if val is None:
        return default

    s = str(val).strip()

    return s if s else default


def _int(
    key: str,
    default: int,
    env: Mapping[str, str] | None = None,
) -> int:
    source = env if env is not None else os.environ
    raw = source.get(key)

    if raw is None:
        return default

    try:
        return int(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _float(
    key: str,
    default: float,
    env: Mapping[str, str] | None = None,
) -> float:
    source = env if env is not None else os.environ
    raw = source.get(key)

    if raw is None:
        return default

    try:
        return float(str(raw).strip())
    except (TypeError, ValueError):
        return default


def _bool(
    key: str,
    default: bool,
    env: Mapping[str, str] | None = None,
) -> bool:
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
    # ------------------------------------------------------------------
    # Run behaviour
    # ------------------------------------------------------------------

    dry_run: bool = True
    manual_override: bool = False
    git_push_enabled: bool = True

    number_of_stories: int = 1
    target_daily_stories: int = 6
    max_stories_per_run: int = 3
    max_candidates_for_triage: int = 60

    max_source_age_hours: int = 48

    # ------------------------------------------------------------------
    # Editorial / image behaviour
    # ------------------------------------------------------------------

    people_image_style: str = "reference"

    image_vlm_check: bool = True
    image_required: bool = True

    # ------------------------------------------------------------------
    # Time / locale
    # ------------------------------------------------------------------

    day_timezone: str = "Africa/Casablanca"

    # ------------------------------------------------------------------
    # Gemini
    # ------------------------------------------------------------------

    gemini_model: str = "gemini-3.8-flash"
    gemini_fallback_model: str = "gemini-2.5-flash"
    gemini_image_model: str = "gemini-3.1-flash-image"

    gemini_api_key: str = ""
    image_api_key: str = ""

    llm_timeout: int = 180

    # ------------------------------------------------------------------
    # Network
    # ------------------------------------------------------------------

    request_timeout: int = 60
    image_request_timeout: int = 180

    max_retries: int = 3

    per_host_delay_seconds: float = 2.0

    user_agent: str = "ViralStoriesFactoryBot/1.0"

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    export_enabled: bool = True
    exports_dir: str = "data/exports"

    # ------------------------------------------------------------------
    # Public image hosting
    # ------------------------------------------------------------------

    public_images_base: str = ""

    # ------------------------------------------------------------------
    # Files
    # ------------------------------------------------------------------

    data_dir: str = "data"
    dry_run_dir: str = "data/dry_run"

    history_file: str = "data/history.json"
    cache_dir: str = "data/cache"

    sources_override_file: str = ""

    # ------------------------------------------------------------------
    # Blogger / Google OAuth
    # ------------------------------------------------------------------

    blogger_blog_id: str = ""

    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    # ------------------------------------------------------------------
    # Twilio WhatsApp
    # ------------------------------------------------------------------

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""
    your_personal_number: str = ""
    whatsapp_content_sid: str = ""

    # ------------------------------------------------------------------
    # Normalisation / validation
    # ------------------------------------------------------------------

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "number_of_stories",
            max(1, min(10, int(self.number_of_stories))),
        )

        object.__setattr__(
            self,
            "target_daily_stories",
            max(1, min(50, int(self.target_daily_stories))),
        )

        object.__setattr__(
            self,
            "max_stories_per_run",
            max(1, min(10, int(self.max_stories_per_run))),
        )

        object.__setattr__(
            self,
            "max_candidates_for_triage",
            max(10, min(500, int(self.max_candidates_for_triage))),
        )

        object.__setattr__(
            self,
            "max_source_age_hours",
            max(1, min(168, int(self.max_source_age_hours))),
        )

        object.__setattr__(
            self,
            "llm_timeout",
            max(30, min(600, int(self.llm_timeout))),
        )

        object.__setattr__(
            self,
            "request_timeout",
            max(5, min(300, int(self.request_timeout))),
        )

        object.__setattr__(
            self,
            "image_request_timeout",
            max(10, min(900, int(self.image_request_timeout))),
        )

        object.__setattr__(
            self,
            "max_retries",
            max(0, min(10, int(self.max_retries))),
        )

        object.__setattr__(
            self,
            "per_host_delay_seconds",
            max(0.0, min(60.0, float(self.per_host_delay_seconds))),
        )

        # Normalise directory strings without resolving them against the
        # current working directory. The factory intentionally uses
        # repository-relative paths.
        for field_name in (
            "data_dir",
            "dry_run_dir",
            "history_file",
            "cache_dir",
            "exports_dir",
        ):
            value = str(getattr(self, field_name) or "").strip()

            if not value:
                if field_name == "data_dir":
                    value = "data"
                elif field_name == "dry_run_dir":
                    value = "data/dry_run"
                elif field_name == "history_file":
                    value = "data/history.json"
                elif field_name == "cache_dir":
                    value = "data/cache"
                else:
                    value = "data/exports"

            object.__setattr__(self, field_name, value)

        # Dry-run exports must never be mixed with live exports.
        #
        # Only replace the default live export directory. If the user
        # explicitly supplied another EXPORTS_DIR, preserve that choice.
        if self.dry_run and self.exports_dir == "data/exports":
            object.__setattr__(
                self,
                "exports_dir",
                "data/dry_run/exports",
            )

    # ------------------------------------------------------------------
    # Environment loader
    # ------------------------------------------------------------------

    @classmethod
    def from_env(
        cls,
        env: Mapping[str, str] | None = None,
    ) -> Settings:
        return cls(
            # ----------------------------------------------------------
            # Run behaviour
            # ----------------------------------------------------------

            dry_run=_bool(
                "DRY_RUN",
                True,
                env,
            ),

            manual_override=_bool(
                "MANUAL_OVERRIDE",
                False,
                env,
            ),

            git_push_enabled=_bool(
                "GIT_PUSH_ENABLED",
                True,
                env,
            ),

            number_of_stories=_int(
                "NUMBER_OF_STORIES",
                1,
                env,
            ),

            target_daily_stories=_int(
                "TARGET_DAILY_STORIES",
                6,
                env,
            ),

            max_stories_per_run=_int(
                "MAX_STORIES_PER_RUN",
                3,
                env,
            ),

            max_candidates_for_triage=_int(
                "MAX_CANDIDATES_FOR_TRIAGE",
                60,
                env,
            ),

            max_source_age_hours=_int(
                "MAX_SOURCE_AGE_HOURS",
                48,
                env,
            ),

            # ----------------------------------------------------------
            # Editorial / image behaviour
            # ----------------------------------------------------------

            people_image_style=_str(
                "PEOPLE_IMAGE_STYLE",
                "reference",
                env,
            ),

            image_vlm_check=_bool(
                "IMAGE_VLM_CHECK",
                True,
                env,
            ),

            image_required=_bool(
                "IMAGE_REQUIRED",
                True,
                env,
            ),

            # ----------------------------------------------------------
            # Time / locale
            # ----------------------------------------------------------

            day_timezone=_str(
                "DAY_TIMEZONE",
                "Africa/Casablanca",
                env,
            ),

            # ----------------------------------------------------------
            # Gemini
            # ----------------------------------------------------------

            gemini_model=_str(
                "GEMINI_MODEL",
                "gemini-3.8-flash",
                env,
            ),

            gemini_fallback_model=_str(
                "GEMINI_FALLBACK_MODEL",
                "gemini-2.5-flash",
                env,
            ),

            gemini_image_model=_str(
                "GEMINI_IMAGE_MODEL",
                "gemini-3.1-flash-image",
                env,
            ),

            gemini_api_key=_str(
                "GEMINI_API_KEY",
                "",
                env,
            ),

            image_api_key=_str(
                "IMAGE_API_KEY",
                "",
                env,
            ),

            llm_timeout=_int(
                "LLM_TIMEOUT",
                180,
                env,
            ),

            # ----------------------------------------------------------
            # Network
            # ----------------------------------------------------------

            request_timeout=_int(
                "REQUEST_TIMEOUT",
                60,
                env,
            ),

            image_request_timeout=_int(
                "IMAGE_REQUEST_TIMEOUT",
                180,
                env,
            ),

            max_retries=_int(
                "MAX_RETRIES",
                3,
                env,
            ),

            per_host_delay_seconds=_float(
                "PER_HOST_DELAY_SECONDS",
                2.0,
                env,
            ),

            user_agent=_str(
                "USER_AGENT",
                "ViralStoriesFactoryBot/1.0",
                env,
            ),

            # ----------------------------------------------------------
            # Export
            # ----------------------------------------------------------

            export_enabled=_bool(
                "EXPORT_ENABLED",
                True,
                env,
            ),

            exports_dir=_str(
                "EXPORTS_DIR",
                "data/exports",
                env,
            ),

            # ----------------------------------------------------------
            # Public images
            # ----------------------------------------------------------

            public_images_base=_str(
                "IMAGE_PUBLIC_BASE_URL",
                "",
                env,
            ),

            # ----------------------------------------------------------
            # Files
            # ----------------------------------------------------------

            data_dir=_str(
                "DATA_DIR",
                "data",
                env,
            ),

            dry_run_dir=_str(
                "DRY_RUN_DIR",
                "data/dry_run",
                env,
            ),

            history_file=_str(
                "HISTORY_FILE",
                "data/history.json",
                env,
            ),

            cache_dir=_str(
                "CACHE_DIR",
                "data/cache",
                env,
            ),

            sources_override_file=_str(
                "SOURCES_OVERRIDE_FILE",
                "",
                env,
            ),

            # ----------------------------------------------------------
            # Blogger / Google OAuth
            # ----------------------------------------------------------

            blogger_blog_id=_str(
                "BLOGGER_BLOG_ID",
                "",
                env,
            ),

            google_client_id=_str(
                "GOOGLE_CLIENT_ID",
                "",
                env,
            ),

            google_client_secret=_str(
                "GOOGLE_CLIENT_SECRET",
                "",
                env,
            ),

            google_refresh_token=_str(
                "GOOGLE_REFRESH_TOKEN",
                "",
                env,
            ),

            # ----------------------------------------------------------
            # Twilio WhatsApp
            # ----------------------------------------------------------

            twilio_account_sid=_str(
                "TWILIO_ACCOUNT_SID",
                "",
                env,
            ),

            twilio_auth_token=_str(
                "TWILIO_AUTH_TOKEN",
                "",
                env,
            ),

            twilio_whatsapp_number=_str(
                "TWILIO_WHATSAPP_NUMBER",
                "",
                env,
            ),

            your_personal_number=_str(
                "YOUR_PERSONAL_NUMBER",
                "",
                env,
            ),

            whatsapp_content_sid=_str(
                "WHATSAPP_CONTENT_SID",
                "",
                env,
            ),
        )

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

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

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        errors: list[str] = []

        if not self.gemini_api_key:
            errors.append("GEMINI_API_KEY is missing")

        if self.number_of_stories < 1:
            errors.append("NUMBER_OF_STORIES must be >= 1")

        if self.target_daily_stories < 1:
            errors.append("TARGET_DAILY_STORIES must be >= 1")

        if self.max_stories_per_run < 1:
            errors.append("MAX_STORIES_PER_RUN must be >= 1")

        if self.max_source_age_hours < 1:
            errors.append("MAX_SOURCE_AGE_HOURS must be >= 1")

        if self.request_timeout < 5:
            errors.append("REQUEST_TIMEOUT must be >= 5")

        if self.image_request_timeout < 10:
            errors.append("IMAGE_REQUEST_TIMEOUT must be >= 10")

        if self.max_retries < 0:
            errors.append("MAX_RETRIES must be >= 0")

        if not self.user_agent.strip():
            errors.append("USER_AGENT must not be empty")

        if self.export_enabled and not self.exports_dir.strip():
            errors.append("EXPORTS_DIR must not be empty")

        if errors:
            raise ValueError(
                "Invalid configuration:\n- "
                + "\n- ".join(errors)
            )

    # ------------------------------------------------------------------
    # Non-fatal configuration warnings
    # ------------------------------------------------------------------

    def warnings(self) -> list[str]:
        warnings: list[str] = []

        if not self.is_blogger_ready() and not self.dry_run:
            warnings.append(
                "Blogger credentials are incomplete; live publishing "
                "will not be available."
            )

        if not self.is_whatsapp_ready():
            warnings.append(
                "Twilio WhatsApp is not fully configured; "
                "WhatsApp delivery will be skipped."
            )

        if self.image_required and not self.image_api_key:
            warnings.append(
                "IMAGE_API_KEY is missing while IMAGE_REQUIRED=true; "
                "image generation may fail."
            )

        if self.git_push_enabled and not self.public_images_base:
            warnings.append(
                "IMAGE_PUBLIC_BASE_URL is not configured; "
                "generated images may not receive a public URL."
            )

        if self.dry_run and self.export_enabled:
            warnings.append(
                f"Dry-run exports will be written to {self.exports_dir}."
            )

        if self.llm_timeout < 120:
            warnings.append(
                "LLM_TIMEOUT is below 120 seconds; complex Gemini "
                "requests may time out."
            )

        return warnings

    # ------------------------------------------------------------------
    # Secret redaction
    # ------------------------------------------------------------------

    def secrets_to_redact(self) -> list[str]:
        secrets: list[str] = []

        for val in [
            self.gemini_api_key,
            self.image_api_key,
            self.google_client_id,
            self.google_client_secret,
            self.google_refresh_token,
            self.twilio_auth_token,
            self.twilio_account_sid,
            self.whatsapp_content_sid,
        ]:
            if val and len(val) >= 4:
                secrets.append(val)

        return secrets
