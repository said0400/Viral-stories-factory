"""Environment-driven configuration. No secrets are ever logged."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _bool(name: str, default: bool) -> bool:
    v = os.getenv(name)
    if v is None or v.strip() == "":
        return default
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _int(name: str, default: int) -> int:
    v = os.getenv(name)
    try:
        return int(v) if v and v.strip() else default
    except ValueError:
        return default


def _float(name: str, default: float) -> float:
    v = os.getenv(name)
    try:
        return float(v) if v and v.strip() else default
    except ValueError:
        return default


def _str(name: str, default: str = "") -> str:
    v = os.getenv(name)
    return v.strip() if v and v.strip() else default


@dataclass(frozen=True)
class Settings:
    target_daily_stories: int = 6
    max_stories_per_run: int = 3
    max_source_age_hours: int = 48
    min_viral_score: int = 60
    day_timezone: str = "Africa/Casablanca"

    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.8-flash"
    gemini_image_model: str = "gemini-3.1-flash-image"
    image_api_key: str = ""

    blogger_blog_id: str = ""
    google_client_id: str = ""
    google_client_secret: str = ""
    google_refresh_token: str = ""

    twilio_account_sid: str = ""
    twilio_auth_token: str = ""
    twilio_whatsapp_number: str = ""
    your_personal_number: str = ""
    whatsapp_content_sid: str = ""

    # NOTE:
    # For REAL people, "reference" is treated as "illustration" by the
    # image generator (see image_generator.choose_strategy). This is an
    # intentional ethical/safety limit; the config name is kept for
    # backward compatibility with .env.example.
    people_image_style: str = "reference"  # reference | illustration | faceless
    facebook_separate_image: bool = True
    image_public_base_url: str = ""
    image_vlm_check: bool = True
    image_required: bool = True

    # Network / HTTP for scraping and public image hosting.
    request_timeout: int = 60
    image_request_timeout: int = 180

    # LLM calls (Gemini) need longer than normal HTTP calls because a
    # single structured-JSON article can take 60-120s to complete.
    llm_timeout: int = 180

    max_retries: int = 3
    per_host_delay: float = 2.0
    user_agent: str = "ViralStoriesFactoryBot/1.0"
    max_story_attempts: int = 3
    max_candidates_for_triage: int = 40
    min_article_chars: int = 600

    history_file: Path = Path("data/history.json")
    data_dir: Path = Path("data")
    sources_override_file: str = ""

    dry_run: bool = True
    dry_run_generate_images: bool = False
    git_push_enabled: bool = False
    manual_override: bool = False
    number_of_stories: int = 0  # 0 = automatic

    # Export / downloadable bundles.
    export_enabled: bool = True

    github_repository: str = ""
    github_ref_name: str = "main"

    def __post_init__(self):
        """Clamp numeric values to prevent logical errors (e.g. negative timeouts)."""
        object.__setattr__(self, 'target_daily_stories', max(1, self.target_daily_stories))
        object.__setattr__(self, 'max_stories_per_run', max(1, self.max_stories_per_run))
        object.__setattr__(self, 'min_viral_score', max(0, min(100, self.min_viral_score)))
        object.__setattr__(self, 'request_timeout', max(10, self.request_timeout))
        object.__setattr__(self, 'image_request_timeout', max(30, self.image_request_timeout))
        object.__setattr__(self, 'llm_timeout', max(60, self.llm_timeout))
        object.__setattr__(self, 'max_retries', max(0, self.max_retries))
        object.__setattr__(self, 'max_story_attempts', max(1, self.max_story_attempts))
        object.__setattr__(self, 'min_article_chars', max(100, self.min_article_chars))

    @classmethod
    def from_env(cls) -> "Settings":
        gem = _str("GEMINI_API_KEY")

        return cls(
            target_daily_stories=_int("TARGET_DAILY_STORIES", 6),
            max_stories_per_run=_int("MAX_STORIES_PER_RUN", 3),
            max_source_age_hours=_int("MAX_SOURCE_AGE_HOURS", 48),
            min_viral_score=_int("MIN_VIRAL_SCORE", 60),
            day_timezone=_str("DAY_TIMEZONE", "Africa/Casablanca"),

            gemini_api_key=gem,
            gemini_model=_str("GEMINI_MODEL", "gemini-3.8-flash"),
            gemini_image_model=_str(
                "GEMINI_IMAGE_MODEL",
                "gemini-3.1-flash-image",
            ),
            image_api_key=_str("IMAGE_API_KEY") or gem,

            blogger_blog_id=_str("BLOGGER_BLOG_ID"),
            google_client_id=_str("GOOGLE_CLIENT_ID"),
            google_client_secret=_str("GOOGLE_CLIENT_SECRET"),
            google_refresh_token=_str("GOOGLE_REFRESH_TOKEN"),

            twilio_account_sid=_str("TWILIO_ACCOUNT_SID"),
            twilio_auth_token=_str("TWILIO_AUTH_TOKEN"),
            twilio_whatsapp_number=_str("TWILIO_WHATSAPP_NUMBER"),
            your_personal_number=_str("YOUR_PERSONAL_NUMBER"),
            whatsapp_content_sid=_str("WHATSAPP_CONTENT_SID"),

            people_image_style=_str(
                "PEOPLE_IMAGE_STYLE",
                "reference",
            ).lower(),
            facebook_separate_image=_bool(
                "FACEBOOK_SEPARATE_IMAGE",
                True,
            ),
            image_public_base_url=_str(
                "IMAGE_PUBLIC_BASE_URL"
            ).rstrip("/"),
            image_vlm_check=_bool("IMAGE_VLM_CHECK", True),
            image_required=_bool("IMAGE_REQUIRED", True),

            request_timeout=_int("REQUEST_TIMEOUT", 60),
            image_request_timeout=_int("IMAGE_REQUEST_TIMEOUT", 180),
            llm_timeout=_int("LLM_TIMEOUT", 180),
            max_retries=_int("MAX_RETRIES", 3),
            per_host_delay=_float(
                "PER_HOST_DELAY_SECONDS",
                2.0,
            ),
            user_agent=_str(
                "USER_AGENT",
                "ViralStoriesFactoryBot/1.0",
            ),
            max_story_attempts=_int("MAX_STORY_ATTEMPTS", 3),
            max_candidates_for_triage=_int(
                "MAX_CANDIDATES_FOR_TRIAGE",
                40,
            ),
            min_article_chars=_int("MIN_ARTICLE_CHARS", 600),

            history_file=Path(
                _str("HISTORY_FILE", "data/history.json")
            ),
            data_dir=Path(
                _str("DATA_DIR", "data")
            ),
            sources_override_file=_str(
                "SOURCES_OVERRIDE_FILE"
            ),

            dry_run=_bool("DRY_RUN", True),
            dry_run_generate_images=_bool(
                "DRY_RUN_GENERATE_IMAGES",
                False,
            ),
            git_push_enabled=_bool(
                "GIT_PUSH_ENABLED",
                False,
            ),
            manual_override=_bool(
                "MANUAL_OVERRIDE",
                False,
            ),
            number_of_stories=_int(
                "NUMBER_OF_STORIES",
                0,
            ),

            export_enabled=_bool("EXPORT_ENABLED", True),

            github_repository=_str(
                "GITHUB_REPOSITORY"
            ),
            github_ref_name=_str(
                "GITHUB_REF_NAME",
                "main",
            ),
        )

    # ---- derived -------------------------------------------------------
    @property
    def images_dir(self) -> Path:
        return self.data_dir / "images"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def dry_run_dir(self) -> Path:
        return self.data_dir / "dry_run"

    @property
    def exports_dir(self) -> Path:
        """Where downloadable story bundles (ZIP) are stored."""
        return self.data_dir / "exports"

    @property
    def public_images_base(self) -> str:
        if self.image_public_base_url:
            return self.image_public_base_url

        if self.github_repository:
            return (
                f"https://raw.githubusercontent.com/"
                f"{self.github_repository}/"
                f"{self.github_ref_name}/data/images"
            )

        return ""

    def secret_values(self) -> list[str]:
        vals = [
            self.gemini_api_key,
            self.image_api_key,
            self.google_client_secret,
            self.google_refresh_token,
            self.twilio_auth_token,
            self.twilio_account_sid,
            self.google_client_id,
            self.twilio_whatsapp_number,
            self.your_personal_number,
            self.whatsapp_content_sid,
            self.blogger_blog_id,
        ]

        # Only redact if string is at least 5 chars to avoid redacting common single chars
        return [
            v for v in vals if v and len(v) >= 5
        ]

    def twilio_configured(self) -> bool:
        """Return True when all required Twilio credentials are present."""
        return bool(
            self.twilio_account_sid
            and self.twilio_auth_token
            and self.twilio_whatsapp_number
            and self.your_personal_number
        )

    def validate(self, *, need_publish: bool) -> list[str]:
        """Return a list of human-readable configuration problems.

        Only truly blocking problems are returned here.
        Missing Twilio credentials are NOT blocking: the pipeline will
        simply skip WhatsApp notifications and mark the story as
        'whatsapp_skipped' instead of 'whatsapp_failed'.
        """
        problems: list[str] = []

        if not self.gemini_api_key:
            problems.append("GEMINI_API_KEY is missing")

        if self.people_image_style not in {
            "reference",
            "illustration",
            "faceless",
        }:
            problems.append(
                "PEOPLE_IMAGE_STYLE must be "
                "'reference', 'illustration' or 'faceless'"
            )

        if need_publish:
            for name, val in [
                ("BLOGGER_BLOG_ID", self.blogger_blog_id),
                ("GOOGLE_CLIENT_ID", self.google_client_id),
                ("GOOGLE_CLIENT_SECRET", self.google_client_secret),
                ("GOOGLE_REFRESH_TOKEN", self.google_refresh_token),
            ]:
                if not val:
                    problems.append(
                        f"{name} is missing (required to publish)"
                    )

        return problems

    def warnings(self, *, need_publish: bool) -> list[str]:
        """Return non-blocking configuration warnings."""
        warns: list[str] = []

        if need_publish and not self.twilio_configured():
            warns.append(
                "Twilio WhatsApp credentials are incomplete; "
                "stories will publish without WhatsApp notifications"
            )

        if need_publish and not self.public_images_base:
            warns.append(
                "No public image base URL configured; "
                "Blogger will use inline base64 images and "
                "WhatsApp will send text-only notifications"
            )

        if self.llm_timeout < 120:
            warns.append(
                f"LLM_TIMEOUT={self.llm_timeout}s is low; "
                "Arabic article generation may time out"
            )

        return warns
