# src/config.py
"""Central configuration: typed env loading, safe defaults, validation."""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping
from zoneinfo import ZoneInfo

DEFAULT_EXPORTS_DIR = "data/exports"
DEFAULT_CF_IMAGE_MODEL = "@cf/black-forest-labs/flux-2-dev"
DEFAULT_CONTENT_MODEL = "gemini-3.8-flash"
DEFAULT_CONTENT_FALLBACKS = "gemini-3.6-flash,gemini-3.5-flash"
DEFAULT_CINEMATIC_STYLE = (
    "premium editorial photojournalism, clear natural directional light, crisp subject detail, "
    "realistic skin and textures, balanced exposure, rich but natural color, clean contrast, "
    "sharp eyes and decisive focal point, uncluttered background, no haze, no heavy grain"
)


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
    people_image_style: str = "reference"   # real source faces are retained where a reference exists
    image_mode: str = "faithful"            # faithful | creative  (article image)
    cinematic_style: str = DEFAULT_CINEMATIC_STYLE
    facebook_image_mode: str = "original"  # original article photos only; no AI image generation
    facebook_layout: str = "auto"           # auto | insets | diptychs | triptychs
    facebook_separate_image: bool = True    # keep a dedicated original-photo Facebook composite
    image_vlm_check: bool = True
    image_required: bool = True

    # --- time
    day_timezone: str = "Africa/Casablanca"

    # --- gemini
    gemini_model: str = "gemini-3.5-flash-lite"              # triage / visual analysis / image check
    gemini_fallback_model: str = "gemini-3.5-flash"
    gemini_content_model: str = DEFAULT_CONTENT_MODEL         # article / posts / titles / fact check
    gemini_content_fallbacks: str = DEFAULT_CONTENT_FALLBACKS  # comma separated
    gemini_image_model: str = "gemini-3.1-flash-lite-image"
    gemini_api_key: str = ""
    gemini_api_key_2: str = ""
    gemini_api_key_3: str = ""
    image_api_key: str = ""
    groq_api_key: str = ""
    groq_model: str = "openai/gpt-oss-120b"
    groq_vision_model: str = "qwen/qwen3.8-27b"
    llm_timeout: int = 180

    # --- image provider
    image_provider: str = "cloudflare"      # cloudflare | gemini
    cloudflare_account_id: str = ""
    cloudflare_api_token: str = ""
    cloudflare_image_model: str = DEFAULT_CF_IMAGE_MODEL
    cloudflare_timeout: int = 300
    cloudflare_image_steps: int = 25
    image_long_side: int = 1536

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
        put("cloudflare_timeout", int(_clamp(int(self.cloudflare_timeout), 30, 900)))
        put("cloudflare_image_steps", int(_clamp(int(self.cloudflare_image_steps), 1, 50)))
        put("image_long_side", int(_clamp(int(self.image_long_side), 1024, 1920)))
        put("max_retries", int(_clamp(int(self.max_retries), 0, 10)))
        put("per_host_delay_seconds", float(_clamp(float(self.per_host_delay_seconds), 0.0, 60.0)))

        # The requested policy is to retain visible faces, not anonymize them.
        # Keep the old env field for compatibility but normalize legacy values.
        put("people_image_style", "reference")

        mode = str(self.image_mode or "").strip().lower()
        put("image_mode", mode if mode in {"faithful", "creative"} else "faithful")

        # Facebook is intentionally composed from original article pixels only.
        put("facebook_image_mode", "original")
        put("facebook_separate_image", True)

        layout = str(self.facebook_layout or "").strip().lower()
        allowed_layouts = {
            "auto", "single_hero", "inset_circle_right", "inset_circle_left",
            "inset_square_right", "inset_square_left", "diptych_split", "diptych_stack",
            "triptych", "triptych_bottom",
        }
        layout = layout.replace("-", "_").replace(" ", "_")
        layout = {
            "single": "single_hero",
            "inset": "inset_circle_right",
            "split": "diptych_split",
        }.get(layout, layout)
        put("facebook_layout", layout if layout in allowed_layouts else "auto")

        provider = str(self.image_provider or "").strip().lower()
        put("image_provider", provider if provider in {"cloudflare", "gemini"} else "cloudflare")

        defaults = {
            "data_dir": "data",
            "dry_run_dir": "data/dry_run",
            "history_file": "data/history.json",
            "cache_dir": "data/cache",
            "exports_dir": DEFAULT_EXPORTS_DIR,
            "user_agent": "ViralStoriesFactoryBot/1.0",
            "day_timezone": "Africa/Casablanca",
            "cloudflare_image_model": DEFAULT_CF_IMAGE_MODEL,
            "gemini_content_model": DEFAULT_CONTENT_MODEL,
            "cinematic_style": DEFAULT_CINEMATIC_STYLE,
        }
        for name, default in defaults.items():
            put(name, str(getattr(self, name) or "").strip() or default)

        put("gemini_content_fallbacks", str(self.gemini_content_fallbacks or "").strip())
        put("cloudflare_account_id", str(self.cloudflare_account_id or "").strip())
        put("cloudflare_api_token", str(self.cloudflare_api_token or "").strip())
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
            image_mode=_str("IMAGE_MODE", "faithful", env),
            cinematic_style=_str("CINEMATIC_STYLE", DEFAULT_CINEMATIC_STYLE, env),
            facebook_image_mode=_str("FACEBOOK_IMAGE_MODE", "original", env),
            facebook_layout=_str("FACEBOOK_LAYOUT", "auto", env),
            facebook_separate_image=_bool("FACEBOOK_SEPARATE_IMAGE", True, env),
            image_vlm_check=_bool("IMAGE_VLM_CHECK", True, env),
            image_required=_bool("IMAGE_REQUIRED", True, env),
            day_timezone=_str("DAY_TIMEZONE", "Africa/Casablanca", env),
            gemini_model=_str("GEMINI_MODEL", "gemini-3.5-flash-lite", env),
            gemini_fallback_model=_str("GEMINI_FALLBACK_MODEL", "gemini-3.5-flash", env),
            gemini_content_model=_str("GEMINI_CONTENT_MODEL", DEFAULT_CONTENT_MODEL, env),
            gemini_content_fallbacks=_str("GEMINI_CONTENT_FALLBACKS", DEFAULT_CONTENT_FALLBACKS, env),
            gemini_image_model=_str("GEMINI_IMAGE_MODEL", "gemini-3.1-flash-lite-image", env),
            gemini_api_key=_str("GEMINI_API_KEY_1", _str("GEMINI_API_KEY", "", env), env),
            gemini_api_key_2=_str("GEMINI_API_KEY_2", "", env),
            gemini_api_key_3=_str("GEMINI_API_KEY_3", "", env),
            image_api_key=_str("IMAGE_API_KEY", "", env),
            groq_api_key=_str("GROQ_API_KEY", "", env),
            groq_model=_str("GROQ_MODEL", "openai/gpt-oss-120b", env),
            groq_vision_model=_str("GROQ_VISION_MODEL", "qwen/qwen3.8-27b", env),
            llm_timeout=_int("LLM_TIMEOUT", 180, env),
            image_provider=_str("IMAGE_PROVIDER", "cloudflare", env),
            cloudflare_account_id=_str("CLOUDFLARE_ACCOUNT_ID", "", env),
            cloudflare_api_token=_str("CLOUDFLARE_API_TOKEN", "", env),
            cloudflare_image_model=_str("CLOUDFLARE_IMAGE_MODEL", DEFAULT_CF_IMAGE_MODEL, env),
            cloudflare_timeout=_int("CLOUDFLARE_TIMEOUT", 300, env),
            cloudflare_image_steps=_int("CLOUDFLARE_IMAGE_STEPS", 25, env),
            image_long_side=_int("IMAGE_LONG_SIDE", 1536, env),
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
    # Derived values
    @property
    def images_dir(self) -> Path:
        return Path(self.data_dir) / "images"

    @property
    def export_path(self) -> Path:
        """Dry-run exports never mix with live exports (unless EXPORTS_DIR was customised)."""
        if self.dry_run and self.exports_dir == DEFAULT_EXPORTS_DIR:
            return Path(self.dry_run_dir) / "exports"
        return Path(self.exports_dir)

    @property
    def needs_images(self) -> bool:
        return (not self.dry_run) or self.dry_run_generate_images

    @property
    def content_models(self) -> list[str]:
        """Model chain for the writing + fact-check steps (strongest first, de-duplicated)."""
        names = [self.gemini_content_model, *self.gemini_content_fallbacks.split(",")]
        out: list[str] = []
        retired = {"gemini-2.5-flash"}

        for name in names:
            name = name.strip()
            if name and name.lower() not in retired and name not in out:
                out.append(name)

        return out or ["gemini-3.5-flash"]

    # ------------------------------------------------------------------
    # Readiness
    def is_blogger_ready(self) -> bool:
        return bool(
            self.blogger_blog_id
            and self.google_client_id
            and self.google_client_secret
            and self.google_refresh_token
        )

    def cloudflare_ready(self) -> bool:
        return bool(self.cloudflare_account_id and self.cloudflare_api_token)

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

        if not any((self.gemini_api_key, self.gemini_api_key_2, self.gemini_api_key_3, self.groq_api_key)):
            problems.append("Configure at least one Gemini API key or GROQ_API_KEY")

        try:
            ZoneInfo(self.day_timezone)
        except Exception:
            problems.append(f"DAY_TIMEZONE is invalid: {self.day_timezone}")

        if (
            self.needs_images
            and self.image_provider == "cloudflare"
            and self.image_required
            and not self.cloudflare_ready()
        ):
            problems.append(
                "CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN are required "
                "when IMAGE_PROVIDER=cloudflare and images are generated"
            )

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

        if not self.groq_api_key:
            out.append("GROQ_API_KEY is missing; Groq-first tasks will fall back to Gemini.")
        if not any((self.gemini_api_key, self.gemini_api_key_2, self.gemini_api_key_3)):
            out.append("Gemini keys are missing; article writing will fall back to Groq.")

        if self.needs_images:
            if self.image_provider == "cloudflare" and not self.cloudflare_ready() and not self.image_required:
                out.append("Cloudflare credentials are missing; images will be skipped (IMAGE_REQUIRED=false).")

            if self.image_provider == "gemini":
                if not (self.image_api_key or self.gemini_api_key or self.gemini_api_key_2 or self.gemini_api_key_3):
                    out.append("No IMAGE_API_KEY or Gemini API key; image generation will fail.")
                else:
                    out.append(
                        "IMAGE_PROVIDER=gemini: Gemini image models may require billing "
                        "(free tier quota can be 0)."
                    )

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
            self.gemini_api_key_2,
            self.gemini_api_key_3,
            self.image_api_key,
            self.groq_api_key,
            self.cloudflare_account_id,
            self.cloudflare_api_token,
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
