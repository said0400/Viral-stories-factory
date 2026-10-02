"""History store: story IDs, dedup layers 1-2, status machine, daily quota, recovery."""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import logger
from .models import StoryCache, StoryState
from .utils import atomic_write_json, iso, normalize_title, title_similarity, utcnow


# ---------------------------------------------------------------------------
# Status machine
# ---------------------------------------------------------------------------

DISCOVERED, SELECTED, GENERATED = (
    "discovered",
    "selected",
    "generated",
)

IMAGE_GENERATED, BLOGGER_PUBLISHED, FACEBOOK_READY = (
    "image_generated",
    "blogger_published",
    "facebook_ready",
)

WHATSAPP_SENT, COMPLETED, FAILED = (
    "whatsapp_sent",
    "completed",
    "failed",
)

IMAGE_FAILED, BLOGGER_FAILED, WHATSAPP_FAILED = (
    "image_failed",
    "blogger_failed",
    "whatsapp_failed",
)


# A story counts toward the daily publishing quota once Blogger has
# successfully published it. WhatsApp failure must NOT cause the story
# to become eligible for another Blogger post.
PUBLISHED_STATES = {
    BLOGGER_PUBLISHED,
    FACEBOOK_READY,
    WHATSAPP_SENT,
    COMPLETED,
    WHATSAPP_FAILED,
}


# States that may be resumed by a later run instead of starting the story
# from scratch. In particular, WHATSAPP_FAILED is intentionally resumable
# so a later run can send WhatsApp without creating another Blogger post.
RESUMABLE_STATES = {
    SELECTED,
    GENERATED,
    IMAGE_GENERATED,
    IMAGE_FAILED,
    BLOGGER_FAILED,
    BLOGGER_PUBLISHED,
    FACEBOOK_READY,
    WHATSAPP_FAILED,
    WHATSAPP_SENT,
}


class History:
    def __init__(
        self,
        path: Path,
        cache_dir: Path,
        tz: str = "UTC",
    ) -> None:
        self.path = Path(path)
        self.cache_dir = Path(cache_dir)
        self.tz = ZoneInfo(tz)
        self.rows: dict[str, StoryState] = {}
        self._load()

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------

    def _load(self) -> None:
        if not self.path.exists():
            return

        try:
            raw = json.loads(
                self.path.read_text(encoding="utf-8") or "[]"
            )
        except json.JSONDecodeError:
            logger.error(
                "HISTORY",
                "history file is corrupt; refusing to continue",
            )
            raise

        if not isinstance(raw, list):
            logger.error(
                "HISTORY",
                "history file has invalid root structure; expected a list",
            )
            raise ValueError("history file must contain a JSON list")

        for item in raw:
            try:
                s = StoryState.model_validate(item)
                self.rows[s.story_id] = s
            except Exception:
                # Keep going when one historical row is malformed.
                # A single damaged row should not make all valid history
                # unusable.
                logger.warn(
                    "HISTORY",
                    "skipped an invalid history row",
                )

    def save(self) -> None:
        atomic_write_json(
            self.path,
            [
                s.model_dump(mode="json")
                for s in self.rows.values()
            ],
        )

    # -----------------------------------------------------------------------
    # Queries
    # -----------------------------------------------------------------------

    def get(self, story_id: str) -> StoryState | None:
        return self.rows.get(story_id)

    def has_url(self, normalized_url: str) -> bool:
        """Dedup layer 1: exact normalized source URL."""
        if not normalized_url:
            return False

        return any(
            s.normalized_url == normalized_url
            for s in self.rows.values()
        )

    def similar_title(
        self,
        source: str,
        title: str,
        threshold: float = 0.82,
    ) -> StoryState | None:
        """Dedup layer 2: same source + near-identical original title."""
        if not source or not title:
            return None

        for s in self.rows.values():
            if (
                s.source == source
                and title_similarity(
                    s.original_title,
                    title,
                ) >= threshold
            ):
                return s

        return None

    def any_similar_published_title(
        self,
        title: str,
        threshold: float = 0.85,
    ) -> bool:
        """
        Check whether a generated/published title is too similar to a
        previous title.

        title_similarity() already normalizes both arguments, so the
        original title is passed directly.
        """
        if not title:
            return False

        return any(
            title_similarity(
                s.blogger_title or s.original_title,
                title,
            ) >= threshold
            for s in self.rows.values()
        )

    def recent_titles(self, n: int = 60) -> list[str]:
        n = max(0, int(n))

        rows = sorted(
            self.rows.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )[:n]

        return [
            s.original_title
            for s in rows
        ]

    def published_today(
        self,
        now: datetime | None = None,
    ) -> int:
        """
        Count stories that reached a published state today in the configured
        timezone.

        WHATSAPP_FAILED is intentionally counted because Blogger has already
        published the article; only the WhatsApp delivery step failed.
        """
        current = now or utcnow()

        if current.tzinfo is None:
            current = current.replace(tzinfo=self.tz)

        today = current.astimezone(self.tz).date()

        count = 0

        for s in self.rows.values():
            if (
                s.status not in PUBLISHED_STATES
                or not s.published_at
            ):
                continue

            try:
                published = datetime.fromisoformat(
                    s.published_at
                )

                if published.tzinfo is None:
                    published = published.replace(
                        tzinfo=self.tz
                    )

                published_date = (
                    published
                    .astimezone(self.tz)
                    .date()
                )

            except (TypeError, ValueError, OverflowError):
                continue

            if published_date == today:
                count += 1

        return count

    def resumable(
        self,
        max_attempts: int,
    ) -> list[StoryState]:
        """
        Return unfinished stories that can be resumed.

        Sorting by updated_at makes the oldest pending work resume first.
        """
        max_attempts = max(0, int(max_attempts))

        out = [
            s
            for s in self.rows.values()
            if (
                s.status in RESUMABLE_STATES
                and s.attempts < max_attempts
            )
        ]

        return sorted(
            out,
            key=lambda s: s.updated_at,
        )

    def image_hashes(self) -> list[tuple[str, str]]:
        """
        Return generated image hashes for duplicate-image detection.

        The first value is the cryptographic hash and the second is the
        perceptual average hash.
        """
        return [
            (
                s.generated_image_hash,
                s.generated_image_ahash,
            )
            for s in self.rows.values()
            if s.generated_image_hash
        ]

    # -----------------------------------------------------------------------
    # Mutations
    # -----------------------------------------------------------------------

    def upsert(
        self,
        state: StoryState,
        save: bool = True,
    ) -> StoryState:
        state.updated_at = iso()

        self.rows[state.story_id] = state

        if save:
            self.save()

        return state

    def set_status(
        self,
        state: StoryState,
        status: str,
        *,
        error: str = "",
        **fields: str,
    ) -> None:
        state.status = status
        state.error = (error or "")[:500]

        for k, v in fields.items():
            setattr(state, k, v)

        self.upsert(state)

    # -----------------------------------------------------------------------
    # Per-story cache
    # -----------------------------------------------------------------------

    def _cache_path(
        self,
        story_id: str,
    ) -> Path:
        return self.cache_dir / f"{story_id}.json"

    def load_cache(
        self,
        story_id: str,
    ) -> StoryCache:
        p = self._cache_path(story_id)

        if p.exists():
            try:
                return StoryCache.model_validate_json(
                    p.read_text(encoding="utf-8")
                )
            except Exception:
                logger.warn(
                    "HISTORY",
                    f"cache for {story_id} unreadable; ignoring",
                )

        return StoryCache()

    def save_cache(
        self,
        story_id: str,
        cache: StoryCache,
    ) -> None:
        atomic_write_json(
            self._cache_path(story_id),
            cache.model_dump(mode="json"),
        )
