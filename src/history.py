"""History store: story IDs, dedup layers 1-2, status machine, daily quota, recovery."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import logger
from .models import StoryCache, StoryState
from .utils import atomic_write_json, iso, title_similarity, utcnow


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


# A story counts toward the daily publication quota once Blogger has
# successfully published it, even if a later promotion/notification stage
# still needs recovery.
PUBLISHED_STATES = {
    BLOGGER_PUBLISHED,
    FACEBOOK_READY,
    WHATSAPP_SENT,
    COMPLETED,
    WHATSAPP_FAILED,
}


# States from which the next scheduled run can safely continue.
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

    # ---- persistence -----------------------------------------------------
    def _load(self) -> None:
        import json

        if not self.path.exists():
            return

        try:
            raw = json.loads(
                self.path.read_text(
                    encoding="utf-8",
                )
                or "[]"
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
                "history root must be a JSON array; refusing to continue",
            )
            raise ValueError(
                "history file root must be a JSON array"
            )

        for item in raw:
            try:
                s = StoryState.model_validate(item)
                self.rows[s.story_id] = s
            except Exception:
                # Keep going on a single bad row.
                logger.warn(
                    "HISTORY",
                    "skipped an invalid history row",
                )

    def save(self) -> None:
        atomic_write_json(
            self.path,
            [
                s.model_dump()
                for s in self.rows.values()
            ],
        )

    # ---- queries ---------------------------------------------------------
    def get(
        self,
        story_id: str,
    ) -> StoryState | None:
        return self.rows.get(story_id)

    def has_url(
        self,
        normalized_url: str,
    ) -> bool:
        """Dedup layer 1."""
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
        """Dedup layer 2: same source + near-identical title."""
        source_key = (source or "").strip().casefold()

        for s in self.rows.values():
            if (
                (s.source or "").strip().casefold()
                == source_key
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
        return any(
            title_similarity(
                s.blogger_title or s.original_title,
                title,
            ) >= threshold
            for s in self.rows.values()
        )

    def recent_titles(
        self,
        n: int = 60,
    ) -> list[str]:
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
        today = (
            (now or utcnow())
            .astimezone(self.tz)
            .date()
        )

        n = 0

        for s in self.rows.values():
            if (
                s.status in PUBLISHED_STATES
                and s.published_at
            ):
                try:
                    dt = datetime.fromisoformat(
                        s.published_at,
                    )

                    if dt.tzinfo is None:
                        dt = dt.replace(
                            tzinfo=self.tz,
                        )

                    d = (
                        dt.astimezone(self.tz)
                        .date()
                    )

                except (
                    ValueError,
                    TypeError,
                ):
                    continue

                if d == today:
                    n += 1

        return n

    def resumable(
        self,
        max_attempts: int,
    ) -> list[StoryState]:
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

    def image_hashes(
        self,
    ) -> list[tuple[str, str]]:
        return [
            (
                s.generated_image_hash,
                s.generated_image_ahash,
            )
            for s in self.rows.values()
            if s.generated_image_hash
        ]

    # ---- mutations -------------------------------------------------------
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
        state.error = error[:500]

        for k, v in fields.items():
            setattr(
                state,
                k,
                v,
            )

        self.upsert(state)

    # ---- per-story cache -------------------------------------------------
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
                    p.read_text(
                        encoding="utf-8",
                    )
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
            cache.model_dump(
                mode="json",
            ),
        )
