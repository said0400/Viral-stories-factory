"""History store: story IDs, dedup layers 1-2, status machine, daily quota, recovery."""
from __future__ import annotations

import re
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from . import logger
from .models import StoryCache, StoryState
from .utils import atomic_write_json, iso, normalize_url, title_similarity, utcnow


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

WHATSAPP_SKIPPED = "whatsapp_skipped"

EXPORT_PENDING = "pending"
EXPORT_READY = "ready"
EXPORT_FAILED = "failed"


PUBLISHED_STATES = {
    BLOGGER_PUBLISHED,
    FACEBOOK_READY,
    WHATSAPP_SENT,
    WHATSAPP_SKIPPED,
    COMPLETED,
    WHATSAPP_FAILED,
}


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
    WHATSAPP_SKIPPED,
}


_STORY_ID_RE = re.compile(
    r"^[A-Za-z0-9._:-]{1,128}$"
)


def is_published_row(s: StoryState) -> bool:
    if s.blogger_url and s.blogger_url.startswith(("http://", "https://")):
        return True
    return s.status in PUBLISHED_STATES and bool(s.published_at)


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

    def _load(self) -> None:
        import json

        if not self.path.exists():
            return

        try:
            raw = json.loads(
                self.path.read_text(
                    encoding="utf-8"
                ) or "[]"
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
                "history root must be a JSON array",
            )
            raise ValueError(
                "history file root must be a JSON array"
            )

        for item in raw:
            try:
                s = StoryState.model_validate(item)
                self.rows[s.story_id] = s

            except Exception:
                logger.warn(
                    "HISTORY",
                    "skipped an invalid history row",
                )

    def save(self) -> None:
        atomic_write_json(
            self.path,
            [s.model_dump() for s in self.rows.values()],
        )

    def get(
        self,
        story_id: str,
    ) -> StoryState | None:
        return self.rows.get(story_id)

    def has_url(
        self,
        normalized_url: str,
    ) -> bool:
        if not normalized_url:
            return False

        target = normalized_url.strip()

        if not target:
            return False

        for s in self.rows.values():
            if (
                s.normalized_url
                and s.normalized_url == target
            ):
                return True

            if s.original_url:
                alt = normalize_url(
                    s.original_url
                )

                if alt and alt == target:
                    return True

        return False

    def similar_title(
        self,
        source: str,
        title: str,
        threshold: float = 0.82,
    ) -> StoryState | None:
        source_key = (
            source or ""
        ).strip().casefold()

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
        for s in self.rows.values():
            if not is_published_row(s):
                continue

            candidate = (
                s.blogger_title
                or s.original_title
                or ""
            ).strip()

            if not candidate:
                continue

            if (
                title_similarity(
                    candidate,
                    title,
                ) >= threshold
            ):
                return True

        return False

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
            if s.original_title
        ]

    def recent_blogger_titles(
        self,
        n: int = 60,
    ) -> list[str]:
        rows = sorted(
            self.rows.values(),
            key=lambda s: s.updated_at,
            reverse=True,
        )[:n]

        out: list[str] = []

        for s in rows:
            t = (
                s.blogger_title
                or s.original_title
                or ""
            ).strip()

            if t:
                out.append(t)

        return out

    def recent_published_titles(
        self,
        n: int = 60,
    ) -> list[str]:
        """Titles of stories that really reached Blogger."""
        rows = sorted(
            (
                s
                for s in self.rows.values()
                if is_published_row(s)
            ),
            key=lambda s: s.updated_at,
            reverse=True,
        )[:n]

        out: list[str] = []

        for s in rows:
            for t in (
                s.original_title,
                s.blogger_title,
            ):
                t = (t or "").strip()

                if t and t not in out:
                    out.append(t)

        return out

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
            if not (
                is_published_row(s)
                and s.published_at
            ):
                continue

            try:
                dt = datetime.fromisoformat(
                    s.published_at
                )

                if dt.tzinfo is None:
                    dt = dt.replace(
                        tzinfo=self.tz
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
                and s.status != FAILED
            )
        ]

        return sorted(
            out,
            key=lambda s: s.updated_at,
        )

    def image_hashes(self) -> list[tuple[str, str]]:
        return [
            (
                s.generated_image_hash,
                s.generated_image_ahash,
            )
            for s in self.rows.values()
            if (
                s.generated_image_hash
                or s.generated_image_ahash
            )
        ]

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
        **fields: object,
    ) -> None:
        state.status = status

        if error:
            state.error = str(error)[:500]

        for k, v in fields.items():
            if hasattr(state, k):
                setattr(
                    state,
                    k,
                    v,
                )

        self.upsert(state)

    def mark_export(
        self,
        state: StoryState,
        *,
        status: str,
        bundle_path: str = "",
        error: str = "",
        save: bool = True,
    ) -> None:
        state.export_status = status

        if bundle_path:
            state.bundle_path = bundle_path

        if status == EXPORT_READY:
            state.export_created_at = iso()
            state.export_error = ""

        elif status == EXPORT_FAILED:
            state.export_error = (
                error or ""
            )[:500]

        self.upsert(
            state,
            save=save,
        )

    def _validate_story_id(
        self,
        story_id: str,
    ) -> str:
        value = (story_id or "").strip()

        if not _STORY_ID_RE.fullmatch(value):
            raise ValueError(
                "invalid story_id for cache path"
            )

        return value

    def _cache_path(
        self,
        story_id: str,
    ) -> Path:
        safe_story_id = self._validate_story_id(
            story_id
        )

        return (
            self.cache_dir
            / f"{safe_story_id}.json"
        )

    def load_cache(
        self,
        story_id: str,
    ) -> StoryCache:
        p = self._cache_path(
            story_id
        )

        if p.exists():
            try:
                return StoryCache.model_validate_json(
                    p.read_text(
                        encoding="utf-8"
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
                mode="json"
            ),
        )
