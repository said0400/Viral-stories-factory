"""Orchestration only. Each stage lives in its own module."""
from __future__ import annotations

import argparse
import base64
import dataclasses
import io
import sys
from datetime import timedelta
from pathlib import Path

from PIL import Image

from . import content as editorial
from . import extractor, history as H, logger, sources as src_mod, visual_analyzer
from .blogger import BloggerClient, BloggerError
from .config import Settings
from .facebook import FacebookError, build_package
from .gemini_client import GeminiClient, GeminiError, ImageGenError
from .image_generator import ImageGenerator
from .image_publisher import publish_images
from .logger import error, log, setup_logging, warn
from .models import SourceArticle, StoryCache, StoryState
from .twilio_whatsapp import WhatsAppClient, WhatsAppError
from .utils import (
    PoliteFetcher,
    atomic_write_json,
    git_commit_and_push,
    iso,
    make_story_id,
    utcnow,
)


class StoryFailed(Exception):
    def __init__(self, stage: str, msg: str) -> None:
        super().__init__(msg)
        self.stage = stage


def data_uri(path: str) -> str:
    """Last-resort inline image for Blogger when no public URL exists."""
    img = Image.open(path).convert("RGB")
    img.thumbnail((1000, 1000))

    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=75)

    return (
        "data:image/jpeg;base64,"
        + base64.b64encode(buf.getvalue()).decode()
    )


class Factory:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.dry = cfg.dry_run

        self.work = (
            dataclasses.replace(
                cfg,
                data_dir=cfg.dry_run_dir,
            )
            if self.dry
            else cfg
        )

        self.hist = H.History(
            cfg.history_file,
            cfg.cache_dir,
            cfg.day_timezone,
        )

        self.gem = GeminiClient(cfg)

        self.fetcher = PoliteFetcher(
            cfg.user_agent,
            cfg.request_timeout,
            cfg.per_host_delay,
        )

        self.imgs = ImageGenerator(
            self.work,
            self.gem,
        )

        self.blogger = (
            None
            if self.dry
            else BloggerClient(cfg)
        )

        self.whatsapp: WhatsAppClient | None = None

        self.completed = 0
        self.partial = 0
        self.failed = 0

    # ===================================================================
    # run
    # ===================================================================

    def run(self) -> int:
        log(
            "RUN",
            f"mode={'DRY RUN' if self.dry else 'LIVE'} | "
            f"target/day={self.cfg.target_daily_stories}",
        )

        published = self.hist.published_today()

        base = (
            self.cfg.number_of_stories
            or self.cfg.max_stories_per_run
        )

        quota = (
            base
            if (
                self.cfg.manual_override
                or self.dry
            )
            else min(
                base,
                max(
                    0,
                    self.cfg.target_daily_stories - published,
                ),
            )
        )

        log(
            "RUN",
            f"published today={published}; quota for this run={quota}",
        )

        if quota <= 0:
            log(
                "RUN",
                "daily target reached; nothing to do",
            )
            return 0

        done = 0

        # ---------------------------------------------------------------
        # Recovery
        # ---------------------------------------------------------------
        if not self.dry:
            for st in self.hist.resumable(
                self.cfg.max_story_attempts
            ):
                if done >= quota:
                    break

                log(
                    "RECOVERY",
                    f"resuming {st.story_id} "
                    f"from status={st.status}",
                )

                done += self._guarded(
                    st,
                    resume=True,
                )

        # ---------------------------------------------------------------
        # New stories
        # ---------------------------------------------------------------
        if done < quota:
            for art in self._discover_and_select(
                quota - done
            ):
                st = StoryState(
                    story_id=make_story_id(
                        art.source_name,
                        art.normalized_url,
                    ),
                    source=art.source_name,
                    original_url=art.original_url,
                    normalized_url=art.normalized_url,
                    original_title=art.original_title,
                    discovered_at=iso(art.discovered_at),
                    selected_at=iso(),
                    status=H.SELECTED,
                )

                if not self.dry:
                    self.hist.upsert(st)

                    self.hist.save_cache(
                        st.story_id,
                        StoryCache(article=art),
                    )

                done += self._guarded(
                    st,
                    resume=False,
                    article=art,
                )

        log(
            "SUMMARY",
            f"completed={self.completed} "
            f"partial(published, needs follow-up)={self.partial} "
            f"failed/pending={self.failed} "
            f"(dry_run={self.dry})",
        )

        return 0

    # ===================================================================
    # discovery
    # ===================================================================

    def _discover_and_select(
        self,
        quota: int,
    ) -> list[SourceArticle]:
        cands: list[SourceArticle] = []

        for sd in src_mod.load_sources(
            self.cfg.sources_override_file
        ):
            if not sd.enabled:
                continue

            try:
                cands += src_mod.discover(
                    sd,
                    self.fetcher,
                )
            except Exception as exc:
                # One broken source must never stop the entire run.
                warn(
                    "DISCOVERY",
                    f"{sd.name} failed: {type(exc).__name__}",
                )

        log(
            "DISCOVERY",
            f"Found {len(cands)} candidate articles",
        )

        # ---------------------------------------------------------------
        # Layer 1 + 2 deduplication
        # ---------------------------------------------------------------
        seen: set[str] = set()
        fresh: list[SourceArticle] = []

        for c in cands:
            if not c.normalized_url:
                continue

            if c.normalized_url in seen:
                continue

            if self.hist.has_url(
                c.normalized_url
            ):
                continue

            if self.hist.similar_title(
                c.source_name,
                c.original_title,
            ):
                continue

            seen.add(c.normalized_url)
            fresh.append(c)

        log(
            "DEDUP",
            f"Removed {len(cands) - len(fresh)} "
            "duplicates (URL / source+title)",
        )

        if not fresh:
            return []

        # ---------------------------------------------------------------
        # Round-robin across sources, newest first.
        # ---------------------------------------------------------------
        by_src: dict[
            str,
            list[SourceArticle],
        ] = {}

        for c in sorted(
            fresh,
            key=lambda a: (
                a.publication_date
                or a.discovered_at
            ),
            reverse=True,
        ):
            by_src.setdefault(
                c.source_name,
                [],
            ).append(c)

        batch: list[SourceArticle] = []

        while (
            len(batch)
            < self.cfg.max_candidates_for_triage
            and any(by_src.values())
        ):
            for k in list(by_src):
                if (
                    by_src[k]
                    and len(batch)
                    < self.cfg.max_candidates_for_triage
                ):
                    batch.append(
                        by_src[k].pop(0)
                    )

        if not batch:
            return []

        # ---------------------------------------------------------------
        # Gemini triage
        # ---------------------------------------------------------------
        try:
            items = editorial.triage(
                self.gem,
                batch,
                self.hist.recent_titles(),
            )
        except (
            GeminiError,
            ValueError,
            IndexError,
            TypeError,
        ) as exc:
            error(
                "SELECTION",
                f"triage failed: {exc}",
            )
            return []

        cutoff = (
            utcnow()
            - timedelta(
                hours=self.cfg.max_source_age_hours
            )
        )

        rejected: dict[str, int] = {}
        picked: list[
            tuple[float, SourceArticle]
        ] = []

        seen_dupe: set[int] = set()

        # ---------------------------------------------------------------
        # Validate Gemini triage indexes before indexing batch.
        # ---------------------------------------------------------------
        valid_items = []

        for it in items:
            if (
                it.index < 0
                or it.index >= len(batch)
            ):
                rejected["invalid triage index"] = (
                    rejected.get(
                        "invalid triage index",
                        0,
                    )
                    + 1
                )
                continue

            if (
                it.duplicate_of < -1
                or it.duplicate_of >= len(batch)
            ):
                # Invalid duplicate reference is treated as no duplicate.
                it.duplicate_of = -1

            valid_items.append(it)

        # Higher-ranked candidates are processed first so that when Gemini
        # marks two stories as duplicates, the better candidate wins.
        for it in sorted(
            valid_items,
            key=editorial.rank_score,
            reverse=True,
        ):
            c = batch[it.index]

            reason = ""

            if not it.suitable:
                reason = "unsuitable"

            elif it.already_published:
                reason = "same event already published"

            elif (
                it.duplicate_of >= 0
                and it.duplicate_of in seen_dupe
            ):
                reason = "same event as a better candidate"

            elif (
                it.viral_score
                < self.cfg.min_viral_score
            ):
                reason = "below quality threshold"

            elif (
                c.publication_date
                and c.publication_date < cutoff
            ):
                if it.is_evergreen:
                    c.age_exception = (
                        "older than MAX_SOURCE_AGE_HOURS "
                        "but evergreen"
                    )

                    log(
                        "SELECTION",
                        f"age exception: "
                        f"{c.original_title[:60]} "
                        f"({c.age_exception})",
                    )
                else:
                    reason = "too old"

            if reason:
                rejected[reason] = (
                    rejected.get(reason, 0)
                    + 1
                )
                continue

            seen_dupe.add(it.index)

            if it.duplicate_of >= 0:
                seen_dupe.add(
                    it.duplicate_of
                )

            picked.append(
                (
                    editorial.rank_score(it),
                    c,
                )
            )

        # ---------------------------------------------------------------
        # Full article extraction + validation
        # ---------------------------------------------------------------
        valid: list[SourceArticle] = []

        for _, c in picked[: quota * 3]:
            try:
                full = extractor.extract_article(
                    c,
                    self.fetcher,
                )
            except Exception as exc:
                rejected["extraction failed"] = (
                    rejected.get(
                        "extraction failed",
                        0,
                    )
                    + 1
                )

                warn(
                    "EXTRACTION",
                    f"{c.original_title[:60]} "
                    f"failed: {type(exc).__name__}",
                )
                continue

            if not full:
                rejected["extraction failed"] = (
                    rejected.get(
                        "extraction failed",
                        0,
                    )
                    + 1
                )
                continue

            full.age_exception = c.age_exception

            ok, why = extractor.is_valid(
                full,
                self.cfg.min_article_chars,
            )

            if not ok:
                rejected[why] = (
                    rejected.get(why, 0)
                    + 1
                )
                continue

            # Canonical URL may reveal a duplicate that was not visible
            # before full extraction.
            if self.hist.has_url(
                full.normalized_url
            ):
                rejected[
                    "duplicate canonical url"
                ] = (
                    rejected.get(
                        "duplicate canonical url",
                        0,
                    )
                    + 1
                )
                continue

            valid.append(full)

            if len(valid) >= quota:
                break

        shortage = (
            ""
            if len(valid) >= quota
            else (
                f"only {len(valid)} story(ies) met "
                "the quality bar; quality is never "
                "lowered to fill the quota"
            )
        )

        log(
            "SELECTION",
            f"available_valid_stories={len(picked)} "
            f"selected_stories={len(valid)} "
            f"rejected_stories={rejected} "
            f"reason_for_shortage={shortage or 'none'}",
        )

        return valid

    # ===================================================================
    # guarded story
    # ===================================================================

    def _guarded(
        self,
        st: StoryState,
        *,
        resume: bool,
        article: SourceArticle | None = None,
    ) -> int:
        tag = f"STORY {st.story_id[:8]}"

        try:
            return int(
                self._process(
                    st,
                    article,
                )
            )

        except StoryFailed as exc:
            error(
                tag,
                f"{exc.stage}: {exc}",
            )

            self._record_failure(
                st,
                exc.stage,
                str(exc),
            )

        except Exception as exc:
            # Never let one story stop the rest of the run.
            error(
                tag,
                f"unexpected {type(exc).__name__}",
            )

            self._record_failure(
                st,
                "unexpected",
                type(exc).__name__,
            )

        self.failed += 1

        return 0

    def _record_failure(
        self,
        st: StoryState,
        stage: str,
        msg: str,
    ) -> None:
        if self.dry:
            return

        st.attempts += 1
        st.failed_stage = stage
        st.error = msg[:300]

        if st.attempts >= self.cfg.max_story_attempts:
            st.status = H.FAILED

        self.hist.upsert(st)

    # ===================================================================
    # pipeline
    # ===================================================================

    def _process(
        self,
        st: StoryState,
        article: SourceArticle | None,
    ) -> bool:
        tag = f"STORY {st.story_id[:8]}"

        cache = (
            StoryCache(article=article)
            if self.dry
            else self.hist.load_cache(
                st.story_id
            )
        )

        article = article or cache.article

        if not article:
            raise StoryFailed(
                "recovery",
                "no cached article to resume from",
            )

        # ---------------------------------------------------------------
        # 1. Original Arabic content + fact check
        # ---------------------------------------------------------------
        if not cache.content:
            log(
                tag,
                "Analyzing / generating content...",
            )

            try:
                cache.content = editorial.generate_verified(
                    self.gem,
                    article,
                    self.hist.recent_titles(),
                    lambda t: (
                        self.hist.any_similar_published_title(t)
                    ),
                )

            except (
                GeminiError,
                ValueError,
            ) as exc:
                raise StoryFailed(
                    "generate",
                    str(exc),
                ) from exc

            st.generated_at = iso()
            st.blogger_title = (
                cache.content.blogger_title
            )

            if not self.dry:
                self.hist.set_status(
                    st,
                    H.GENERATED,
                )

                self.hist.save_cache(
                    st.story_id,
                    cache,
                )

        content = cache.content

        # ---------------------------------------------------------------
        # 2. Visual analysis + AI image
        # ---------------------------------------------------------------
        need_image = (
            not self.dry
            or self.cfg.dry_run_generate_images
        )

        if (
            need_image
            and not (
                cache.image
                and Path(
                    cache.image.path
                ).exists()
            )
        ):
            try:
                if not cache.visual:
                    ref = (
                        visual_analyzer.acquire_source_image(
                            article,
                            self.fetcher,
                        )
                    )

                    log(
                        "VISUAL",
                        (
                            "Reference image found"
                            if ref
                            else "No reference image available"
                        ),
                    )

                    cache.visual = visual_analyzer.analyze(
                        self.gem,
                        article,
                        (
                            (ref[0], ref[1])
                            if ref
                            else None
                        ),
                    )

                    log(
                        "VISUAL",
                        f"Subject type: "
                        f"{cache.visual.subject_type}; "
                        f"people="
                        f"{cache.visual.contains_real_people}",
                    )

                    self._ref = ref

                else:
                    self._ref = (
                        visual_analyzer.acquire_source_image(
                            article,
                            self.fetcher,
                        )
                        if cache.visual.reference_required
                        else None
                    )

                ref = self._ref

                cache.image = self.imgs.generate(
                    story_id=st.story_id,
                    article=article,
                    v=cache.visual,
                    title=content.blogger_title,
                    article_scene=content.article_scene_idea,
                    facebook_scene=content.facebook_scene_idea,
                    source_ref=(
                        (ref[0], ref[1])
                        if ref
                        else None
                    ),
                    source_url=(
                        ref[2]
                        if ref
                        else ""
                    ),
                    source_sha=(
                        ref[3]
                        if ref
                        else ""
                    ),
                    source_ahash=(
                        ref[4]
                        if ref
                        else ""
                    ),
                    known=self.hist.image_hashes(),
                )

            except (
                ImageGenError,
                GeminiError,
            ) as exc:
                st.image_status = "failed"

                if not self.dry:
                    self.hist.set_status(
                        st,
                        H.IMAGE_FAILED,
                        error=str(exc),
                    )

                if self.cfg.image_required:
                    raise StoryFailed(
                        "image",
                        str(exc),
                    ) from exc

                warn(
                    tag,
                    "image failed; continuing without "
                    "image (IMAGE_REQUIRED=false)",
                )

            else:
                st.image_status = "generated"
                st.image_generated_at = iso()

                st.subject_type = (
                    cache.visual.subject_type
                )

                st.identity_confidence = (
                    cache.image.identity_confidence
                )

                st.source_image_url = (
                    cache.image.source_image_url
                )

                st.source_image_hash = (
                    cache.image.source_image_hash
                )

                st.generated_image_path = (
                    cache.image.path
                )

                st.generated_image_hash = (
                    cache.image.generated_hash
                )

                st.generated_image_ahash = (
                    cache.image.generated_ahash
                )

                if not self.dry:
                    self.hist.set_status(
                        st,
                        H.IMAGE_GENERATED,
                    )

                    self.hist.save_cache(
                        st.story_id,
                        cache,
                    )

        # ---------------------------------------------------------------
        # Dry run ends here.
        # ---------------------------------------------------------------
        if self.dry:
            self._write_dry(
                st,
                cache,
            )

            log(
                tag,
                "DRY RUN complete "
                "(nothing published, not marked completed)",
            )

            return False

        # ---------------------------------------------------------------
        # 3. Blogger
        #
        # Important recovery rule:
        # If blogger_url already exists, Blogger is NEVER called again.
        # This is what makes whatsapp_failed recovery idempotent.
        # ---------------------------------------------------------------
        if not st.blogger_url:
            self._publish_blogger(
                st,
                cache,
                article,
            )

        # ---------------------------------------------------------------
        # 4. Facebook package
        #
        # It can only be generated after a REAL Blogger URL exists.
        # ---------------------------------------------------------------
        if (
            st.facebook_status != "ready"
            or not cache.facebook
        ):
            self._facebook(
                st,
                cache,
                article,
            )

        # ---------------------------------------------------------------
        # 5. WhatsApp
        # ---------------------------------------------------------------
        if st.whatsapp_status != "sent":
            self._whatsapp(
                st,
                cache,
                article,
            )

        # ---------------------------------------------------------------
        # 6. Final validation
        # ---------------------------------------------------------------
        problems = self._final_checks(
            st,
            cache,
        )

        if problems:
            warn(
                tag,
                "NOT completed: "
                + "; ".join(problems),
            )

            self.partial += 1

            return bool(st.blogger_url)

        self.hist.set_status(
            st,
            H.COMPLETED,
        )

        log(
            tag,
            "COMPLETED",
        )

        self.completed += 1

        if self.cfg.git_push_enabled:
            git_commit_and_push(
                [
                    self.cfg.history_file,
                    self.cfg.cache_dir,
                ],
                f"history: {st.story_id}",
            )

        return True

    # ===================================================================
    # stages
    # ===================================================================

    def _publish_blogger(
        self,
        st: StoryState,
        cache: StoryCache,
        art: SourceArticle,
    ) -> None:
        tag = f"STORY {st.story_id[:8]}"

        content = cache.content
        img = cache.image

        if not content:
            raise StoryFailed(
                "validate",
                "content missing",
            )

        # ---------------------------------------------------------------
        # Validation before publishing
        # ---------------------------------------------------------------
        if not content.blogger_title.strip():
            raise StoryFailed(
                "validate",
                "title invalid",
            )

        if len(content.blogger_html) < 200:
            raise StoryFailed(
                "validate",
                "content invalid",
            )

        if (
            self.cfg.image_required
            and not (
                img
                and Path(img.path).exists()
            )
        ):
            raise StoryFailed(
                "validate",
                "valid image missing",
            )

        image_url = ""

        if (
            img
            and Path(img.path).exists()
        ):
            paths = [
                p
                for p in (
                    img.path,
                    img.facebook_path,
                )
                if p
            ]

            urls = publish_images(
                self.cfg,
                paths,
                st.story_id,
            )

            img.public_url = urls.get(
                img.path,
                "",
            )

            img.facebook_public_url = urls.get(
                img.facebook_path,
                "",
            )

            image_url = (
                img.public_url
                or data_uri(img.path)
            )

            if not img.public_url:
                warn(
                    "IMAGE",
                    "using inline image in Blogger "
                    "(no public URL)",
                )

        html = editorial.render_blogger_html(
            content,
            art,
            image_url,
            st.story_id,
        )

        log(
            "BLOGGER",
            "Publishing...",
        )

        if self.blogger is None:
            raise StoryFailed(
                "blogger",
                "Blogger client is unavailable in live mode",
            )

        try:
            res = self.blogger.publish(
                story_id=st.story_id,
                title=content.blogger_title,
                html=html,
                labels=content.labels,
                description=content.seo_description,
            )

        except BloggerError as exc:
            self.hist.set_status(
                st,
                H.BLOGGER_FAILED,
                error=str(exc),
            )

            raise StoryFailed(
                "blogger",
                str(exc),
            ) from exc

        st.blogger_post_id = res.post_id
        st.blogger_url = res.url
        st.published_at = res.published_at

        self.hist.set_status(
            st,
            H.BLOGGER_PUBLISHED,
        )

        self.hist.save_cache(
            st.story_id,
            cache,
        )

        log(
            "BLOGGER",
            f"Published successfully\n"
            f"[BLOGGER] URL: {res.url}",
        )

    def _facebook(
        self,
        st: StoryState,
        cache: StoryCache,
        art: SourceArticle,
    ) -> None:
        if not st.blogger_url:
            st.facebook_status = "failed"

            self.hist.set_status(
                st,
                st.status,
                error="facebook: missing Blogger URL",
            )

            warn(
                "FACEBOOK",
                "package skipped: Blogger URL missing",
            )

            return

        if not cache.content:
            st.facebook_status = "failed"

            self.hist.set_status(
                st,
                st.status,
                error="facebook: content missing",
            )

            warn(
                "FACEBOOK",
                "package skipped: content missing",
            )

            return

        try:
            img_path = (
                (
                    cache.image.facebook_path
                    or cache.image.path
                )
                if cache.image
                else ""
            )

            cache.facebook = build_package(
                cache.content,
                st.blogger_url,
                art.original_url,
                img_path,
            )

        except FacebookError as exc:
            st.facebook_status = "failed"

            # Preserve the current publishing state so the story remains
            # resumable. Blogger URL is intentionally kept.
            self.hist.set_status(
                st,
                st.status,
                error=f"facebook: {exc}",
            )

            warn(
                "FACEBOOK",
                f"package failed: {exc} "
                "(Blogger URL kept)",
            )

            return

        st.facebook_status = "ready"

        self.hist.set_status(
            st,
            H.FACEBOOK_READY,
        )

        self.hist.save_cache(
            st.story_id,
            cache,
        )

        log(
            "FACEBOOK",
            "Promotion package ready",
        )

    def _whatsapp(
        self,
        st: StoryState,
        cache: StoryCache,
        art: SourceArticle,
    ) -> None:
        if not cache.facebook:
            return

        if not st.blogger_url:
            warn(
                "WHATSAPP",
                "skipped: Blogger URL missing",
            )
            return

        try:
            if self.whatsapp is None:
                self.whatsapp = WhatsAppClient(
                    self.cfg
                )

            img = cache.image
            url = ""

            if img:
                paths = [
                    p
                    for p in (
                        img.facebook_path
                        or img.path,
                    )
                    if p
                ]

                if paths:
                    urls = publish_images(
                        self.cfg,
                        paths,
                        st.story_id,
                        wait_seconds=60,
                    )

                    url = next(
                        iter(urls.values()),
                        "",
                    )

            res = self.whatsapp.send_package(
                title=cache.content.blogger_title,
                blogger_url=st.blogger_url,
                source_name=art.source_name,
                source_url=art.original_url,
                fb=cache.facebook,
                image_public_url=url,
                status=(
                    "blogger_published / "
                    "facebook_ready"
                ),
            )

        except WhatsAppError as exc:
            st.whatsapp_status = "failed"

            self.hist.set_status(
                st,
                H.WHATSAPP_FAILED,
                error=str(exc),
            )

            warn(
                "WHATSAPP",
                f"failed: {exc} "
                "(Blogger post kept; will resend next run)",
            )

            return

        st.whatsapp_status = "sent"

        st.whatsapp_message_id = (
            res.message_ids[0]
            if res.message_ids
            else ""
        )

        self.hist.set_status(
            st,
            H.WHATSAPP_SENT,
        )

        log(
            "WHATSAPP",
            "Notification sent",
        )

    def _final_checks(
        self,
        st: StoryState,
        cache: StoryCache,
    ) -> list[str]:
        problems: list[str] = []

        if (
            not st.blogger_post_id
            or not st.blogger_url
            or not st.blogger_url.startswith(
                "https://"
            )
        ):
            problems.append(
                "blogger_url_invalid"
            )

        if not cache.content:
            problems.append(
                "content_invalid"
            )

        if (
            self.cfg.image_required
            and st.image_status != "generated"
        ):
            problems.append(
                "image_invalid"
            )

        if not cache.facebook:
            problems.append(
                "facebook_package_invalid"
            )

        if st.whatsapp_status != "sent":
            problems.append(
                "whatsapp_not_sent"
            )

        return problems

    def _write_dry(
        self,
        st: StoryState,
        cache: StoryCache,
    ) -> None:
        out = (
            self.cfg.dry_run_dir
            / f"{st.story_id}.json"
        )

        atomic_write_json(
            out,
            cache.model_dump(mode="json"),
        )

        log(
            "DRY-RUN",
            f"saved preview to {out}",
        )


# =======================================================================
# CLI
# =======================================================================

def parse_args(
    argv: list[str] | None = None,
) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Viral Stories Factory"
    )

    ap.add_argument(
        "--dry-run",
        dest="dry_run",
        action="store_true",
        default=None,
    )

    ap.add_argument(
        "--live",
        dest="dry_run",
        action="store_false",
    )

    ap.add_argument(
        "--stories",
        type=int,
        default=None,
        help="number of stories this run",
    )

    ap.add_argument(
        "--override",
        action="store_true",
        help="ignore the daily target",
    )

    return ap.parse_args(argv)


def main(
    argv: list[str] | None = None,
) -> int:
    args = parse_args(argv)

    cfg = Settings.from_env()

    if args.dry_run is not None:
        cfg = dataclasses.replace(
            cfg,
            dry_run=args.dry_run,
        )

    if args.stories is not None:
        if args.stories < 1:
            error(
                "CONFIG",
                "--stories must be >= 1",
            )
            return 2

        cfg = dataclasses.replace(
            cfg,
            number_of_stories=args.stories,
        )

    if args.override:
        cfg = dataclasses.replace(
            cfg,
            manual_override=True,
        )

    setup_logging(
        cfg.secret_values()
    )

    problems = cfg.validate(
        need_publish=not cfg.dry_run
    )

    if problems:
        for p in problems:
            error(
                "CONFIG",
                p,
            )

        return 2

    return Factory(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
