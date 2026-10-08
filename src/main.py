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
from . import extractor, history as H, sources as src_mod, visual_analyzer
from .blogger import BloggerClient, BloggerError
from .config import Settings
from .exporter import export_bundle
from .facebook import FacebookError, build_content_package, build_package
from .gemini_client import GeminiClient, GeminiError, ImageGenError
from .groq_client import GroqClient
from .image_generator import ImageGenerator
from .image_publisher import publish_images
from .llm_router import LLMRouter
from .logger import error, log, setup_logging, warn
from .models import SourceArticle, StoryCache, StoryState
from .twilio_whatsapp import WhatsAppClient, WhatsAppError
from .utils import PoliteFetcher, atomic_write_json, git_commit_and_push, iso, make_story_id, utcnow


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
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def _is_file(path: str | None) -> bool:
    """Path('') is '.', which exists - so empty strings must be rejected explicitly."""
    return bool(path) and Path(str(path)).is_file()


class Factory:
    def __init__(self, cfg: Settings) -> None:
        self.cfg = cfg
        self.dry = cfg.dry_run
        self.work = dataclasses.replace(cfg, data_dir=cfg.dry_run_dir) if self.dry else cfg
        self.hist = H.History(cfg.history_file, cfg.cache_dir, cfg.day_timezone)
        self.gem = GeminiClient(cfg)
        self.groq = GroqClient(cfg)
        self.llm = LLMRouter(self.gem, self.groq)
        self.fetcher = PoliteFetcher(cfg.user_agent, cfg.request_timeout, cfg.per_host_delay_seconds)
        self.imgs = ImageGenerator(self.work, self.llm, fetcher=self.fetcher)
        self.blogger = None if self.dry else BloggerClient(cfg)
        self.whatsapp: WhatsAppClient | None = None
        self.completed = self.partial = self.failed = 0

    # =================================================================== helpers
    def _push_state(self, message: str) -> None:
        """Persist history/cache so a partially processed story survives a CI runner."""
        if self.dry or not self.cfg.git_push_enabled:
            return

        try:
            git_commit_and_push([self.cfg.history_file, self.cfg.cache_dir], message)
        except Exception as exc:
            warn("GIT", f"state push failed ({type(exc).__name__})")

    def _export(self, st: StoryState, cache: StoryCache, art: SourceArticle) -> None:
        """Incrementally generate offline downloadable ZIP bundles."""
        if not self.cfg.export_enabled:
            return

        try:
            z = export_bundle(self.work.export_path, st, cache, art)

            if self.dry:
                return  # never touch history.json during a dry run

            if z:
                self.hist.mark_export(st, status=H.EXPORT_READY, bundle_path=str(z), save=True)
                log("EXPORT", f"bundle ready: {z.name}")
            else:
                self.hist.mark_export(st, status=H.EXPORT_FAILED, error="export_bundle returned None", save=True)

        except Exception as exc:
            warn("EXPORT", f"bundle failed ({type(exc).__name__}: {exc})")

            if not self.dry:
                self.hist.mark_export(st, status=H.EXPORT_FAILED, error=str(exc), save=True)

    # =================================================================== run
    def run(self) -> int:
        log("RUN", f"mode={'DRY RUN' if self.dry else 'LIVE'} | target/day={self.cfg.target_daily_stories}")

        if not self.dry:
            migrated = self.hist.sanitize_cached_sources()
            if migrated:
                log("HISTORY", f"redacted source text from {migrated} legacy cache file(s)")

        base = self.cfg.number_of_stories if self.cfg.number_of_stories > 0 else self.cfg.max_stories_per_run
        recovery_attempts = 0

        # ---------------------------------------------------------- recovery
        if not self.dry:
            for st in self.hist.resumable(self.cfg.max_story_attempts):
                if recovery_attempts >= base:
                    break

                log("RECOVERY", f"resuming {st.story_id} from status={st.status}")
                recovery_attempts += 1
                self._guarded(st, article=None)

        # Recovery is not blocked by today's publication target. Recalculate
        # after recovery because a previously unpublished story may now publish.
        published = self.hist.published_today()
        daily_quota = base if self.cfg.manual_override or self.dry else min(
            base, max(0, self.cfg.target_daily_stories - published)
        )
        remaining_run_slots = max(0, base - recovery_attempts)
        quota = min(daily_quota, remaining_run_slots)

        log(
            "RUN",
            f"published today={published}; recovered attempts={recovery_attempts}; "
            f"quota for new stories={quota}",
        )

        if quota <= 0:
            log("RUN", "no new-story quota remains; recovery pass is complete")
            return 0

        # ---------------------------------------------------------- discovery
        for art in self._discover_and_select(quota):
            st = StoryState(
                story_id=make_story_id(art.source_name, art.normalized_url),
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
                self.hist.save_cache(st.story_id, StoryCache(article=art))

            self._guarded(st, article=art)

        log(
            "SUMMARY",
            f"completed={self.completed} "
            f"partial(published, needs follow-up)={self.partial} "
            f"failed/pending={self.failed} (dry_run={self.dry})",
        )

        return 0

    # =================================================================== discovery
    def _discover_and_select(self, quota: int) -> list[SourceArticle]:
        cands: list[SourceArticle] = []

        for sd in src_mod.load_sources(self.cfg.sources_override_file):
            if not sd.enabled:
                continue

            try:
                cands += src_mod.discover(sd, self.fetcher)
            except Exception as exc:
                warn("DISCOVERY", f"{sd.name} failed: {type(exc).__name__}")

        log("DISCOVERY", f"Found {len(cands)} candidate articles")

        seen: set[str] = set()
        fresh: list[SourceArticle] = []

        for c in cands:
            if (
                c.normalized_url in seen
                or self.hist.has_url(c.normalized_url)
                or self.hist.similar_title(c.source_name, c.original_title)
            ):
                continue

            seen.add(c.normalized_url)
            fresh.append(c)

        log("DEDUP", f"Removed {len(cands) - len(fresh)} duplicates (URL / source+title)")

        if not fresh:
            return []

        by_src: dict[str, list[SourceArticle]] = {}

        for c in sorted(fresh, key=lambda a: a.publication_date or a.discovered_at, reverse=True):
            by_src.setdefault(c.source_name, []).append(c)

        batch: list[SourceArticle] = []
        limit = self.cfg.max_candidates_for_triage

        while len(batch) < limit and any(by_src.values()):
            for k in list(by_src):
                if by_src[k] and len(batch) < limit:
                    batch.append(by_src[k].pop(0))

        try:
            items = editorial.triage(self.llm, batch, self.hist.recent_published_titles())
        except GeminiError as exc:
            error("SELECTION", f"triage failed: {exc}")
            return []

        cutoff = utcnow() - timedelta(hours=self.cfg.max_source_age_hours)

        rejected: dict[str, int] = {}
        picked: list[tuple[float, SourceArticle]] = []
        seen_dupe: set[int] = set()

        def reject(reason: str) -> None:
            rejected[reason] = rejected.get(reason, 0) + 1

        for it in sorted(items, key=editorial.rank_score, reverse=True):
            if it.index < 0 or it.index >= len(batch):
                reject("invalid triage index")
                continue

            c = batch[it.index]
            reason = ""

            if not it.suitable:
                reason = "unsuitable"
            elif it.already_published:
                reason = "same event already published"
            elif it.index in seen_dupe or (it.duplicate_of >= 0 and it.duplicate_of in seen_dupe):
                reason = "same event as a better candidate"
            elif it.viral_score < self.cfg.min_viral_score:
                reason = "below quality threshold"
            elif c.publication_date and c.publication_date < cutoff:
                if it.is_evergreen:
                    c.age_exception = "older than MAX_SOURCE_AGE_HOURS but evergreen"
                    log("SELECTION", f"age exception: {c.original_title[:60]} ({c.age_exception})")
                else:
                    reason = "too old"

            if reason:
                reject(reason)
                continue

            seen_dupe.add(it.index)

            if 0 <= it.duplicate_of < len(batch):
                seen_dupe.add(it.duplicate_of)

            picked.append((editorial.rank_score(it), c))

        valid: list[SourceArticle] = []

        for _, c in picked[: quota * 3]:
            full = extractor.extract_article(c, self.fetcher)

            if not full:
                reject("extraction failed")
                continue

            if full.publication_date and full.publication_date < cutoff and not c.age_exception:
                reject("too old (after extraction)")
                continue

            full.age_exception = c.age_exception

            ok, why = extractor.is_valid(full, self.cfg.min_article_chars)

            if not ok:
                reject(why)
                continue

            if self.hist.has_url(full.normalized_url):
                reject("duplicate canonical url")
                continue

            valid.append(full)

            if len(valid) >= quota:
                break

        shortage = (
            ""
            if len(valid) >= quota
            else f"only {len(valid)} story(ies) met the quality bar; quality is never lowered to fill the quota"
        )

        log(
            "SELECTION",
            f"triage_qualified_candidates={len(picked)} selected_stories={len(valid)} "
            f"rejected_stories={rejected} reason_for_shortage={shortage or 'none'}",
        )

        return valid

    # =================================================================== guarded story
    def _guarded(self, st: StoryState, *, article: SourceArticle | None) -> int:
        tag = f"STORY {st.story_id[:8]}"

        try:
            return int(self._process(st, article))

        except StoryFailed as exc:
            error(tag, f"{exc.stage}: {exc}")
            self._record_failure(st, exc.stage, str(exc))

        except Exception as exc:
            error(tag, f"unexpected {type(exc).__name__}")
            self._record_failure(st, "unexpected", type(exc).__name__)

        self.failed += 1
        return 0

    def _record_failure(self, st: StoryState, stage: str, msg: str) -> None:
        if self.dry:
            return

        st.attempts += 1
        st.failed_stage = stage
        st.error = str(msg)[:300]

        if st.attempts >= self.cfg.max_story_attempts and not st.blogger_url:
            st.status = H.FAILED

        self.hist.upsert(st)
        self._push_state(f"history: {st.story_id} ({stage} failed)")

    # =================================================================== pipeline
    def _process(self, st: StoryState, article: SourceArticle | None) -> bool:
        tag = f"STORY {st.story_id[:8]}"

        cache = StoryCache(article=article) if self.dry else self.hist.load_cache(st.story_id)

        article = article or cache.article

        if article and not article.article_text and not cache.content:
            refreshed = extractor.extract_article(article, self.fetcher)
            if not refreshed:
                raise StoryFailed("recovery", "cached source text is missing and re-extraction failed")
            article = refreshed
            cache.article = refreshed

        if not article:
            raise StoryFailed("recovery", "no cached article to resume from")

        if cache.article is None:
            cache.article = article

        # ---- 1. Original Arabic content (+ fact check)
        if not cache.content:
            log(tag, "Analyzing / generating content...")

            try:
                cache.content = editorial.generate_verified(
                    self.llm,
                    article,
                    self.hist.recent_blogger_titles(),
                    lambda t: self.hist.any_similar_published_title(t),
                )
            except (GeminiError, ValueError) as exc:
                raise StoryFailed("generate", str(exc)) from exc

            st.generated_at = iso()
            st.blogger_title = cache.content.blogger_title

            try:
                cache.facebook = build_content_package(cache.content)
            except FacebookError:
                cache.facebook = None

            if not self.dry:
                if st.status in (H.DISCOVERED, H.SELECTED):
                    self.hist.set_status(st, H.GENERATED)
                else:
                    self.hist.upsert(st)

                self.hist.save_cache(st.story_id, cache)

            self._export(st, cache, article)

        content = cache.content

        # ---- 2. Article hero image (AI) + original-photo Facebook composite
        image_exists = bool(
            cache.image
            and _is_file(cache.image.path)
            and _is_file(cache.image.facebook_path)
        )

        if image_exists:
            st.image_status = "generated"

        need_image = (
            (not self.dry or self.cfg.dry_run_generate_images)
            and not image_exists
        )

        if need_image:
            try:
                # Stories cached before the thumbnail brief existed get one now (never raises).
                if not content.thumbnail_prompt:
                    brief = editorial.generate_thumbnail_prompt(self.llm, article, content)

                    if brief:
                        content = content.model_copy(update={"thumbnail_prompt": brief})
                        cache.content = content

                        if not self.dry:
                            self.hist.save_cache(st.story_id, cache)

                ref = None

                if not cache.visual:
                    ref = visual_analyzer.acquire_source_image(article, self.fetcher)
                    log("VISUAL", "Reference image found" if ref else "No reference image available")

                    cache.visual = visual_analyzer.analyze(self.llm, article, (ref[0], ref[1]) if ref else None)

                    log(
                        "VISUAL",
                        f"Subject type: {cache.visual.subject_type}; people={cache.visual.contains_real_people}",
                    )

                elif cache.visual.reference_required or self.cfg.image_mode == "faithful":
                    ref = visual_analyzer.acquire_source_image(article, self.fetcher)
                cache.image = self.imgs.generate(
                    story_id=st.story_id,
                    article=article,
                    v=cache.visual,
                    title=content.blogger_title,
                    article_scene=content.article_scene_idea,
                    facebook_scene=content.facebook_scene_idea,
                    facebook_detail_scene=content.facebook_detail_scene_idea,
                    facebook_composition_type=content.facebook_composition_type,
                    thumbnail_prompt=content.thumbnail_prompt,
                    source_ref=(ref[0], ref[1]) if ref else None,
                    source_url=ref[2] if ref else "",
                    source_sha=ref[3] if ref else "",
                    source_ahash=ref[4] if ref else "",
                    known=self.hist.image_hashes(),
                )

                if self.cfg.image_required and not (
                    cache.image and _is_file(cache.image.path)
                ):
                    raise ImageGenError(
                        "article image provider failed; the Facebook source-photo image was preserved"
                    )

            except (ImageGenError, GeminiError) as exc:
                st.image_status = "failed"

                if not self.dry:
                    self.hist.set_status(st, H.IMAGE_FAILED, error=str(exc))
                    self.hist.save_cache(st.story_id, cache)

                if self.cfg.image_required:
                    raise StoryFailed("image", str(exc)) from exc

                warn(tag, "image failed; continuing without image (IMAGE_REQUIRED=false)")

            else:
                article_image_ok = bool(cache.image and _is_file(cache.image.path))
                st.image_status = "generated" if article_image_ok else "failed"
                if article_image_ok:
                    st.image_generated_at = iso()
                    st.subject_type = cache.visual.subject_type
                    st.identity_confidence = cache.image.identity_confidence
                    st.source_image_url = cache.image.source_image_url
                    st.source_image_hash = cache.image.source_image_hash
                    st.source_image_ahash = cache.image.source_image_ahash
                    st.generated_image_path = cache.image.path
                    st.generated_image_hash = cache.image.generated_hash
                    st.generated_image_ahash = cache.image.generated_ahash
                else:
                    warn(tag, "article image unavailable; source-photo Facebook image remains available")

                if not self.dry:
                    if article_image_ok and st.status in (H.SELECTED, H.GENERATED, H.IMAGE_FAILED):
                        self.hist.set_status(st, H.IMAGE_GENERATED)
                    elif not article_image_ok:
                        self.hist.set_status(
                            st,
                            H.IMAGE_FAILED,
                            error=cache.image.notes if cache.image else "article image unavailable",
                        )
                    else:
                        self.hist.upsert(st)

                    self.hist.save_cache(st.story_id, cache)

                self._export(st, cache, article)

        # ---- dry run ends here
        if self.dry:
            self._write_dry(st, cache)
            self._export(st, cache, article)
            log(tag, "DRY RUN complete (nothing published, not marked completed)")
            return False

        # ---- 3. Blogger
        if not st.blogger_url:
            self._publish_blogger(st, cache, article)

        # ---- 4. Facebook package (only with the REAL Blogger URL)
        if (
            st.facebook_status != "ready"
            or not cache.facebook
            or st.blogger_url not in (cache.facebook.first_comment or "")
            or st.blogger_url in (cache.facebook.post or "")
        ):
            self._facebook(st, cache, article)

        self._export(st, cache, article)

        # ---- 5. WhatsApp
        if st.whatsapp_status not in ("sent", H.WHATSAPP_SKIPPED):
            self._whatsapp(st, cache, article)

        # ---- 6. Final validation
        problems = self._final_checks(st, cache)

        if problems:
            warn(tag, "NOT fully completed: " + "; ".join(problems))

            self.partial += 1
            st.attempts += 1
            self.hist.upsert(st)
            self._push_state(f"history: {st.story_id} (partial)")

            return bool(st.blogger_url)

        self.hist.set_status(st, H.COMPLETED)
        log(tag, "COMPLETED")

        self.completed += 1
        self._push_state(f"history: {st.story_id}")

        return True

    # ------------------------------------------------------------------ stages
    def _publish_blogger(self, st: StoryState, cache: StoryCache, art: SourceArticle) -> None:
        if self.blogger is None:
            raise StoryFailed("blogger", "Blogger client is not available")

        content = cache.content
        img = cache.image

        if not content.blogger_title.strip() or len(content.blogger_html) < 200:
            raise StoryFailed("validate", "title/content invalid")

        has_image = bool(img and _is_file(img.path))

        if self.cfg.image_required and not has_image:
            raise StoryFailed("validate", "valid image missing")

        image_url = ""

        if has_image:
            urls = publish_images(self.cfg, [img.path, img.facebook_path], st.story_id)

            img.public_url = urls.get(img.path, "")
            img.facebook_public_url = urls.get(img.facebook_path, "")

            image_url = img.public_url or data_uri(img.path)

            if not img.public_url:
                warn("IMAGE", "using inline image in Blogger (no public URL)")

        html = editorial.render_blogger_html(content, art, image_url, st.story_id)

        log("BLOGGER", "Publishing...")

        try:
            res = self.blogger.publish(
                story_id=st.story_id,
                title=content.blogger_title,
                html=html,
                labels=content.labels,
                description=content.seo_description,
            )
        except BloggerError as exc:
            self.hist.set_status(st, H.BLOGGER_FAILED, error=str(exc))
            raise StoryFailed("blogger", str(exc)) from exc

        st.blogger_post_id = res.post_id
        st.blogger_url = res.url
        st.published_at = res.published_at

        # cache.content.blogger_html stays pure (no external image URLs) so exports remain correct.
        self.hist.set_status(
            st,
            H.BLOGGER_PUBLISHED,
            blogger_post_id=res.post_id,
            blogger_url=res.url,
            published_at=res.published_at,
            blogger_title=content.blogger_title,
        )
        self.hist.save_cache(st.story_id, cache)

        log("BLOGGER", f"Published successfully\n[BLOGGER] URL: {res.url}")

    def _facebook(self, st: StoryState, cache: StoryCache, art: SourceArticle) -> None:
        try:
            # Facebook assets must be source-photo composites; never substitute
            # the AI-generated Blogger hero image.
            img_path = cache.image.facebook_path if cache.image else ""
            if not img_path or not _is_file(img_path):
                raise FacebookError("a source-photo Facebook image is required; no AI fallback is allowed")

            cache.facebook = build_package(cache.content, st.blogger_url, art.original_url, img_path)

        except FacebookError as exc:
            st.facebook_status = "failed"
            self.hist.set_status(st, st.status, error=f"facebook: {exc}")
            warn("FACEBOOK", f"package failed: {exc} (Blogger URL kept)")
            return

        st.facebook_status = "ready"

        self.hist.set_status(st, H.FACEBOOK_READY)
        self.hist.save_cache(st.story_id, cache)

        log("FACEBOOK", "Promotion package ready")

    def _whatsapp(self, st: StoryState, cache: StoryCache, art: SourceArticle) -> None:
        if st.facebook_status != "ready" or not cache.facebook:
            warn("WHATSAPP", "Facebook package not ready; skipping WhatsApp")
            return

        if not self.cfg.twilio_configured():
            log("WHATSAPP", "Twilio credentials incomplete; skipping notification")
            st.whatsapp_status = H.WHATSAPP_SKIPPED
            self.hist.set_status(st, st.status, whatsapp_status=H.WHATSAPP_SKIPPED)
            return

        try:
            if self.whatsapp is None:
                self.whatsapp = WhatsAppClient(self.cfg)

            img = cache.image
            url = ""

            if img:
                path = img.facebook_path

                if _is_file(path):
                    urls = publish_images(self.cfg, [path], st.story_id, wait_seconds=60)
                    url = urls.get(path, "")

            res = self.whatsapp.send_package(
                title=cache.content.blogger_title,
                blogger_url=st.blogger_url,
                source_name=art.source_name,
                source_url=art.original_url,
                fb=cache.facebook,
                image_public_url=url,
                status="blogger_published / facebook_ready",
                already_sent_parts=st.whatsapp_sent_parts,
            )

        except WhatsAppError as exc:
            st.whatsapp_status = "failed"

            partial = exc.partial_result

            if partial and partial.sent_parts:
                st.whatsapp_sent_parts = list(set(st.whatsapp_sent_parts + partial.sent_parts))

                if partial.message_ids:
                    st.whatsapp_message_ids = list(set(st.whatsapp_message_ids + partial.message_ids))

            self.hist.set_status(
                st,
                H.WHATSAPP_FAILED,
                error=str(exc),
                whatsapp_sent_parts=st.whatsapp_sent_parts,
                whatsapp_message_ids=st.whatsapp_message_ids,
            )

            warn(
                "WHATSAPP",
                f"failed: {exc} (Blogger post kept; sent_parts={st.whatsapp_sent_parts}; "
                "will resend remaining next run)",
            )
            return

        st.whatsapp_status = "sent"
        st.whatsapp_message_ids = list(set(st.whatsapp_message_ids + res.message_ids))
        st.whatsapp_sent_parts = list(set(st.whatsapp_sent_parts + res.sent_parts))
        st.whatsapp_message_id = res.message_ids[0] if res.message_ids else st.whatsapp_message_id

        self.hist.set_status(
            st,
            H.WHATSAPP_SENT,
            whatsapp_status="sent",
            whatsapp_sent_parts=st.whatsapp_sent_parts,
            whatsapp_message_ids=st.whatsapp_message_ids,
            whatsapp_message_id=st.whatsapp_message_id,
        )

        log("WHATSAPP", "Notification sent")

    def _final_checks(self, st: StoryState, cache: StoryCache) -> list[str]:
        problems: list[str] = []

        if not st.blogger_post_id or not st.blogger_url.startswith("https://"):
            problems.append("blogger_url_invalid")

        if not cache.content:
            problems.append("content_invalid")

        if self.cfg.image_required and st.image_status != "generated":
            problems.append("image_invalid")

        if not cache.image or not _is_file(cache.image.facebook_path):
            problems.append("facebook_source_image_invalid")

        if (
            st.facebook_status != "ready"
            or not cache.facebook
            or st.blogger_url not in (cache.facebook.first_comment or "")
            or st.blogger_url in (cache.facebook.post or "")
        ):
            problems.append("facebook_package_invalid")

        if st.whatsapp_status not in ("sent", H.WHATSAPP_SKIPPED):
            problems.append("whatsapp_not_sent")

        return problems

    def _write_dry(self, st: StoryState, cache: StoryCache) -> None:
        out = Path(self.cfg.dry_run_dir) / f"{st.story_id}.json"
        preview = cache.model_copy(deep=True)
        if preview.article:
            preview.article = preview.article.model_copy(
                update={
                    "article_text": "",
                    "description": preview.article.description[:500],
                }
            )
        atomic_write_json(out, preview.model_dump(mode="json"))

        log("DRY-RUN", f"saved preview to {out}")


# ======================================================================= CLI
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Viral Stories Factory")

    ap.add_argument("--dry-run", dest="dry_run", action="store_true", default=None)
    ap.add_argument("--live", dest="dry_run", action="store_false")
    ap.add_argument("--stories", type=int, default=None, help="number of stories this run")
    ap.add_argument("--override", action="store_true", help="ignore the daily target")

    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    cfg = Settings.from_env()

    if args.dry_run is not None:
        cfg = dataclasses.replace(cfg, dry_run=args.dry_run)

    if args.stories is not None:
        cfg = dataclasses.replace(cfg, number_of_stories=max(1, args.stories))

    if args.override:
        cfg = dataclasses.replace(cfg, manual_override=True)

    setup_logging(cfg.secret_values())

    need_publish = not cfg.dry_run

    for w in cfg.warnings(need_publish=need_publish):
        warn("CONFIG", w)

    problems = cfg.validate(need_publish=need_publish)

    if problems:
        for p in problems:
            error("CONFIG", p)
        return 2

    return Factory(cfg).run()


if __name__ == "__main__":
    sys.exit(main())
