"""Live-mode flow with fakes: WhatsApp fails -> next run resumes WITHOUT a second Blogger post."""
import dataclasses
from datetime import datetime, timezone

from src import image_generator as image_mod
from src import main as m
from src.config import Settings
from src.models import BloggerResult, SourceArticle, WhatsAppResult
from src.twilio_whatsapp import WhatsAppError
from tests.test_pipeline_smoke import FakeGemini, FakeProvider, _source_photo


class FakeBlogger:
    calls = 0

    def __init__(self, cfg):
        pass

    def publish(self, **kw):
        FakeBlogger.calls += 1

        return BloggerResult(
            post_id="p1",
            url="https://blog.blogspot.com/2026/10/x.html",
            published_at=datetime.now(timezone.utc).isoformat(),
        )


class FakeWA:
    fail = True

    def __init__(self, cfg):
        pass

    def send_package(self, **kw):
        if FakeWA.fail:
            raise WhatsAppError("boom")

        return WhatsAppResult(
            message_ids=["SM1"],
        )


def test_recovery_no_duplicate_blogger(tmp_path, monkeypatch):
    FakeBlogger.calls = 0
    FakeWA.fail = True

    art = SourceArticle(
        source_name="Reuters",
        original_title="Dog rescued",
        original_url="https://reuters.com/a-2026-10-01",
        normalized_url="https://reuters.com/a-2026-10-01",
        discovered_at=datetime.now(timezone.utc),
        publication_date=datetime.now(timezone.utc),
        description="d",
        article_text="x " * 400,
    )

    monkeypatch.setattr(
        m,
        "GeminiClient",
        FakeGemini,
    )
    monkeypatch.setattr(
        image_mod,
        "fetch_photos",
        lambda *args, **kwargs: [_source_photo(51), _source_photo(52)],
    )

    monkeypatch.setattr(
        m,
        "BloggerClient",
        FakeBlogger,
    )

    monkeypatch.setattr(
        m,
        "WhatsAppClient",
        FakeWA,
    )

    monkeypatch.setattr(
        m,
        "publish_images",
        lambda cfg, paths, sid, wait_seconds=0: {
            p: "https://img/" + p.split("/")[-1]
            for p in paths
        },
    )

    monkeypatch.setattr(
        m.src_mod,
        "load_sources",
        lambda f="": [
            type(
                "S",
                (),
                {
                    "enabled": True,
                    "name": "Reuters",
                },
            )()
        ],
    )

    monkeypatch.setattr(
        m.src_mod,
        "discover",
        lambda sd, f: [art],
    )

    monkeypatch.setattr(
        m.extractor,
        "extract_article",
        lambda c, f: c,
    )

    cfg = dataclasses.replace(
        Settings(),
        dry_run=False,
        data_dir=tmp_path,
        history_file=tmp_path / "history.json",
        cache_dir=tmp_path / "cache",
        exports_dir=tmp_path / "exports",
        gemini_api_key="k",
        target_daily_stories=6,
        max_stories_per_run=1,
        blogger_blog_id="b",
        twilio_account_sid="AC00000000000000000000000000000000",
        twilio_auth_token="test-auth-token",
        twilio_whatsapp_number="+15550000000",
        your_personal_number="+15550000001",
        image_vlm_check=False,
        facebook_separate_image=True,
    )

    # ─────────────────────────────────────────────
    # RUN 1
    # Blogger succeeds, WhatsApp fails.
    # ─────────────────────────────────────────────
    f1 = m.Factory(cfg)
    f1.imgs.provider = FakeProvider()

    assert f1.run() == 0

    assert len(f1.hist.rows) == 1

    row = list(f1.hist.rows.values())[0]

    assert row.status == "whatsapp_failed"
    assert row.blogger_url
    assert row.blogger_url == "https://blog.blogspot.com/2026/10/x.html"
    assert FakeBlogger.calls == 1

    assert row.whatsapp_status == "failed"
    assert row.facebook_status == "ready"

    # ─────────────────────────────────────────────
    # RUN 2
    # WhatsApp now succeeds.
    # Existing Blogger post must NOT be created again.
    # ─────────────────────────────────────────────
    FakeWA.fail = False

    f2 = m.Factory(cfg)
    f2.imgs.provider = FakeProvider()

    assert f2.run() == 0

    assert len(f2.hist.rows) == 1

    row = list(f2.hist.rows.values())[0]

    assert row.status == "completed"
    assert row.whatsapp_status == "sent"

    # Critical idempotency assertion:
    # Blogger was called only once across both runs.
    assert FakeBlogger.calls == 1

    # Critical recovery assertion:
    # The second run resumed the existing story instead of creating
    # another history entry.
    assert len(f2.hist.rows) == 1
