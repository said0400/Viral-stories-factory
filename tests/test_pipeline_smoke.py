"""End-to-end dry-run with fakes: discovery -> triage -> content -> fact check -> image."""
import dataclasses
import io
from datetime import datetime, timezone

from PIL import Image

from src import image_generator as image_mod
from src import main as m
from src.config import Settings
from src.models import (
    ArticleBodySchema,
    EditorialMetadataSchema,
    FactCheckSchema,
    ImageCheckSchema,
    SourceArticle,
    TriageItem,
    TriageResult,
    VisualAnalysis,
    VisualSchema,
)
from src.gemini_client import ImageGenError
from src.photo_composer import PhotoAssessmentBatch, PhotoAssessmentSchema


class FakeGemini:
    def __init__(self, *a, **k):
        pass

    def generate_json(self, prompt, schema, **kw):
        if schema is TriageResult:
            n = prompt.count("\n[") + 1

            return TriageResult(
                items=[
                    TriageItem(
                        index=i,
                        suitable=True,
                        viral_score=80,
                        curiosity_score=80,
                        share_score=70,
                        comment_score=60,
                        originality_score=70,
                        emotional_score=60,
                        is_evergreen=False,
                        duplicate_of=-1,
                        already_published=False,
                        event_key="e",
                        reason="r",
                    )
                    for i in range(n)
                ]
            )

        if schema is ArticleBodySchema:
            return ArticleBodySchema(
                blogger_html=(
                    "<p>" + "مقدمة " * 65 + "</p>"
                    "<h2>ما الذي حدث؟</h2><p>" + "تفصيل " * 65 + "</p>"
                    "<h2>لماذا يلفت الأمر الانتباه؟</h2><p>" + "سياق " * 65 + "</p>"
                    "<p>" + "خلاصة " * 65 + "</p>"
                ),
            )

        if schema is EditorialMetadataSchema:
            return EditorialMetadataSchema(
                blogger_title="عنوان تجريبي",
                seo_description="وصف",
                labels=["غرائب"],
                facebook_title="عنوان فيسبوك",
                facebook_post="منشور فيسبوك طويل بما يكفي لإثارة الفضول دون كشف النهاية.",
                facebook_hashtags=["#قصة", "#غرائب", "#قصص_حقيقية"],
                first_comment_hook="التفاصيل هنا",
                article_scene_idea="a cozy street",
                facebook_scene_idea="a market",
                facebook_detail_scene_idea="a closer view of the same market",
                facebook_composition_type="INSET_CIRCLE_RIGHT",
            )

        if schema is FactCheckSchema:
            return FactCheckSchema(
                all_claims_supported=True,
                unsupported_claims=[],
            )

        if schema is VisualSchema:
            return VisualSchema(
                subject_type="animal",
                identity_critical=True,
                contains_real_people=False,
                involves_minors=False,
                identity_features=["orange fur"],
                scene_features=["garden"],
                new_scene_direction="low angle",
                identity_confidence="medium",
                summary="a cat",
            )

        if schema is ImageCheckSchema:
            return ImageCheckSchema(
                relevant_to_story=True,
                contains_text_or_watermark=False,
                obvious_defects=False,
                reason="ok",
            )

        if schema is PhotoAssessmentBatch:
            count = len(kw.get("images") or [])
            return PhotoAssessmentBatch(
                items=[
                    PhotoAssessmentSchema(
                        index=i,
                        relevance_score=90 - i * 5,
                        visual_impact_score=85 - i * 5,
                        focus_center_x=500,
                        focus_center_y=450,
                        focus_width=380,
                        focus_height=420,
                        focal_description="main story subject",
                    )
                    for i in range(count)
                ]
            )

        raise AssertionError(schema)


def _png(seed=0):
    im = Image.new(
        "RGB",
        (900, 700),
        (10, 90, 160),
    )

    for x in range(0, 900, 6):
        for y in range(0, 700, 4):
            im.putpixel(
                (x, y),
                (
                    (x * 3 + seed * 83) % 255,
                    (y * 5 + seed * 59) % 255,
                    (x + y + seed * 37) % 255,
                ),
            )

    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def _source_photo(seed=0):
    return Image.open(io.BytesIO(_png(seed))).convert("RGB")


class FakeProvider:
    def __init__(self):
        self.calls = 0
        self.requests = []

    def generate(self, prompt, refs, aspect):
        self.calls += 1
        self.requests.append((prompt, refs, aspect))
        return _png(self.calls), "image/png"


def test_dry_run_end_to_end(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)

    art = SourceArticle(
        source_name="Bored Panda",
        original_title="Cat found in a wall",
        original_url="https://boredpanda.com/cat-found",
        normalized_url="https://boredpanda.com/cat-found",
        discovered_at=now,
        publication_date=now,
        description="d",
        article_text="x " * 400,
        main_image_url="",
    )

    monkeypatch.setattr(
        m,
        "GeminiClient",
        FakeGemini,
    )
    monkeypatch.setattr(
        image_mod,
        "fetch_photos",
        lambda *args, **kwargs: [_source_photo(31), _source_photo(32)],
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
                    "name": "Bored Panda",
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
        dry_run=True,
        dry_run_generate_images=True,
        data_dir=tmp_path,
        dry_run_dir=tmp_path / "dry_run",
        history_file=tmp_path / "history.json",
        cache_dir=tmp_path / "cache",
        exports_dir=tmp_path / "exports",
        image_vlm_check=True,
        gemini_api_key="k",
    )

    f = m.Factory(cfg)

    f.imgs.provider = FakeProvider()

    assert f.run() == 0
    assert f.imgs.provider.calls == 1  # article image only; Facebook is source-pixel compositing
    prompt, refs, aspect = f.imgs.provider.requests[0]
    assert len(refs) == 2
    assert aspect == "16:9"
    assert "YouTube-style editorial thumbnail" in prompt
    assert "absolutely no written characters" in prompt

    previews = list(
        (tmp_path / "dry_run").glob("*.json")
    )

    assert len(previews) == 1

    generated_images = list(
        (tmp_path / "dry_run" / "images").glob("*_generated.jpg")
    )

    assert generated_images

    facebook_images = list((tmp_path / "dry_run" / "images").glob("*_facebook.jpg"))
    assert facebook_images
    assert Image.open(facebook_images[0]).size == (1080, 1080)

    history_file = tmp_path / "history.json"

    assert (
        not history_file.exists()
        or history_file.read_text(encoding="utf-8").strip()
        in ("", "[]")
    )


def test_facebook_source_composite_survives_article_image_provider_failure(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    art = SourceArticle(
        source_name="Bored Panda",
        original_title="Cat found in a wall",
        original_url="https://boredpanda.com/cat-found",
        normalized_url="https://boredpanda.com/cat-found",
        discovered_at=now,
        description="A source description.",
        article_text="article context " * 40,
    )
    monkeypatch.setattr(
        image_mod,
        "fetch_photos",
        lambda *args, **kwargs: [_source_photo(41), _source_photo(42)],
    )

    class BrokenArticleProvider:
        calls = 0

        def generate(self, *args, **kwargs):
            self.calls += 1
            raise ImageGenError("simulated Cloudflare article-image outage")

    cfg = dataclasses.replace(
        Settings(),
        dry_run=True,
        data_dir=tmp_path,
        image_vlm_check=False,
        gemini_api_key="k",
    )
    generator = image_mod.ImageGenerator(cfg, FakeGemini(), fetcher=object())
    provider = BrokenArticleProvider()
    generator.provider = provider

    result = generator.generate(
        story_id="source-only-test",
        article=art,
        v=VisualAnalysis(summary="cat story", subject_type="animal"),
        title="Test title",
        article_scene="cat",
        facebook_scene="cat",
        source_ref=None,
        source_url="",
        source_sha="",
        source_ahash="",
        known=[],
    )

    assert not result.path
    assert result.facebook_path
    assert Image.open(result.facebook_path).size == (1080, 1080)
    assert provider.calls > 0
