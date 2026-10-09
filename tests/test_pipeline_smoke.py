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
    ImageResult,
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


def test_article_image_survives_facebook_source_photo_failure(tmp_path, monkeypatch):
    now = datetime.now(timezone.utc)
    art = SourceArticle(
        source_name="Bored Panda",
        original_title="Cat found in a wall",
        original_url="https://boredpanda.com/cat-found",
        normalized_url="https://boredpanda.com/cat-found",
        discovered_at=now,
        article_text="article context " * 40,
    )
    monkeypatch.setattr(image_mod, "fetch_photos", lambda *args, **kwargs: [])
    cfg = dataclasses.replace(
        Settings(),
        dry_run=True,
        data_dir=tmp_path,
        image_vlm_check=False,
        gemini_api_key="k",
    )
    generator = image_mod.ImageGenerator(cfg, FakeGemini(), fetcher=object())
    provider = FakeProvider()
    generator.provider = provider

    result = generator.generate(
        story_id="article-only-test",
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

    assert result.path
    assert Image.open(result.path).size[0] >= 512
    assert not result.facebook_path
    assert provider.calls == 1


def test_image_required_helper_accepts_either_saved_image(tmp_path):
    article_image = tmp_path / "article.jpg"
    facebook_image = tmp_path / "facebook.jpg"
    article_image.write_bytes(b"article")
    facebook_image.write_bytes(b"facebook")

    assert m._has_any_image(ImageResult(facebook_path=str(facebook_image)))
    assert m._has_any_image(ImageResult(path=str(article_image)))
    assert not m._has_any_image(ImageResult())


def test_cloudflare_credentials_are_not_required_for_original_facebook_image():
    cfg = dataclasses.replace(
        Settings(),
        dry_run=True,
        dry_run_generate_images=True,
        image_required=True,
        image_provider="cloudflare",
        cloudflare_account_id="",
        cloudflare_api_token="",
    )

    assert not any("CLOUDFLARE_ACCOUNT_ID" in problem for problem in cfg.validate())
    assert any("original-photo Facebook image" in warning for warning in cfg.warnings())


def test_blogger_accepts_facebook_only_image_when_image_is_required(
    tmp_path, monkeypatch
):
    from src.models import BloggerResult, GeneratedContent, StoryCache, StoryState

    facebook_image = tmp_path / "facebook.jpg"
    facebook_image.write_bytes(b"facebook")
    cfg = dataclasses.replace(Settings(), image_required=True)

    class FakeHistory:
        def set_status(self, *_args, **_kwargs):
            pass

        def save_cache(self, *_args, **_kwargs):
            pass

    class FakeBlogger:
        def publish(self, **kwargs):
            assert "<img" not in kwargs["html"]
            return BloggerResult(
                post_id="post-1",
                url="https://blog.example/post",
                published_at="now",
            )

    monkeypatch.setattr(
        m,
        "publish_images",
        lambda _cfg, paths, _story_id: {paths[0]: "https://cdn.example/facebook.jpg"},
    )
    monkeypatch.setattr(
        m.editorial, "render_blogger_html", lambda *_args: "<p>story</p>"
    )

    factory = m.Factory.__new__(m.Factory)
    factory.cfg = cfg
    factory.hist = FakeHistory()
    factory.blogger = FakeBlogger()
    factory._push_state = lambda *_args: None
    content = GeneratedContent(
        blogger_title="Title",
        blogger_html="<p>" + "story " * 40 + "</p>",
        seo_description="Description",
        labels=["story"],
        facebook_title="Facebook title",
        facebook_post="Facebook post",
        first_comment_hook="Comment",
        article_scene_idea="scene",
        facebook_scene_idea="scene",
    )
    cache = StoryCache(
        content=content,
        image=ImageResult(path="", facebook_path=str(facebook_image)),
    )
    state = StoryState(
        story_id="story-only-facebook",
        source="source",
        original_url="https://source.example/story",
        normalized_url="https://source.example/story",
        original_title="Story",
    )
    article = SourceArticle(
        source_name="source",
        original_title="Story",
        original_url="https://source.example/story",
        normalized_url="https://source.example/story",
        discovered_at=datetime.now(timezone.utc),
    )

    factory._publish_blogger(state, cache, article)

    assert state.blogger_url == "https://blog.example/post"
    assert cache.image.facebook_public_url == "https://cdn.example/facebook.jpg"
