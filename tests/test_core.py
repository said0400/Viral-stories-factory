import io
from datetime import datetime, timezone

from PIL import Image

from src import content as editorial
from src.cloudflare_client import _prepare_reference, _size_for
from src.config import Settings
from src.facebook import FacebookError, build_package
from src.history import History
from src.image_generator import REALISTIC_NEWS_STYLE, build_prompt, choose_strategy, validate_image
from src.models import (
    GeneratedContent,
    SourceArticle,
    StoryState,
    VisualAnalysis,
)
from src.photo_composer import (
    PhotoAssessmentBatch,
    PhotoAssessmentSchema,
    SelectedPhoto,
    analyze_source_photos,
    compose_original_square,
    compose_square,
)
from src.twilio_whatsapp import _chunks
from src.utils import make_story_id, normalize_url, parse_datetime, title_similarity
from src.visual_analyzer import ahash, hamming


def test_normalize_url_strips_tracking():
    a = normalize_url("http://www.Example.com/a/b/?utm_source=x&fbclid=1&id=5#frag")
    assert a == "https://example.com/a/b?id=5"


def test_story_id_stable():
    u = normalize_url("https://x.com/a?utm_campaign=1")
    assert make_story_id("Reuters", u) == make_story_id(
        "reuters", normalize_url("https://x.com/a")
    )


def test_title_similarity():
    assert (
        title_similarity(
            "Man finds gold ring in garden",
            "Man finds a gold ring in his garden",
        )
        > 0.8
    )
    assert title_similarity("Cat saves family", "Stock markets fall") < 0.6


def test_parse_datetime_variants():
    assert parse_datetime("2026-10-01T10:00:00Z").tzinfo is not None
    assert parse_datetime("Wed, 01 Oct 2026 10:00:00 GMT").year == 2026
    assert parse_datetime("garbage") is None


def _state(i, status="discovered", pub=None):
    return StoryState(
        story_id=i,
        source="Reuters",
        original_url="u" + i,
        normalized_url="n" + i,
        original_title="Title " + i,
        status=status,
        published_at=pub or "",
    )


def test_history_dedup_quota_recovery(tmp_path):
    h = History(
        tmp_path / "h.json",
        tmp_path / "c",
        "Africa/Casablanca",
    )

    now = datetime.now(timezone.utc).isoformat()

    h.upsert(_state("1", "completed", now))
    h.upsert(_state("2", "blogger_published", now))
    h.upsert(_state("3", "whatsapp_failed", now))
    h.upsert(_state("4", "failed"))

    assert h.has_url("n1")
    assert not h.has_url("zzz")

    assert h.published_today() == 3

    assert {s.story_id for s in h.resumable(3)} == {"2", "3"}

    assert h.similar_title("Reuters", "Title 1") is not None

    h2 = History(
        tmp_path / "h.json",
        tmp_path / "c",
    )

    assert len(h2.rows) == 4


def test_sanitize_html_blocks_injection():
    out = editorial.sanitize_html(
        '<p onclick="x()">مرحبا</p>'
        '<script>alert(1)</script>'
        '<img src=x>'
        '<a href="http://e">link</a>'
        '<h1>t</h1>'
    )

    assert "script" not in out
    assert "onclick" not in out
    assert "<img" not in out
    assert "<a" not in out
    assert "<p>مرحبا</p>" in out


def _content(
    post="جملة طويلة كفاية لنشر منشور فيسبوك يدفع للقراءة والفضول."
):
    return GeneratedContent(
        blogger_title="ت",
        blogger_html="<p>x</p>",
        seo_description="d",
        labels=["غرائب"],
        facebook_title="عنوان",
        facebook_post=post,
        facebook_hashtags=["#قصص_حقيقية", "#غرائب", "#قصة"],
        first_comment_hook="التفاصيل هنا",
        article_scene_idea="a",
        facebook_scene_idea="b",
    )


def test_facebook_requires_real_blogger_url():
    ok = build_package(
        _content(),
        "https://blog.blogspot.com/2026/10/x.html",
        "https://src.com/a",
    )

    assert "blogspot.com" not in ok.post
    assert ok.post.startswith("عنوان\n\n")
    assert "#قصص_حقيقية" in ok.post
    assert "blogspot.com" in ok.first_comment

    for bad in (
        "",
        "http://x.com/a",
        "https://src.com/a",
    ):
        try:
            build_package(
                _content(),
                bad,
                "https://src.com/a",
            )
            assert False, bad
        except FacebookError:
            pass

    try:
        build_package(
            _content("see {BLOGGER_URL} " + "x" * 50),
            "https://b.blogspot.com/p",
            "s",
        )
        assert False
    except FacebookError:
        pass


def test_blogger_html_has_marker_attribution_and_ai_note():
    art = SourceArticle(
        source_name="Reuters",
        original_title="t",
        original_url="https://www.reuters.com/a",
        normalized_url="n",
        discovered_at=datetime.now(timezone.utc),
    )

    html = editorial.render_blogger_html(
        _content(),
        art,
        "https://img/x.jpg",
        "abc123",
    )

    assert "story_id:abc123" in html
    assert "Reuters" in html
    assert "https://img/x.jpg" in html


def test_people_use_real_face_reference_including_minors():
    v = VisualAnalysis(
        subject_type="person",
        contains_real_people=True,
    )

    assert choose_strategy(v, True, "illustration") == ("reference_identity", REALISTIC_NEWS_STYLE)

    kid = VisualAnalysis(
        subject_type="person",
        contains_real_people=True,
        involves_minors=True,
    )

    assert choose_strategy(kid, True, "illustration") == ("reference_identity", REALISTIC_NEWS_STYLE)

    animal = VisualAnalysis(
        subject_type="animal",
        reference_required=True,
    )

    assert choose_strategy(
        animal,
        True,
        "illustration",
    )[0] == "reference_identity"

    assert choose_strategy(
        animal,
        False,
        "illustration",
    )[0] == "editorial"

    p = build_prompt(
        "people_safe",
        "illustration",
        v,
        "a street scene",
        "T",
        "16:9",
    )

    assert "Show visible faces naturally and clearly" in p
    assert "anonymize people" in p


def test_facebook_composition_templates_are_square_and_text_free():
    photos = [Image.new("RGB", (800, 600), color) for color in ((180, 30, 30), (30, 180, 30), (30, 30, 180))]
    for layout, expected_count in (
        ("INSET_CIRCLE_RIGHT", 2),
        ("INSET_SQUARE_LEFT", 2),
        ("DIPTYCH_SPLIT", 2),
        ("DIPTYCH_STACK", 2),
        ("TRIPTYCH", 3),
        ("TRIPTYCH_BOTTOM", 3),
    ):
        canvas, used = compose_square(photos[:expected_count], layout)
        assert canvas.size == (1080, 1080)
        assert used == expected_count


def test_original_photo_analysis_selects_and_enlarges_story_focus():
    now = datetime.now(timezone.utc)
    article = SourceArticle(
        source_name="Example",
        original_title="A story with a distinctive object",
        original_url="https://example.com/story",
        normalized_url="https://example.com/story",
        discovered_at=now,
        description="A person shows an important object.",
        article_text="The article describes the object and the event.",
    )
    photos = [Image.new("RGB", (800, 600), (20, 40, 220)), Image.new("RGB", (800, 600), (220, 30, 20))]

    class FakePhotoGemini:
        def generate_json(self, prompt, schema, **kwargs):
            assert schema is PhotoAssessmentBatch
            assert len(kwargs["images"]) == 2
            return PhotoAssessmentBatch(
                items=[
                    PhotoAssessmentSchema(
                        index=0,
                        relevance_score=35,
                        visual_impact_score=40,
                        focus_center_x=500,
                        focus_center_y=500,
                        focus_width=300,
                        focus_height=300,
                        focal_description="background",
                    ),
                    PhotoAssessmentSchema(
                        index=1,
                        relevance_score=95,
                        visual_impact_score=90,
                        focus_center_x=700,
                        focus_center_y=300,
                        focus_width=200,
                        focus_height=200,
                        focal_description="important object",
                    ),
                ]
            )

    selected = analyze_source_photos(FakePhotoGemini(), article, photos, limit=2)
    assert selected[0].image is photos[1]
    assert selected[0].focus_box == (600, 200, 800, 400)

    canvas, used = compose_original_square(selected, "INSET_CIRCLE_RIGHT")
    assert canvas.size == (1080, 1080)
    assert used == 1  # the unrelated background image falls below the relevance threshold
    assert canvas.getpixel((540, 900)) == (220, 30, 20)


def test_original_photo_quad_grid_uses_four_focus_crops():
    selected = [
        SelectedPhoto(
            Image.new("RGB", (800, 600), color),
            (250, 180, 750, 820),
            90 - i,
            85 - i,
            f"photo {i}",
        )
        for i, color in enumerate(((220, 20, 20), (20, 220, 20), (20, 20, 220), (220, 220, 20)))
    ]
    canvas, used = compose_original_square(selected, "QUAD_GRID")
    assert canvas.size == (1080, 1080)
    assert used == 4
    assert canvas.getpixel((100, 100)) != canvas.getpixel((1000, 1000))


def _img(color, noise=False):
    im = Image.new("RGB", (800, 600), color)

    if noise:
        for x in range(0, 800, 7):
            for y in range(0, 600, 5):
                im.putpixel(
                    (x, y),
                    (
                        x % 255,
                        y % 255,
                        (x * y) % 255,
                    ),
                )

    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


def test_image_validation():
    ok, why, _ = validate_image(
        _img((0, 0, 0)),
        [],
    )

    assert not ok
    assert "black" in why

    ok, why, _ = validate_image(
        b"notanimage",
        [],
    )

    assert not ok

    good = _img(
        (120, 30, 200),
        noise=True,
    )

    ok, why, im = validate_image(
        good,
        [],
    )

    assert ok

    ah = ahash(im)

    assert not validate_image(
        good,
        [],
        source_ahash=ah,
    )[0]

    assert not validate_image(
        good,
        [("x", ah)],
    )[0]

    assert hamming(ah, ah) == 0


def test_high_quality_image_dimensions_and_cloudflare_reference_limit():
    assert _size_for("1:1", 1536) == (1536, 1536)
    assert _size_for("16:9", 1536) == (1536, 864)
    assert _size_for("16:9", 1537) == (1536, 864)

    src = Image.new("RGB", (1200, 900), (90, 120, 150))
    buf = io.BytesIO()
    src.save(buf, "PNG")
    prepared, mime = _prepare_reference(buf.getvalue(), "image/png")
    ref = Image.open(io.BytesIO(prepared))
    assert mime == "image/jpeg"
    assert max(ref.size) <= 480

    cfg = Settings.from_env({})
    assert cfg.cloudflare_image_model.endswith("flux-2-dev")
    assert cfg.image_long_side == 1536
    assert cfg.cloudflare_image_steps == 25
    assert "gemini-2.5-flash" not in cfg.content_models
    assert cfg.gemini_fallback_model == "gemini-3.5-flash"
    overridden = Settings.from_env({
        "GEMINI_CONTENT_FALLBACKS": "gemini-3.6-flash,gemini-2.5-flash",
    })
    assert overridden.content_models == ["gemini-3.8-flash", "gemini-3.6-flash"]


def test_whatsapp_chunking():
    text = "\n".join(
        ["line " * 20] * 40
    )

    parts = _chunks(
        text,
        500,
    )

    assert all(len(p) <= 520 for p in parts)
    assert len(parts) > 3
