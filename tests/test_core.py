import io
from datetime import datetime, timezone

from PIL import Image

from src import content as editorial
from src.facebook import FacebookError, build_package
from src.history import History
from src.image_generator import build_prompt, choose_strategy, validate_image
from src.models import GeneratedContent, SourceArticle, StoryState, VisualAnalysis
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

    assert "blogspot.com" in ok.post
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


def test_people_never_get_realistic_face_strategy():
    v = VisualAnalysis(
        subject_type="person",
        contains_real_people=True,
    )

    assert choose_strategy(
        v,
        True,
        "illustration",
    ) == ("people_safe", "illustration")

    kid = VisualAnalysis(
        subject_type="person",
        contains_real_people=True,
        involves_minors=True,
    )

    assert choose_strategy(
        kid,
        True,
        "illustration",
    ) == ("people_safe", "faceless")

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

    assert "NOT resemble any real individual" in p


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
