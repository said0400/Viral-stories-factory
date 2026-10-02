"""End-to-end dry-run with fakes: discovery -> triage -> content -> fact check -> image."""
import dataclasses
import io
from datetime import datetime, timezone

from PIL import Image

from src import main as m
from src.config import Settings
from src.models import (ContentSchema, FactCheckSchema, ImageCheckSchema, SourceArticle,
                        TriageItem, TriageResult, VisualSchema)


class FakeGemini:
    def __init__(self, *a, **k): pass

    def generate_json(self, prompt, schema, **kw):
        if schema is TriageResult:
            n = prompt.count("\n[") + 1
            return TriageResult(items=[TriageItem(index=i, suitable=True, viral_score=80, curiosity_score=80,
                share_score=70, comment_score=60, originality_score=70, emotional_score=60, is_evergreen=False,
                duplicate_of=-1, already_published=False, event_key="e", reason="r") for i in range(n)])
        if schema is ContentSchema:
            return ContentSchema(blogger_title="عنوان تجريبي", blogger_html="<p>" + "نص " * 80 + "</p>",
                seo_description="وصف", labels=["غرائب"], facebook_title="عنوان فيسبوك",
                facebook_post="منشور فيسبوك طويل بما يكفي لإثارة الفضول دون كشف النهاية.",
                first_comment_hook="التفاصيل هنا", article_scene_idea="a cozy street", facebook_scene_idea="a market")
        if schema is FactCheckSchema:
            return FactCheckSchema(all_claims_supported=True, unsupported_claims=[])
        if schema is VisualSchema:
            return VisualSchema(subject_type="animal", identity_critical=True, contains_real_people=False,
                involves_minors=False, identity_features=["orange fur"], scene_features=["garden"],
                new_scene_direction="low angle", identity_confidence="medium", summary="a cat")
        if schema is ImageCheckSchema:
            return ImageCheckSchema(relevant_to_story=True, contains_text_or_watermark=False,
                                    obvious_defects=False, reason="ok")
        raise AssertionError(schema)


def _png():
    im = Image.new("RGB", (900, 700), (10, 90, 160))
    for x in range(0, 900, 6):
        for y in range(0, 700, 4):
            im.putpixel((x, y), ((x * 3) % 255, (y * 5) % 255, (x + y) % 255))
    b = io.BytesIO(); im.save(b, "PNG"); return b.getvalue()


class FakeProvider:
    def generate(self, prompt, refs, aspect):
        return _png(), "image/png"


def test_dry_run_end_to_end(tmp_path, monkeypatch):
    art = SourceArticle(source_name="Bored Panda", original_title="Cat found in a wall", original_url="https://boredpanda.com/cat-found",
        normalized_url="https://boredpanda.com/cat-found", discovered_at=datetime.now(timezone.utc),
        publication_date=datetime.now(timezone.utc), description="d", article_text="x " * 400, main_image_url="")
    monkeypatch.setattr(m, "GeminiClient", FakeGemini)
    monkeypatch.setattr(m.src_mod, "load_sources", lambda f="": [type("S", (), {"enabled": True, "name": "Bored Panda"})()])
    monkeypatch.setattr(m.src_mod, "discover", lambda sd, f: [art])
    monkeypatch.setattr(m.extractor, "extract_article", lambda c, f: c)
    cfg = dataclasses.replace(Settings(), dry_run=True, dry_run_generate_images=True, data_dir=tmp_path,
                              history_file=tmp_path / "history.json", image_vlm_check=True, gemini_api_key="k")
    f = m.Factory(cfg)
    f.imgs.provider = FakeProvider()
    assert f.run() == 0
    previews = list((tmp_path / "dry_run").glob("*.json"))
    assert len(previews) == 1
    assert list((tmp_path / "dry_run" / "images").glob("*_generated.jpg"))
    assert not (tmp_path / "history.json").exists() or (tmp_path / "history.json").read_text().strip() in ("", "[]")
