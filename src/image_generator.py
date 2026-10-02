"""Replaceable image layer. Reference-aware for places/animals/objects; people get a
non-photoreal illustration or a faceless scene (never a real person's face)."""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageStat

from . import logger
from .config import Settings
from .gemini_client import GeminiClient, GeminiError, ImageGenError
from .models import ImageResult, SourceArticle, VisualAnalysis, ImageCheckSchema
from .utils import sha256_hex
from .visual_analyzer import ahash, hamming

NEGATIVE = ("unrelated person, different animal, generic location, duplicate composition, distorted face, "
            "extra limbs, random text, captions, watermark, logo, low quality, blurry subject")


class ImageProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, references: list[tuple[bytes, str]] | None,
                 aspect_ratio: str) -> tuple[bytes, str]: ...


class GeminiImageProvider(ImageProvider):
    def __init__(self, gem: GeminiClient) -> None:
        self.gem = gem

    def generate(self, prompt, references, aspect_ratio):
        return self.gem.generate_image(prompt, references=references, aspect_ratio=aspect_ratio)


# ------------------------------------------------------------------ strategy
def choose_strategy(v: VisualAnalysis, has_ref: bool, people_style: str) -> tuple[str, str]:
    """Return (strategy, style). Real people are never recreated photo-realistically."""
    if v.contains_real_people:
        style = "faceless" if v.involves_minors else people_style
        return "people_safe", style
    if has_ref and v.reference_required:
        return "reference_identity", "photorealistic"
    return "editorial", "editorial illustration"


def build_prompt(strategy: str, style: str, v: VisualAnalysis, scene_idea: str,
                 title: str, aspect: str, simple: bool = False) -> str:
    base = [f"Create ONE high-quality image for an article titled: \"{title}\".",
            f"Aspect ratio {aspect}. Strong visual hierarchy, clear main subject, no text, no captions, no watermark, no logo."]
    scene = v.new_scene_direction or scene_idea
    if strategy == "reference_identity":
        keep = "; ".join(v.identity_features) or "all distinctive visual features"
        base += ["The attached photo is a REFERENCE of the real subject. Depict THE SAME real subject "
                 f"(keep exactly: {keep}) in a NEW scene: {scene_idea}. {scene}.",
                 "Change camera angle, composition, lighting and moment. Do NOT copy the reference photo; do NOT invent a similar-looking different subject.",
                 "Realistic, cinematic photography look."]
    elif strategy == "people_safe":
        if style == "faceless":
            base += [f"Scene: {scene_idea}. Show any people only from behind, in silhouette, shadow or far away - "
                     "no visible faces. Focus on place, objects, atmosphere and the key moment of the story."]
        else:
            base += [f"Stylised editorial DIGITAL ILLUSTRATION (clearly not a photograph), scene: {scene_idea}.",
                     "Any people are generic stylised figures with simplified features; they must NOT resemble any real individual. "
                     "Do not depict real faces."]
        if v.scene_features:
            base.append("Setting cues (non-human only): " + "; ".join(v.scene_features[:6]) + ".")
    else:
        base += [f"Editorial illustration of: {scene_idea}. Do not claim to reproduce a real event exactly; "
                 "avoid inventing specific details."]
    if not simple:
        base.append("Avoid: " + NEGATIVE + ".")
    return " ".join(base)


# ------------------------------------------------------------------ validation
def validate_image(data: bytes, known: list[tuple[str, str]], source_ahash: str = "") -> tuple[bool, str, Image.Image | None]:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return False, "cannot open/corrupt", None
    if not data:
        return False, "empty file", None
    if min(img.size) < 512:
        return False, f"too small {img.size}", None
    g = img.convert("L")
    st = ImageStat.Stat(g)
    if st.mean[0] < 8 or st.stddev[0] < 6:
        return False, "black/blank image", None
    ah = ahash(img)
    if source_ahash and hamming(ah, source_ahash) <= 3:
        return False, "too similar to the source image", None
    for sha, kn in known:
        if kn and hamming(ah, kn) <= 2:
            return False, "duplicate of an earlier generated image", None
    return True, "", img


def _save_jpeg(img: Image.Image, path: Path) -> tuple[str, str]:
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))
    path.parent.mkdir(parents=True, exist_ok=True)
    q = 88
    while True:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=q, optimize=True)
        if buf.tell() < 4_500_000 or q <= 60:   # stay under WhatsApp's 5MB image limit
            break
        q -= 8
    path.write_bytes(buf.getvalue())
    return sha256_hex(buf.getvalue()), ahash(img)


def vlm_check(gem: GeminiClient, jpeg: bytes, title: str, summary: str) -> tuple[bool, str]:
    try:
        r = gem.generate_json(
            f"Story: {title}\n{summary}\nIs this image relevant to the story, free of text/watermarks and obvious defects "
            "(deformed hands/faces, garbled anatomy)?", ImageCheckSchema,
            images=[(jpeg, "image/jpeg")], temperature=0.0, tag="IMGCHECK")
        ok = r.relevant_to_story and not r.contains_text_or_watermark and not r.obvious_defects
        return ok, r.reason
    except GeminiError:
        return True, "check unavailable (skipped)"


# ------------------------------------------------------------------ main entry
class ImageGenerator:
    def __init__(self, cfg: Settings, gem: GeminiClient, provider: ImageProvider | None = None) -> None:
        self.cfg, self.gem = cfg, gem
        self.provider = provider or GeminiImageProvider(gem)

    def _one(self, *, kind: str, strategy: str, style: str, v: VisualAnalysis, scene_idea: str,
             title: str, aspect: str, ref: tuple[bytes, str] | None, known: list[tuple[str, str]],
             source_ahash: str, out_path: Path, summary: str) -> tuple[str, str, str]:
        """Attempts: full prompt+ref -> simpler prompt+ref -> simple prompt, no ref."""
        plans = [(False, ref), (True, ref), (True, None)] if ref else [(False, None), (True, None)]
        for i, (simple, r) in enumerate(plans, 1):
            if i > self.cfg.max_retries + 1:
                break
            if strategy == "reference_identity" and r is None:
                strat, sty = "editorial", "editorial illustration"   # no reference -> do not claim same subject
            else:
                strat, sty = strategy, style
            prompt = build_prompt(strat, sty, v, scene_idea, title, aspect, simple=simple)
            try:
                logger.log("IMAGE", f"{kind}: generating ({strat}/{sty}, attempt {i}, ref={'yes' if r else 'no'})")
                data, _ = self.provider.generate(prompt, [r] if r else None, aspect)
            except (ImageGenError, GeminiError) as exc:
                logger.warn("IMAGE", f"{kind}: attempt {i} failed ({exc})")
                continue
            ok, why, img = validate_image(data, known, source_ahash)
            if not ok:
                logger.warn("IMAGE", f"{kind}: rejected ({why})")
                continue
            sha, ah = _save_jpeg(img, out_path)
            if self.cfg.image_vlm_check:
                good, reason = vlm_check(self.gem, out_path.read_bytes(), title, summary)
                if not good:
                    logger.warn("IMAGE", f"{kind}: relevance check failed ({reason})")
                    continue
            logger.log("IMAGE", f"{kind}: validation passed")
            return sha, ah, f"{strat}/{sty}"
        raise ImageGenError(f"{kind}: no valid image after attempts")

    def generate(self, *, story_id: str, article: SourceArticle, v: VisualAnalysis, title: str,
                 article_scene: str, facebook_scene: str, source_ref: tuple[bytes, str] | None,
                 source_url: str, source_sha: str, source_ahash: str,
                 known: list[tuple[str, str]]) -> ImageResult:
        strategy, style = choose_strategy(v, bool(source_ref), self.cfg.people_image_style)
        # identity preservation is only claimed when a reference was used for a non-human subject
        confidence = v.identity_confidence if strategy == "reference_identity" else "low"
        out = self.cfg.images_dir / f"{story_id}_generated.jpg"
        sha, ah, used = self._one(kind="article", strategy=strategy, style=style, v=v,
                                  scene_idea=article_scene, title=title, aspect="16:9",
                                  ref=source_ref if strategy == "reference_identity" else None,
                                  known=known, source_ahash=source_ahash, out_path=out,
                                  summary=v.summary or article.description)
        res = ImageResult(path=str(out), strategy=strategy, style=style, identity_confidence=confidence,
                          generated_hash=sha, generated_ahash=ah, source_image_url=source_url,
                          source_image_hash=source_sha, source_image_ahash=source_ahash,
                          notes=f"identity preservation depends on provider/model capabilities; used={used}")
        if self.cfg.facebook_separate_image:
            fb_out = self.cfg.images_dir / f"{story_id}_facebook.jpg"
            try:
                self._one(kind="facebook", strategy=strategy, style=style, v=v, scene_idea=facebook_scene,
                          title=title, aspect="1:1",
                          ref=source_ref if strategy == "reference_identity" else None,
                          known=[*known, (sha, ah)], source_ahash=source_ahash, out_path=fb_out,
                          summary=v.summary or article.description)
                res.facebook_path = str(fb_out)
            except ImageGenError as exc:
                logger.warn("IMAGE", f"facebook image failed; reusing article image ({exc})")
                res.facebook_path = res.path
        else:
            res.facebook_path = res.path
        return res
