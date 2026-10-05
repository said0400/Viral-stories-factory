"""Replaceable image layer.

IMAGE_MODE=faithful (default): re-render the SOURCE photo as a cinematic version of the same scene (article image).
IMAGE_MODE=creative: reference-aware editorial photojournalism.
Stories with minors, or without a usable source image, always use the creative path.

FACEBOOK_IMAGE_MODE=photo (default): the Facebook image is composed from the article's own photos
(no AI, no filter, no text). If no photo passes screening, a natural realistic generated image is used.
All generated images follow a strict photojournalism standard: raw press photograph, zero text, zero graphics, zero borders.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageStat

from . import logger
from .cloudflare_client import CloudflareClient
from .config import Settings
from .gemini_client import GeminiClient, GeminiError, ImageGenError, ImageQuotaError
from .models import ImageCheckSchema, ImageResult, SourceArticle, VisualAnalysis
from .photo_composer import compose, fetch_photos, screen_photos
from .utils import PoliteFetcher, sha256_hex
from .visual_analyzer import ahash, hamming

ARTICLE_ASPECT = "16:9"
FACEBOOK_ASPECT = "1:1"    # used only by the generated (fallback) Facebook image

FAITHFUL = "faithful_restyle"

# Neutral photojournalistic style for hyper-realistic viral storytelling photos
REALISTIC_NEWS_STYLE = (
    "authentic raw press photograph, shot on 35mm lens, natural ambient lighting, "
    "editorial news photojournalism, realistic human skin textures, sharp focus, "
    "unedited documentary photograph, real-life context, zero filters, zero CGI, zero illustration"
)

NEUTRAL_STYLE = REALISTIC_NEWS_STYLE

NEGATIVE = (
    "illustration, vector, cartoon, 3d render, painting, graphic design, artwork, poster, "
    "fake looking, CGI, digital art, stylized, smooth plastic skin, Photoshop edit, collage, "
    "text, lettering, watermark, caption, logo, brand name, borders, frames, graphic overlays, arrows, "
    "distorted face, extra limbs, deformed anatomy, blurry subject, ugly composition"
)

NEGATIVE_FAITHFUL = (
    "cartoon, illustration, painting, anime, 3d render, added graphic frames, borders, "
    "changed faces, altered key subjects, random text, captions, watermark, logo, blurry"
)


class ImageProvider(ABC):
    @abstractmethod
    def generate(
        self,
        prompt: str,
        references: list[tuple[bytes, str]] | None,
        aspect_ratio: str,
    ) -> tuple[bytes, str]:
        ...


class GeminiImageProvider(ImageProvider):
    def __init__(self, gem: GeminiClient) -> None:
        self.gem = gem

    def generate(self, prompt, references, aspect_ratio):
        return self.gem.generate_image(prompt, references=references, aspect_ratio=aspect_ratio)


class CloudflareImageProvider(ImageProvider):
    def __init__(self, client: CloudflareClient) -> None:
        self.client = client

    def generate(self, prompt, references, aspect_ratio):
        return self.client.generate_image(prompt, references=references, aspect_ratio=aspect_ratio)


def build_provider(cfg: Settings, gem: GeminiClient) -> ImageProvider:
    if cfg.image_provider == "gemini":
        return GeminiImageProvider(gem)

    return CloudflareImageProvider(CloudflareClient(cfg))


# ------------------------------------------------------------------ helpers
def crop_to_aspect(ref: tuple[bytes, str], aspect: str) -> tuple[bytes, str]:
    """Crop the source photo to the target aspect ratio (top-biased: faces are usually in the upper part)."""
    data, mime = ref

    try:
        a, b = aspect.split(":", 1)
        target = float(a) / float(b)

        img = Image.open(io.BytesIO(data))
        img.load()
        img = img.convert("RGB")

        w, h = img.size
        current = w / h

        if abs(current - target) > 0.02:
            if current > target:
                new_w = int(round(h * target))
                left = (w - new_w) // 2
                img = img.crop((left, 0, left + new_w, h))
            else:
                new_h = int(round(w / target))
                top = int((h - new_h) * 0.25)
                img = img.crop((0, top, w, top + new_h))

        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=90)

        return buf.getvalue(), "image/jpeg"

    except Exception:
        return data, mime


# ------------------------------------------------------------------ strategy
def choose_strategy(v: VisualAnalysis, has_ref: bool, people_style: str) -> tuple[str, str]:
    """
    Return (strategy, style) for the creative path.

    - Real people are never recreated photorealistically without safe rules.
    - Reference identity is used for identity-critical subjects.
    """
    subject_type = getattr(v, "subject_type", None) or "other"

    if people_style not in {"reference", "illustration", "faceless"}:
        people_style = "illustration"

    def people_safe() -> tuple[str, str]:
        style = "faceless" if v.involves_minors else people_style
        if style == "reference":
            style = REALISTIC_NEWS_STYLE
        return "people_safe", style

    if subject_type == "person":
        return people_safe()

    if subject_type in {"multiple_subjects", "event", "scene"} and v.contains_real_people:
        if has_ref and v.reference_required and v.identity_critical:
            return "reference_identity", REALISTIC_NEWS_STYLE
        return people_safe()

    if has_ref and v.reference_required:
        return "reference_identity", REALISTIC_NEWS_STYLE

    return "editorial", REALISTIC_NEWS_STYLE


def _faithful_prompt(style_text: str, aspect: str, simple: bool) -> str:
    parts = [
        f"Square 1:1 aspect ratio press photograph." if aspect == "1:1" else f"Aspect ratio {aspect}.",
        "The attached image is the SOURCE PHOTOGRAPH.",
        "Re-render it as an authentic press news photograph of the EXACT same scene.",
        (
            "Keep exactly the same subjects, the same number of people, the same faces, expressions, poses, "
            "clothing, objects, setting, background layout and framing."
        ),
        "Do not add, remove, replace or invent any person, animal, object, text or detail.",
        f"Change only the visual treatment: {style_text}.",
        "Authentic press news photograph look. Not a cartoon, not an illustration, not a painting, no graphics.",
        "No text, no captions, no watermark, no logo, no artificial yellow/red graphic borders.",
    ]

    if not simple:
        parts.append("Avoid: " + NEGATIVE_FAITHFUL + ".")

    return " ".join(parts)


def build_prompt(
    strategy: str,
    style: str,
    v: VisualAnalysis,
    scene_idea: str,
    title: str,
    aspect: str,
    composition_type: str = "SINGLE_HERO",
    simple: bool = False,
) -> str:
    """Provider-neutral photographic press prompt with explicit structural layout handling."""
    if strategy == FAITHFUL:
        return _faithful_prompt(style, aspect, simple)

    base = [
        f"Square 1:1 authentic press news photograph for a story titled: '{title}'." if aspect == "1:1" else f"Aspect ratio {aspect} authentic press news photograph.",
        "CAMERA & STYLE: Shot on 35mm DSLR camera, raw unedited press photojournalism, authentic natural lighting, real human textures, zero digital editing, zero CGI.",
        "STRICT NO-GRAPHICS RULE: Absolutely NO text, NO watermarks, NO captions, NO logos, NO artificial graphic frames, NO borders, NO arrows, NO illustration style.",
    ]

    comp = str(composition_type or "SINGLE_HERO").upper()

    if "DIPTYCH" in comp or "SPLIT" in comp:
        base.append(
            "COMPOSITION LAYOUT: Seamless side-by-side split-screen press photograph. "
            "Left side shows one key aspect of the story, right side shows the second related aspect. "
            f"Scene description: {scene_idea}."
        )
    elif "DETAIL" in comp or "MAIN_PLUS" in comp:
        base.append(
            "COMPOSITION LAYOUT: Dynamic press photograph featuring a clear main subject in frame, "
            "with a sharp focal point on a crucial secondary detail within the same real-life environment. "
            f"Scene description: {scene_idea}."
        )
    elif "FOREGROUND" in comp:
        base.append(
            "COMPOSITION LAYOUT: News photograph with shallow depth of field. Main subject sharp in the foreground, "
            "with contextual environment naturally visible in the background. "
            f"Scene description: {scene_idea}."
        )
    else:  # SINGLE_HERO
        base.append(
            f"COMPOSITION LAYOUT: Single powerful focal point editorial press photo. Scene description: {scene_idea}."
        )

    subject_type = getattr(v, "subject_type", None) or "other"

    if strategy == "reference_identity":
        keep = (
            "; ".join(v.identity_features[:12])
            if v.identity_features
            else "all clearly visible distinctive identity features"
        )

        base += [
            "The attached image is a REFERENCE for the specific real subject described by the story.",
            f"Subject type: {subject_type}.",
            f"Preserve these identity-critical characteristics: {keep}.",
            (
                "Use the SAME specific non-human subject when the reference supports its identity. "
                "Do not replace it with a generic animal, vehicle, building, place or object."
            ),
            "Change camera angle, composition, framing, lighting and/or moment. Do NOT copy the original photograph.",
            "Realistic editorial photojournalistic photograph look.",
        ]

        if v.contains_real_people:
            base += [
                "People may appear naturally as part of the press shot.",
                "Keep human faces natural without unnatural AI distortion.",
            ]

    elif strategy == "people_safe":
        if style == "faceless" or v.involves_minors:
            base += [
                (
                    "Show people naturally from candid angles, side profiles, rear views, cropped without direct full faces, "
                    "or in medium shots where full facial reconstruction is not required."
                ),
                "Focus on the realistic situation, human actions, hands, environment, or surrounding objects.",
            ]
        else:
            base += [
                "Depict everyday real people naturally in an authentic real-life environment.",
                "Maintain raw photographic texture and natural camera lighting.",
            ]

        if v.scene_features:
            base.append("Supported setting cues: " + "; ".join(v.scene_features[:6]) + ".")

    else:
        base += [
            f"Authentic news photograph of: {scene_idea}.",
            "Use only factual elements supported by the story.",
        ]

    if v.involves_minors:
        base.append(
            "If minors are present, do not show direct identifiable faces. "
            "Use distant, rear-view, silhouette or non-identifying depiction."
        )

    if not simple:
        base.append("Avoid: " + NEGATIVE + ".")

    return " ".join(base)


# ------------------------------------------------------------------ validation
def validate_image(
    data: bytes,
    known: list[tuple[str, str]],
    source_ahash: str = "",
) -> tuple[bool, str, Image.Image | None]:
    """Validate integrity, dimensions and duplication."""
    if not data:
        return False, "empty file", None

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception:
        return False, "cannot open/corrupt", None

    if min(img.size) < 400:
        return False, f"too small {img.size}", None

    stat = ImageStat.Stat(img.convert("L"))

    if stat.mean[0] < 8 or stat.stddev[0] < 6:
        return False, "black/blank image", None

    generated = ahash(img)

    if source_ahash and hamming(generated, source_ahash) <= 3:
        return False, "too similar to the source image", None

    for _, known_ahash in known:
        if known_ahash and hamming(generated, known_ahash) <= 2:
            return False, "duplicate of an earlier generated image", None

    return True, "", img


def _save_jpeg(img: Image.Image, path: Path) -> tuple[str, str]:
    """Save a normalised JPEG under the WhatsApp size limit; return (sha256, ahash)."""
    img = img.convert("RGB")
    img.thumbnail((1600, 1600))

    path.parent.mkdir(parents=True, exist_ok=True)

    quality = 88

    while True:
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=quality, optimize=True)

        if buf.tell() < 4_500_000 or quality <= 60:
            break

        quality -= 8

    payload = buf.getvalue()
    path.write_bytes(payload)

    return sha256_hex(payload), ahash(img)


def vlm_check(gem: GeminiClient, jpeg: bytes, title: str, summary: str) -> tuple[bool, str]:
    """Creative-path quality gate: relevance to the story. Fail-open if unavailable."""
    try:
        r = gem.generate_json(
            (
                f"Story: {title}\n{summary}\n\n"
                "Evaluate ONLY the supplied image against the story for news publishing:\n"
                "1. Is the image relevant to the story?\n"
                "2. Is it COMPLETELY FREE of visible text, captions, watermarks, logos, or artificial graphic overlays/borders?\n"
                "3. Does it look like a realistic photograph (not an obvious cartoon, 3D render, artwork, or illustration)?\n"
                "4. Is it free of obvious visual defects such as deformed hands, extra limbs, or severe rendering artifacts?"
            ),
            ImageCheckSchema,
            images=[(jpeg, "image/jpeg")],
            temperature=0.0,
            tag="IMGCHECK",
        )

        ok = r.relevant_to_story and not r.contains_text_or_watermark and not r.obvious_defects

        return ok, r.reason

    except GeminiError:
        return True, "check unavailable (skipped)"
    except Exception as exc:
        logger.warn("IMGCHECK", f"VLM check error ({type(exc).__name__}); skipping")
        return True, "check unavailable (skipped)"


def fidelity_check(
    gem: GeminiClient,
    source: tuple[bytes, str],
    result_jpeg: bytes,
    forbid_text: bool = False,
) -> tuple[bool, str]:
    """Faithful-path quality gate: is the result a faithful restyle of the SOURCE photo? Fail-open.

    relevant_to_story and obvious_defects always decide. The text field decides only when
    `forbid_text` is set (Facebook), because the article source itself may legitimately contain text.
    """
    text_rule = (
        "contains_text_or_watermark = true if Image 2 shows ANY visible text, letters, captions, "
        "subtitles, watermark or logo anywhere in it."
        if forbid_text
        else "contains_text_or_watermark: not used, answer false."
    )

    try:
        r = gem.generate_json(
            (
                "Image 1 is the SOURCE photograph. Image 2 is a restyled version of it.\n"
                "Judge ONLY whether Image 2 is a faithful restyle of Image 1.\n"
                "relevant_to_story = true when Image 2 shows the same scene as Image 1: the same kind of "
                "subjects, the same number of people, the same setting and a similar composition. "
                "A different colour grade or lighting is fine.\n"
                f"{text_rule}\n"
                "obvious_defects = true only for clearly deformed faces, hands or anatomy, "
                "or severe rendering artifacts.\n"
                "Keep reason to one short sentence."
            ),
            ImageCheckSchema,
            images=[(source[0], source[1] or "image/jpeg"), (result_jpeg, "image/jpeg")],
            temperature=0.0,
            tag="IMGCHECK",
        )

        ok = r.relevant_to_story and not r.obvious_defects

        if forbid_text and r.contains_text_or_watermark:
            ok = False

        return ok, r.reason

    except GeminiError:
        return True, "check unavailable (skipped)"
    except Exception as exc:
        logger.warn("IMGCHECK", f"fidelity check error ({type(exc).__name__}); skipping")
        return True, "check unavailable (skipped)"


# ------------------------------------------------------------------ main entry
class ImageGenerator:
    def __init__(
        self,
        cfg: Settings,
        gem: GeminiClient,
        provider: ImageProvider | None = None,
        fetcher: PoliteFetcher | None = None,
    ) -> None:
        self.cfg = cfg
        self.gem = gem
        self._provider = provider
        self.fetcher = fetcher

    @property
    def provider(self) -> ImageProvider:
        # Built lazily so a dry run without images never needs image credentials.
        if self._provider is None:
            self._provider = build_provider(self.cfg, self.gem)
        return self._provider

    def _keep_rejected(self, data: bytes, name: str) -> None:
        """Dry runs only: keep rejected images (and the source) inside the exported artifact for inspection."""
        if not self.cfg.dry_run or not data:
            return

        try:
            folder = self.cfg.export_path / "rejected"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
        except Exception:
            pass

    def _facebook_photo(self, article: SourceArticle, v: VisualAnalysis, out_path: Path) -> tuple[str, str] | None:
        """Facebook image from the article's own screened photos. Returns (sha, ahash) or None to use the fallback."""
        if self.cfg.facebook_image_mode != "photo":
            return None

        if v.involves_minors:
            logger.warn("IMAGE", "facebook photo mode skipped (minors involved); using generated image")
            return None

        if self.fetcher is None:
            logger.warn("IMAGE", "facebook photo mode unavailable (no fetcher); using generated image")
            return None

        try:
            candidates = fetch_photos(article, self.fetcher)

            if not candidates:
                logger.warn("IMAGE", "facebook photo mode: no usable source photo; using generated image")
                return None

            clean = screen_photos(self.gem, candidates)

            if not clean:
                logger.warn(
                    "IMAGE",
                    f"facebook photo mode: none of {len(candidates)} photo(s) passed screening "
                    "(text/logo, minors, collage); using generated image",
                )
                return None

            canvas, used = compose([candidates[i] for i in clean], self.cfg.facebook_layout)
            sha, ah = _save_jpeg(canvas, out_path)

            logger.log("IMAGE", f"facebook: real-photo layout built from {used} of {len(clean)} approved photo(s)")

            return sha, ah

        except Exception as exc:
            logger.warn("IMAGE", f"facebook photo layout failed ({type(exc).__name__}); using generated image")
            return None

    def _one(
        self,
        *,
        kind: str,
        strategy: str,
        style: str,
        v: VisualAnalysis,
        scene_idea: str,
        title: str,
        aspect: str,
        ref: tuple[bytes, str] | None,
        known: list[tuple[str, str]],
        source_ahash: str,
        out_path: Path,
        summary: str,
        composition_type: str = "SINGLE_HERO",
        forbid_text: bool = False,
    ) -> tuple[str, str, str]:
        """
        Bounded attempts (cost control).

        Faithful: full+ref, simple+ref (never falls back to an unrelated image).
        With reference: full+ref, simple+ref, simple without ref (editorial fallback).
        Without reference: full, simple.
        A quota error stops immediately.
        """
        if strategy == FAITHFUL:
            plans: list[tuple[bool, tuple[bytes, str] | None]] = [(False, ref), (True, ref)]
        elif ref:
            plans = [(False, ref), (True, ref), (True, None)]
        else:
            plans = [(False, None), (True, None)]

        max_attempts = max(1, min(len(plans), self.cfg.max_retries + 1, 3))

        for i, (simple, r) in enumerate(plans[:max_attempts], 1):
            if strategy == "reference_identity" and r is None:
                strat, sty = "editorial", REALISTIC_NEWS_STYLE
            else:
                strat, sty = strategy, style

            prompt = build_prompt(
                strat, sty, v, scene_idea, title, aspect, composition_type=composition_type, simple=simple
            )

            try:
                logger.log(
                    "IMAGE",
                    f"{kind}: generating ({strat}, layout={composition_type}, attempt {i}/{max_attempts}, ref={'yes' if r else 'no'})",
                )
                data, _mime = self.provider.generate(prompt, [r] if r else None, aspect)

            except ImageQuotaError:
                raise

            except (ImageGenError, GeminiError) as exc:
                logger.warn("IMAGE", f"{kind}: attempt {i} failed ({exc})")
                continue

            ok, why, img = validate_image(data, known, source_ahash)

            if not ok or img is None:
                logger.warn("IMAGE", f"{kind}: rejected ({why or 'no decoded image'})")
                continue

            try:
                sha, ah = _save_jpeg(img, out_path)
            except Exception as exc:
                logger.warn("IMAGE", f"{kind}: failed to save image ({type(exc).__name__})")
                continue

            if self.cfg.image_vlm_check:
                payload = b""

                try:
                    payload = out_path.read_bytes()

                    if strat == FAITHFUL and r is not None:
                        good, reason = fidelity_check(self.gem, r, payload, forbid_text)
                    else:
                        good, reason = vlm_check(self.gem, payload, title, summary)

                except Exception as exc:
                    logger.warn("IMAGE", f"{kind}: quality check failed ({type(exc).__name__}); continuing")
                    good, reason = True, "check unavailable (skipped)"

                if not good:
                    logger.warn("IMAGE", f"{kind}: quality check failed ({reason})")

                    if strat == FAITHFUL and r is not None:
                        self._keep_rejected(r[0], f"{out_path.stem}_source.jpg")

                    self._keep_rejected(payload, f"{out_path.stem}_rejected{i}.jpg")

                    try:
                        out_path.unlink(missing_ok=True)
                    except Exception:
                        pass

                    continue

            logger.log("IMAGE", f"{kind}: validation passed")

            return sha, ah, strat

        raise ImageGenError(f"{kind}: no valid image after attempts")

    def generate(
        self,
        *,
        story_id: str,
        article: SourceArticle,
        v: VisualAnalysis,
        title: str,
        article_scene: str,
        facebook_scene: str,
        facebook_composition_type: str = "SINGLE_HERO",
        source_ref: tuple[bytes, str] | None,
        source_url: str,
        source_sha: str,
        source_ahash: str,
        known: list[tuple[str, str]],
    ) -> ImageResult:
        """
        Generate the article image (16:9) and the Facebook image.

        Unique filenames per story: {story_id}_generated.jpg / {story_id}_facebook.jpg
        """
        faithful = (
            self.cfg.image_mode == "faithful"
            and source_ref is not None
            and not v.involves_minors
        )

        if faithful:
            strategy, style = FAITHFUL, self.cfg.cinematic_style
            article_ref = crop_to_aspect(source_ref, ARTICLE_ASPECT)
            facebook_ref = crop_to_aspect(source_ref, FACEBOOK_ASPECT)
            check_ahash = ""            # the output is SUPPOSED to resemble the source
            confidence = v.identity_confidence
            reference_used = True
        else:
            if self.cfg.image_mode == "faithful":
                reason = "minors involved" if v.involves_minors else "no usable source image"
                logger.warn("IMAGE", f"faithful mode not possible ({reason}); using creative path")

            strategy, style = choose_strategy(v, bool(source_ref), self.cfg.people_image_style)
            article_ref = facebook_ref = source_ref if strategy == "reference_identity" else None
            check_ahash = source_ahash
            confidence = (
                v.identity_confidence
                if (strategy == "reference_identity" and source_ref is not None and v.reference_required)
                else "low"
            )
            reference_used = article_ref is not None

        summary = v.summary or article.description
        out = self.cfg.images_dir / f"{story_id}_generated.jpg"

        sha, ah, used = self._one(
            kind="article",
            strategy=strategy,
            style=style,
            v=v,
            scene_idea=article_scene,
            title=title,
            aspect=ARTICLE_ASPECT,
            ref=article_ref,
            known=known,
            source_ahash=check_ahash,
            out_path=out,
            summary=summary,
            composition_type="SINGLE_HERO",
        )

        res = ImageResult(
            path=str(out),
            strategy=strategy,
            style=style,
            identity_confidence=confidence,
            generated_hash=sha,
            generated_ahash=ah,
            source_image_url=source_url,
            source_image_hash=source_sha,
            source_image_ahash=source_ahash,
            notes=(
                "Output fidelity depends on provider/model capabilities; "
                f"mode={self.cfg.image_mode}; provider={self.cfg.image_provider}; used={used}; "
                f"reference_used={'yes' if reference_used else 'no'}; "
                f"facebook_mode={self.cfg.facebook_image_mode}"
            ),
        )

        # ---------------------------------------------------------------- Facebook image
        fb_out = self.cfg.images_dir / f"{story_id}_facebook.jpg"
        photo = self._facebook_photo(article, v, fb_out)

        if photo is not None:
            res.facebook_path = str(fb_out)
            res.facebook_image_hash, res.facebook_image_ahash = photo

        elif self.cfg.facebook_separate_image:
            fb_style = REALISTIC_NEWS_STYLE if faithful else style

            try:
                fb_sha, fb_ah, _ = self._one(
                    kind="facebook",
                    strategy=strategy,
                    style=fb_style,
                    v=v,
                    scene_idea=facebook_scene,
                    title=title,
                    aspect=FACEBOOK_ASPECT,
                    ref=facebook_ref,
                    # Faithful: both images derive from the same photo, so do not compare them.
                    known=known if faithful else [*known, (sha, ah)],
                    source_ahash=check_ahash,
                    out_path=fb_out,
                    summary=summary,
                    composition_type=facebook_composition_type,
                    forbid_text=True,
                )

                res.facebook_path = str(fb_out)
                res.facebook_image_hash = fb_sha
                res.facebook_image_ahash = fb_ah

            except ImageQuotaError as exc:
                logger.warn("IMAGE", f"facebook image skipped (quota): reusing article image ({exc})")

                res.facebook_path = res.path
                res.facebook_image_hash = sha
                res.facebook_image_ahash = ah

            except ImageGenError as exc:
                logger.warn("IMAGE", f"facebook image failed; reusing article image ({exc})")

                res.facebook_path = res.path
                res.facebook_image_hash = sha
                res.facebook_image_ahash = ah

        else:
            res.facebook_path = res.path
            res.facebook_image_hash = sha
            res.facebook_image_ahash = ah

        return res
