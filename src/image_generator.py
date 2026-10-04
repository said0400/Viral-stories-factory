"""Replaceable image layer.

IMAGE_MODE=faithful (default): re-render the SOURCE photo as a cinematic version of the same scene.
IMAGE_MODE=creative: reference-aware illustrations (old behaviour).
Stories with minors, or without a usable source image, always use the creative path.

FACEBOOK_IMAGE_MODE=photo (default): the Facebook image is composed from the article's own photos
(no AI, no text). Stories involving minors fall back to a generated image.
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
from .photo_composer import compose, fetch_photos
from .utils import PoliteFetcher, sha256_hex
from .visual_analyzer import ahash, hamming

ARTICLE_ASPECT = "16:9"
FACEBOOK_ASPECT = "1:1"    # used only by the generated (non-photo) Facebook image

FAITHFUL = "faithful_restyle"

NEGATIVE = (
    "unrelated person, different animal, different vehicle, different building, "
    "generic location, duplicate composition, copied source photograph, "
    "distorted face, recognizable real person's face, extra limbs, "
    "deformed anatomy, random text, captions, watermark, logo, "
    "low quality, blurry subject, duplicate subject, invented factual details"
)

NEGATIVE_FAITHFUL = (
    "cartoon, illustration, painting, anime, 3d render, extra people, missing people, "
    "changed faces, changed clothing, different background, added objects, "
    "random text, captions, watermark, logo, blurry, distorted anatomy"
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

    - Real people are never recreated photorealistically.
    - Reference identity is used only for identity-critical NON-HUMAN subjects.
    """
    subject_type = getattr(v, "subject_type", None) or "other"

    if people_style not in {"reference", "illustration", "faceless"}:
        people_style = "illustration"

    def people_safe() -> tuple[str, str]:
        style = "faceless" if v.involves_minors else people_style
        if style == "reference":
            style = "illustration"
        return "people_safe", style

    if subject_type == "person":
        return people_safe()

    if subject_type in {"multiple_subjects", "event", "scene"} and v.contains_real_people:
        if has_ref and v.reference_required and v.identity_critical:
            return "reference_identity", "photorealistic"
        return people_safe()

    if has_ref and v.reference_required:
        return "reference_identity", "photorealistic"

    return "editorial", "editorial illustration"


def _faithful_prompt(style_text: str, aspect: str, simple: bool) -> str:
    parts = [
        f"Aspect ratio {aspect}.",
        "The attached image is the SOURCE PHOTOGRAPH.",
        "Re-render it as a faithful cinematic version of the SAME scene.",
        (
            "Keep exactly the same subjects, the same number of people, the same faces, expressions, poses, "
            "clothing, objects, setting, background layout and framing."
        ),
        "Do not add, remove, replace or invent any person, animal, object, text or detail.",
        f"Change only the visual treatment: {style_text}.",
        "Photorealistic photograph look. Not a cartoon, not an illustration, not a painting.",
        "No text, no captions, no watermark, no logo.",
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
    simple: bool = False,
) -> str:
    """Provider-neutral image prompt."""
    if strategy == FAITHFUL:
        return _faithful_prompt(style, aspect, simple)

    base = [
        f'Create ONE high-quality image for an article titled: "{title}".',
        f"Aspect ratio {aspect}.",
        "Strong visual hierarchy.",
        "Clear main subject.",
        "No text, no captions, no watermark, no logo.",
        "Create a NEW composition rather than copying the source image.",
    ]

    scene = v.new_scene_direction or scene_idea or "a visually clear editorial scene related to the story"
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
            f"Create a NEW scene based on: {scene_idea}.",
            f"Additional scene direction: {scene}.",
            "Change camera angle, composition, framing, lighting and/or moment. Do NOT copy the original photograph.",
            "Do not create a similar-looking substitute subject. Preserve the specific subject's supported identity.",
            "Do not invent factual details that are not supported by the story or visible reference.",
            "Realistic cinematic photography look.",
        ]

        if v.contains_real_people:
            base += [
                "People may appear only as incidental contextual elements unless the story establishes them as the subject.",
                "Do not identify, reconstruct or reproduce the face of any real person.",
            ]

    elif strategy == "people_safe":
        if style == "faceless":
            base += [
                f"Scene: {scene_idea}.",
                (
                    "Show people only from behind, in silhouette, in shadow, cropped without faces, "
                    "or sufficiently far away that faces are not visible."
                ),
                "No recognizable or reconstructed real person's face.",
                "Focus on the place, objects, atmosphere and supported context of the story.",
                (
                    "Do not invent clothing, facial features, expressions, age appearance, "
                    "ethnicity or other personal attributes."
                ),
            ]
        else:
            base += [
                "Stylised editorial DIGITAL ILLUSTRATION, clearly not a photograph.",
                f"Scene: {scene_idea}.",
                "Any people must be generic stylised figures with simplified non-identifying features.",
                "They must NOT resemble any real individual.",
                "Do not depict or reconstruct a real person's face.",
                "Do not invent factual details about the people.",
            ]

        if v.scene_features:
            base.append("Supported setting cues only: " + "; ".join(v.scene_features[:6]) + ".")

    else:
        base += [
            f"Editorial illustration of: {scene_idea}.",
            "The image is illustrative and must not claim to reproduce a real event exactly.",
            "Use only factual elements supported by the story.",
            (
                "Avoid inventing specific people, objects, locations, architecture, "
                "clothing, weather, injuries or actions."
            ),
        ]

    if v.involves_minors:
        base.append(
            "If minors are present, do not show identifiable faces. "
            "Use distant, rear-view, silhouette or non-identifying depiction."
        )

    if v.contains_real_people:
        base.append("Never guess or reconstruct facial identity from the reference.")

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

    if min(img.size) < 512:
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
                "Evaluate ONLY the supplied image against the story.\n"
                "Is the image relevant to the story?\n"
                "Is it free of visible text, watermarks and logos?\n"
                "Does it have obvious visual defects such as deformed hands, faces, "
                "anatomy or severe rendering artifacts?\n"
                "Do not reject an image merely because it is an illustration."
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
) -> tuple[bool, str]:
    """Faithful-path quality gate: is the result a faithful restyle of the SOURCE photo? Fail-open.

    Only `relevant_to_story` and `obvious_defects` decide. The text field is ignored here: the source
    itself may contain text, and a faithful restyle then legitimately contains it too.
    """
    try:
        r = gem.generate_json(
            (
                "Image 1 is the SOURCE photograph. Image 2 is a cinematic restyled version of it.\n"
                "Judge ONLY whether Image 2 is a faithful restyle of Image 1.\n"
                "relevant_to_story = true when Image 2 shows the same scene as Image 1: the same kind of "
                "subjects, the same number of people, the same setting and a similar composition. "
                "A different colour grade, lighting or mood is expected and fine.\n"
                "contains_text_or_watermark: not used, answer false.\n"
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
        """Facebook image from the article's own photos. Returns (sha, ahash) or None to use the generated path."""
        if self.cfg.facebook_image_mode != "photo":
            return None

        if v.involves_minors:
            logger.warn("IMAGE", "facebook photo mode skipped (minors involved); using generated image")
            return None

        if self.fetcher is None:
            logger.warn("IMAGE", "facebook photo mode unavailable (no fetcher); using generated image")
            return None

        try:
            photos = fetch_photos(article, self.fetcher, limit=2)

            if not photos:
                logger.warn("IMAGE", "facebook photo mode: no usable source photo; using generated image")
                return None

            canvas = compose(photos, self.cfg.facebook_layout)
            sha, ah = _save_jpeg(canvas, out_path)

            logger.log("IMAGE", f"facebook: photo layout built from {len(photos)} source photo(s)")

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
                strat, sty = "editorial", "editorial illustration"
            else:
                strat, sty = strategy, style

            prompt = build_prompt(strat, sty, v, scene_idea, title, aspect, simple=simple)

            try:
                logger.log(
                    "IMAGE",
                    f"{kind}: generating ({strat}, attempt {i}/{max_attempts}, ref={'yes' if r else 'no'})",
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
                        good, reason = fidelity_check(self.gem, r, payload)
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
            try:
                fb_sha, fb_ah, _ = self._one(
                    kind="facebook",
                    strategy=strategy,
                    style=style,
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
