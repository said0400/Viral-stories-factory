"""Replaceable image layer.

IMAGE_MODE=faithful (default): re-render the SOURCE photo as a cinematic version of the same scene.
IMAGE_MODE=creative: reference-aware editorial photojournalism.

Facebook uses source photos as references for newly generated high-resolution panels,
then composes a square, text-free image using a story-specific layout.
"""
from __future__ import annotations

import io
from abc import ABC, abstractmethod
from pathlib import Path

from PIL import Image, ImageDraw, ImageOps, ImageStat

from . import logger
from .cloudflare_client import CloudflareClient
from .config import Settings
from .gemini_client import GeminiClient, GeminiError, ImageGenError, ImageQuotaError
from .models import ImageCheckSchema, ImageResult, SourceArticle, VisualAnalysis
from .photo_composer import compose_square, fetch_photos, to_reference
from .utils import PoliteFetcher, sha256_hex
from .visual_analyzer import ahash, hamming

ARTICLE_ASPECT = "16:9"
FACEBOOK_ASPECT = "1:1"    # all Facebook panels and final composites are square

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
    subject_type = getattr(v, "subject_type", None) or "other"

    def people_strategy() -> tuple[str, str]:
        # Prefer a source reference for a real person instead of inventing a different face.
        if has_ref and v.contains_real_people:
            return "reference_identity", REALISTIC_NEWS_STYLE
        # The user explicitly opted out of anonymizing/illustrating visible people.
        return "people_safe", REALISTIC_NEWS_STYLE

    if subject_type == "person":
        return people_strategy()

    if subject_type in {"multiple_subjects", "event", "scene"} and v.contains_real_people:
        return people_strategy()

    if has_ref and v.reference_required:
        return "reference_identity", REALISTIC_NEWS_STYLE

    return "editorial", REALISTIC_NEWS_STYLE


def _faithful_prompt(
    style_text: str,
    aspect: str,
    simple: bool,
    title: str = "",
    scene_idea: str = "",
) -> str:
    parts = [
        f"Create a premium, high-resolution square 1:1 editorial photograph." if aspect == "1:1" else f"Create a premium, high-resolution {aspect} editorial hero photograph.",
        "The attached image is the SOURCE PHOTOGRAPH and the only authority for the real people and factual scene.",
        "Create a polished, believable editorial photojournalism image of the SAME verified moment, not a literal low-quality copy.",
        "Preserve each real person's recognizable identity, apparent age, face, expression, pose, hair, clothing, and all story-critical objects and setting.",
        "You MAY improve camera framing, crop, perspective, exposure, lighting, focus, and tonal balance to make the image more compelling and legible.",
        "Make the main person or subject large and immediately recognizable; keep eyes, faces, and story-critical details tack sharp.",
        "Use a clean, uncluttered composition with clear foreground/background separation, rich but natural color, realistic skin texture, balanced highlights and shadows, and professional lens rendering.",
        "Do not copy blur, low resolution, compression artifacts, dull exposure, or awkward cropping from the source.",
        "Do not add, remove, replace, or invent people, objects, events, or factual details; do not change who did what.",
        "Preserve legible, story-relevant physical signs already present in the source if possible; never invent or rewrite lettering.",
        f"Visual treatment: {style_text}.",
        "No text overlays, headlines, captions, watermarks, logos, arrows, decorative borders, collages, or graphic frames.",
    ]

    if title:
        parts.append(f"Editorial subject context (do not render as text): {title}.")
    if scene_idea:
        parts.append(f"Factual scene brief, subordinate to the reference photo: {scene_idea}.")

    if aspect == "1:1":
        parts.append("Use a bold mobile-first crop with one obvious focal subject; keep the subject clear even at small feed size.")
    else:
        parts.append("Use a strong horizontal hero composition with enough scene context, while keeping the main subject prominent and readable.")

    if not simple:
        parts.append("Avoid: " + NEGATIVE_FAITHFUL + ", tiny distant subjects, excessive shallow-focus blur, fog, heavy grain, muddy shadows, blown highlights, flat frontal lighting, plastic skin, awkward crop, busy background, unbalanced composition.")

    return " ".join(parts)


def build_prompt(
    strategy: str,
    style: str,
    v: VisualAnalysis,
    scene_idea: str,
    title: str,
    aspect: str,
    detail_scene_idea: str = "",
    composition_type: str = "SINGLE_HERO",
    simple: bool = False,
) -> str:
    """Build structural prompt based on exact prompt templates."""
    if strategy == FAITHFUL:
        return _faithful_prompt(style, aspect, simple, title=title, scene_idea=scene_idea)

    comp = str(composition_type or "SINGLE_HERO").upper()

    # EXACT SPLIT-PANEL PROMPT TEMPLATE requested by user
    if "DIPTYCH" in comp or "SPLIT" in comp:
        subject_a = scene_idea if scene_idea else "main subject of the story"
        subject_b = detail_scene_idea if detail_scene_idea else "related secondary subject or contrasting perspective of the story"

        return (
            "A high-definition, professional split-panel photograph, designed for a social media information card. "
            "The image is vertically divided into two distinct sections. "
            f"The left panel features a close-up photograph of {subject_a}. "
            f"The right panel features a close-up photograph of {subject_b}. "
            "Both panels are clean, free of any text, overlays, logos, or watermarks. "
            "The background is simple, ensuring the focus remains entirely on the subjects. "
            "Realistic camera lighting, sharp details, cinematic quality, photojournalism, no text."
        )

    base = [
        f"Square 1:1 authentic press news photograph for a story titled: '{title}'." if aspect == "1:1" else f"Aspect ratio {aspect} authentic press news photograph.",
        "CAMERA & STYLE: premium editorial photojournalism captured on a professional full-frame camera; tack-sharp focal subject, natural directional light, balanced exposure, authentic skin and material texture, controlled contrast, high detail, believable photographic depth.",
        "NO ADDED GRAPHICS: no headlines, captions, watermarks, logos, arrows, or artificial frames; preserve only legible story-relevant lettering already in the supplied reference.",
        f"SCENE DESCRIPTION: {scene_idea}. Make the main story subject prominent, visually distinct, and easy to understand at phone-feed size.",
    ]

    subject_type = getattr(v, "subject_type", None) or "other"

    if strategy == "reference_identity":
        keep = "; ".join(v.identity_features[:12]) if v.identity_features else "all clearly visible distinctive identity features"
        base += [
            "The attached image is a REFERENCE for the specific real subject described by the story.",
            f"Subject type: {subject_type}.",
            f"Preserve these identity-critical characteristics: {keep}.",
            "Realistic editorial photojournalistic photograph look.",
        ]

    elif strategy == "people_safe":
        base.append("Show visible faces naturally and clearly; do not obscure, blur, replace, or anonymize people.")

    if not simple:
        base.append("Avoid: " + NEGATIVE + ".")

    return " ".join(base)


# ------------------------------------------------------------------ PROGRAMMATIC INSET COMPOSITION
def create_circle_mask(size: int) -> Image.Image:
    mask = Image.new("L", (size * 4, size * 4), 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((0, 0, size * 4 - 1, size * 4 - 1), fill=255)
    return mask.resize((size, size), Image.LANCZOS)


def composite_inset(
    main_bytes: bytes,
    detail_bytes: bytes,
    shape: str = "INSET_CIRCLE",
) -> bytes:
    try:
        main_img = Image.open(io.BytesIO(main_bytes)).convert("RGB")
        detail_img = Image.open(io.BytesIO(detail_bytes)).convert("RGB")

        main_img = ImageOps.fit(main_img, (1080, 1080), Image.LANCZOS)

        inset_size = 345
        margin = 35
        border_width = 8

        detail_cropped = ImageOps.fit(detail_img, (inset_size, inset_size), Image.LANCZOS)

        x = 1080 - inset_size - margin
        y = margin

        if shape == "INSET_SQUARE":
            border_box = (x - border_width, y - border_width, x + inset_size + border_width, y + inset_size + border_width)
            draw = ImageDraw.Draw(main_img)
            draw.rectangle(border_box, fill=(240, 240, 240))
            main_img.paste(detail_cropped, (x, y))

        else:  # INSET_CIRCLE
            outer_size = inset_size + (border_width * 2)
            ring_img = Image.new("RGB", (outer_size, outer_size), (255, 204, 0))  # Yellow ring
            ring_mask = create_circle_mask(outer_size)
            inset_mask = create_circle_mask(inset_size)

            main_img.paste(ring_img, (x - border_width, y - border_width), ring_mask)
            main_img.paste(detail_cropped, (x, y), inset_mask)

        buf = io.BytesIO()
        main_img.save(buf, "JPEG", quality=90, optimize=True)
        return buf.getvalue()

    except Exception as exc:
        logger.warn("IMAGE", f"composite_inset failed ({exc}); returning main image")
        return main_bytes


# ------------------------------------------------------------------ validation
def validate_image(
    data: bytes,
    known: list[tuple[str, str]],
    source_ahash: str = "",
) -> tuple[bool, str, Image.Image | None]:
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
    img = img.convert("RGB")
    img.thumbnail((1920, 1920))

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
    try:
        r = gem.generate_json(
            (
                f"Story: {title}\n{summary}\n\n"
                "Evaluate ONLY the supplied image against the story for news publishing:\n"
                "1. Is the image clearly relevant to the story?\n"
                "2. Is it free of added text, captions, watermarks, logos, and artificial graphic overlays?\n"
                "3. Is the main subject large, immediately recognizable, well-framed, and in crisp focus?\n"
                "4. Does the exposure, lighting, color, and background look polished enough for a professional news/social feed?\n"
                "5. Reject if the subject is tiny, soft/blurry, poorly cropped, muddy, badly lit, or visibly low-resolution.\n"
                "6. Reject obvious anatomy defects, extra limbs, severe rendering artifacts, cartoons, or CGI."
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
    text_rule = (
        "contains_text_or_watermark = true only if Image 2 adds a headline, caption, watermark, logo, "
        "or lettering that is not present in Image 1. Do not flag genuine signs or labels already in Image 1."
        if forbid_text
        else "contains_text_or_watermark: not used, answer false."
    )

    try:
        r = gem.generate_json(
            (
                "Image 1 is the SOURCE photograph. Image 2 is a restyled version of it.\n"
                "Judge ONLY whether Image 2 is a faithful restyle of Image 1.\n"
                "relevant_to_story = true when Image 2 shows the same scene as Image 1.\n"
                f"{text_rule}\n"
                "obvious_defects = true for deformed faces/hands/anatomy OR a blurred, poorly exposed, badly cropped, low-detail, or amateur-looking result.\n"
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
        if self._provider is None:
            self._provider = build_provider(self.cfg, self.gem)
        return self._provider

    @provider.setter
    def provider(self, value: ImageProvider) -> None:
        # Public setter keeps provider substitution straightforward in tests and deployments.
        self._provider = value

    def _keep_rejected(self, data: bytes, name: str) -> None:
        if not self.cfg.dry_run or not data:
            return

        try:
            folder = self.cfg.export_path / "rejected"
            folder.mkdir(parents=True, exist_ok=True)
            (folder / name).write_bytes(data)
        except Exception:
            pass

    def _facebook_composite(
        self,
        *,
        story_id: str,
        article: SourceArticle,
        v: VisualAnalysis,
        title: str,
        scene: str,
        detail_scene: str,
        composition_type: str,
        source_ref: tuple[bytes, str] | None,
        known: list[tuple[str, str]],
        out_path: Path,
    ) -> tuple[str, str, str]:
        """Create AI-restyled components from source photos, then compose a text-free square."""
        photos = fetch_photos(article, self.fetcher, limit=3) if self.fetcher else []

        if source_ref:
            try:
                ref_img = Image.open(io.BytesIO(source_ref[0])).convert("RGB")
                ref_hash = ahash(ref_img)
                if not any(hamming(ref_hash, ahash(p)) <= 6 for p in photos):
                    photos.insert(0, ref_img)
            except Exception:
                logger.warn("IMAGE", "Facebook source reference could not be decoded; continuing with discovered images")

        refs = [to_reference(p) for p in photos[:3]]
        layout = str(self.cfg.facebook_layout or "auto").strip().lower()
        if layout == "auto":
            layout = str(composition_type or "inset_circle_right").strip().lower()
        layout = layout.replace("-", "_")
        number_needed = 3 if layout in {"triptych", "triptych_bottom"} else 2
        if not refs:
            logger.warn("IMAGE", "No source photo was usable; Facebook components will be generated from verified story context")

        component_images: list[Image.Image] = []
        component_hashes = list(known)
        component_paths: list[Path] = []
        generated_strategies: list[str] = []

        try:
            for index in range(number_needed):
                if index < len(refs):
                    ref = refs[index]
                    strategy = FAITHFUL
                    prompt_scene = scene if index == 0 else detail_scene or scene
                elif refs:
                    # If the publisher only supplies one image, create a second
                    # related framing from that reference rather than publishing it raw.
                    ref = refs[0]
                    strategy = "reference_identity"
                    prompt_scene = detail_scene or "A distinct, closer editorial view of the same story subject and setting."
                else:
                    ref = None
                    strategy = "editorial"
                    prompt_scene = scene if index == 0 else detail_scene or scene

                temp_path = self.cfg.images_dir / f"{story_id}_fb_component_{index}.jpg"
                component_paths.append(temp_path)
                source_hash = ahash(Image.open(io.BytesIO(ref[0])).convert("RGB")) if ref else ""
                sha, image_hash, used = self._one(
                    kind=f"facebook component {index + 1}",
                    strategy=strategy,
                    style=self.cfg.cinematic_style if strategy == FAITHFUL else REALISTIC_NEWS_STYLE,
                    v=v,
                    scene_idea=prompt_scene,
                    title=title,
                    aspect=FACEBOOK_ASPECT,
                    ref=ref,
                    known=component_hashes,
                    source_ahash="" if strategy == FAITHFUL else source_hash,
                    out_path=temp_path,
                    summary=v.summary or article.description,
                    composition_type="SINGLE_HERO",
                    # Compare generated text against reference text: preserve genuine
                    # scene signage, but reject added captions/watermarks.
                    forbid_text=strategy == FAITHFUL,
                )
                component_hashes.append((sha, image_hash))
                generated_strategies.append(used)
                component_images.append(Image.open(temp_path).convert("RGB"))

            canvas, used_count = compose_square(component_images, layout)
            digest, visual_hash = _save_jpeg(canvas, out_path)
            logger.log(
                "IMAGE",
                f"Facebook square ready: layout={layout}, generated_panels={used_count}, source_refs={len(refs)}",
            )
            return digest, visual_hash, "+".join(generated_strategies)
        finally:
            for path in component_paths:
                try:
                    path.unlink(missing_ok=True)
                except OSError:
                    pass

    def _one_raw(
        self,
        *,
        strategy: str,
        style: str,
        v: VisualAnalysis,
        scene_idea: str,
        title: str,
        aspect: str,
        ref: tuple[bytes, str] | None,
        detail_scene_idea: str = "",
        composition_type: str = "SINGLE_HERO",
    ) -> bytes:
        prompt = build_prompt(
            strategy, style, v, scene_idea, title, aspect, detail_scene_idea=detail_scene_idea, composition_type=composition_type, simple=False
        )
        data, _mime = self.provider.generate(prompt, [ref] if ref else None, aspect)
        return data

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
        detail_scene_idea: str = "",
        composition_type: str = "SINGLE_HERO",
        forbid_text: bool = False,
    ) -> tuple[str, str, str]:
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
                strat, sty, v, scene_idea, title, aspect, detail_scene_idea=detail_scene_idea, composition_type=composition_type, simple=simple
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
        facebook_detail_scene: str = "",
        facebook_composition_type: str = "INSET_CIRCLE_RIGHT",
        source_ref: tuple[bytes, str] | None,
        source_url: str,
        source_sha: str,
        source_ahash: str,
        known: list[tuple[str, str]],
    ) -> ImageResult:
        faithful = self.cfg.image_mode == "faithful" and source_ref is not None

        if faithful:
            strategy, style = FAITHFUL, self.cfg.cinematic_style
            article_ref = crop_to_aspect(source_ref, ARTICLE_ASPECT)
            facebook_ref = crop_to_aspect(source_ref, FACEBOOK_ASPECT)
            check_ahash = ""
            confidence = v.identity_confidence
            reference_used = True
        else:
            if self.cfg.image_mode == "faithful":
                logger.warn("IMAGE", "faithful mode not possible (no usable source image); using creative path")

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

        if self.cfg.facebook_separate_image:
            try:
                fb_sha, fb_ah, used = self._facebook_composite(
                    story_id=story_id,
                    article=article,
                    v=v,
                    title=title,
                    scene=facebook_scene or article_scene,
                    detail_scene=facebook_detail_scene,
                    composition_type=facebook_composition_type,
                    source_ref=source_ref,
                    known=[*known, (sha, ah)],
                    out_path=fb_out,
                )
                res.facebook_path = str(fb_out)
                res.facebook_image_hash = fb_sha
                res.facebook_image_ahash = fb_ah
                res.notes += f"; facebook_layout={self.cfg.facebook_layout or facebook_composition_type}; facebook_generation={used}"

            except (ImageQuotaError, ImageGenError) as exc:
                logger.warn("IMAGE", f"Facebook composite failed; creating a square fallback from the article image ({exc})")
                try:
                    article_img = Image.open(res.path).convert("RGB")
                    fallback, _ = compose_square([article_img, article_img], "inset_circle_right")
                    fb_sha, fb_ah = _save_jpeg(fallback, fb_out)
                    res.facebook_path = str(fb_out)
                    res.facebook_image_hash = fb_sha
                    res.facebook_image_ahash = fb_ah
                except Exception as fallback_exc:
                    logger.warn("IMAGE", f"Square fallback failed ({type(fallback_exc).__name__}); reusing article image")
                    res.facebook_path = res.path
                    res.facebook_image_hash = sha
                    res.facebook_image_ahash = ah

        else:
            res.facebook_path = res.path
            res.facebook_image_hash = sha
            res.facebook_image_ahash = ah

        return res
