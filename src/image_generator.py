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


NEGATIVE = (
    "unrelated person, different animal, different vehicle, different building, "
    "generic location, duplicate composition, copied source photograph, "
    "distorted face, recognizable real person's face, extra limbs, "
    "deformed anatomy, random text, captions, watermark, logo, "
    "low quality, blurry subject, duplicate subject, invented factual details"
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

    def generate(
        self,
        prompt,
        references,
        aspect_ratio,
    ):
        return self.gem.generate_image(
            prompt,
            references=references,
            aspect_ratio=aspect_ratio,
        )


# ------------------------------------------------------------------
# strategy
# ------------------------------------------------------------------
def choose_strategy(
    v: VisualAnalysis,
    has_ref: bool,
    people_style: str,
) -> tuple[str, str]:
    """
    Return (strategy, style).

    Rules:
    - Real people are never recreated photorealistically.
    - A photo containing incidental people can still be used as a reference
      for a non-human identity-critical subject.
    - Reference identity is used only when VisualAnalysis explicitly marks
      the reference as required.
    """

    subject_type = (
        v.subject_type
        if getattr(v, "subject_type", None)
        else "other"
    )

    people_style = (
        people_style
        if people_style in {
            "reference",
            "illustration",
            "faceless",
        }
        else "illustration"
    )

    # --------------------------------------------------------------
    # Real people as the primary subject
    # --------------------------------------------------------------
    if subject_type == "person":
        style = (
            "faceless"
            if v.involves_minors
            else people_style
        )

        # "reference" is NOT used for photorealistic identity recreation
        # of real people. It is converted to a safe illustrative treatment.
        if style == "reference":
            style = "illustration"

        return "people_safe", style

    # --------------------------------------------------------------
    # Multiple subjects / scenes containing people
    # --------------------------------------------------------------
    if subject_type in {
        "multiple_subjects",
        "event",
        "scene",
    } and v.contains_real_people:
        # If the important identity is non-human and the analyzer explicitly
        # requires a reference, preserve that non-human subject even when
        # people happen to appear in the source image.
        if (
            has_ref
            and v.reference_required
            and v.identity_critical
        ):
            return "reference_identity", "photorealistic"

        style = (
            "faceless"
            if v.involves_minors
            else people_style
        )

        if style == "reference":
            style = "illustration"

        return "people_safe", style

    # --------------------------------------------------------------
    # Non-human identity-critical subjects
    # --------------------------------------------------------------
    if (
        has_ref
        and v.reference_required
        and v.identity_critical
    ):
        return "reference_identity", "photorealistic"

    # --------------------------------------------------------------
    # Reference requested but identity confidence is insufficient
    # --------------------------------------------------------------
    if (
        has_ref
        and v.reference_required
    ):
        return "reference_identity", "photorealistic"

    # --------------------------------------------------------------
    # Safe editorial fallback
    # --------------------------------------------------------------
    return "editorial", "editorial illustration"


def build_prompt(
    strategy: str,
    style: str,
    v: VisualAnalysis,
    scene_idea: str,
    title: str,
    aspect: str,
    simple: bool = False,
) -> str:
    """
    Build a provider-neutral image prompt.

    The prompt explicitly separates:
    - identity-critical information
    - scene/composition information
    - safety restrictions around real people
    """

    base = [
        f'Create ONE high-quality image for an article titled: "{title}".',
        f"Aspect ratio {aspect}.",
        "Strong visual hierarchy.",
        "Clear main subject.",
        "No text, no captions, no watermark, no logo.",
        "Create a NEW composition rather than copying the source image.",
    ]

    scene = (
        v.new_scene_direction
        or scene_idea
        or "a visually clear editorial scene related to the story"
    )

    subject_type = (
        v.subject_type
        if getattr(v, "subject_type", None)
        else "other"
    )

    # --------------------------------------------------------------
    # Reference-aware identity generation
    # --------------------------------------------------------------
    if strategy == "reference_identity":
        keep = (
            "; ".join(v.identity_features[:12])
            if v.identity_features
            else "all clearly visible distinctive identity features"
        )

        base += [
            "The attached image is a REFERENCE for the specific real "
            "subject described by the story.",
            f"Subject type: {subject_type}.",
            f"Preserve these identity-critical characteristics: {keep}.",
            (
                "Use the SAME specific non-human subject when the reference "
                "supports its identity. Do not replace it with a generic "
                "animal, vehicle, building, place or object."
            ),
            f"Create a NEW scene based on: {scene_idea}.",
            f"Additional scene direction: {scene}.",
            (
                "Change camera angle, composition, framing, lighting and/or "
                "moment. Do NOT copy the original photograph."
            ),
            (
                "Do not create a similar-looking substitute subject. "
                "Preserve the specific subject's supported identity."
            ),
            (
                "Do not invent factual details that are not supported by "
                "the story or visible reference."
            ),
            "Realistic cinematic photography look.",
        ]

        if v.contains_real_people:
            base += [
                (
                    "People may appear only as incidental contextual elements "
                    "unless the story establishes them as the subject."
                ),
                (
                    "Do not identify, reconstruct or reproduce the face of "
                    "any real person."
                ),
            ]

    # --------------------------------------------------------------
    # People-safe generation
    # --------------------------------------------------------------
    elif strategy == "people_safe":
        if style == "faceless":
            base += [
                f"Scene: {scene_idea}.",
                (
                    "Show people only from behind, in silhouette, in shadow, "
                    "cropped without faces, or sufficiently far away that "
                    "faces are not visible."
                ),
                (
                    "No recognizable or reconstructed real person's face."
                ),
                (
                    "Focus on the place, objects, atmosphere and supported "
                    "context of the story."
                ),
                (
                    "Do not invent clothing, facial features, expressions, "
                    "age appearance, ethnicity or other personal attributes."
                ),
            ]
        else:
            base += [
                (
                    "Stylised editorial DIGITAL ILLUSTRATION, clearly not "
                    "a photograph."
                ),
                f"Scene: {scene_idea}.",
                (
                    "Any people must be generic stylised figures with "
                    "simplified non-identifying features."
                ),
                (
                    "They must NOT resemble any real individual."
                ),
                (
                    "Do not depict or reconstruct a real person's face."
                ),
                (
                    "Do not invent factual details about the people."
                ),
            ]

        if v.scene_features:
            base.append(
                "Supported setting cues only: "
                + "; ".join(v.scene_features[:6])
                + "."
            )

    # --------------------------------------------------------------
    # Editorial illustration
    # --------------------------------------------------------------
    else:
        base += [
            f"Editorial illustration of: {scene_idea}.",
            (
                "The image is illustrative and must not claim to reproduce "
                "a real event exactly."
            ),
            (
                "Use only factual elements supported by the story."
            ),
            (
                "Avoid inventing specific people, objects, locations, "
                "architecture, clothing, weather, injuries or actions."
            ),
        ]

    # --------------------------------------------------------------
    # Universal restrictions
    # --------------------------------------------------------------
    if v.involves_minors:
        base += [
            (
                "If minors are present, do not show identifiable faces. "
                "Use distant, rear-view, silhouette or non-identifying "
                "depiction."
            ),
        ]

    if v.contains_real_people:
        base += [
            (
                "Never guess or reconstruct facial identity from the "
                "reference."
            ),
        ]

    if not simple:
        base.append(
            "Avoid: " + NEGATIVE + "."
        )

    return " ".join(base)


# ------------------------------------------------------------------
# validation
# ------------------------------------------------------------------
def validate_image(
    data: bytes,
    known: list[tuple[str, str]],
    source_ahash: str = "",
) -> tuple[bool, str, Image.Image | None]:
    """Validate generated image integrity, dimensions and duplication."""
    if not data:
        return False, "empty file", None

    try:
        img = Image.open(
            io.BytesIO(data)
        )
        img.load()
    except Exception:
        return False, "cannot open/corrupt", None

    if min(img.size) < 512:
        return False, f"too small {img.size}", None

    # Reject obviously empty / nearly uniform images.
    g = img.convert("L")
    st = ImageStat.Stat(g)

    if st.mean[0] < 8 or st.stddev[0] < 6:
        return False, "black/blank image", None

    generated_ahash = ahash(img)

    # The generated image must not simply reproduce the source image.
    if (
        source_ahash
        and hamming(
            generated_ahash,
            source_ahash,
        ) <= 3
    ):
        return False, "too similar to the source image", None

    # Avoid duplicates against earlier generated images.
    for _, known_ahash in known:
        if (
            known_ahash
            and hamming(
                generated_ahash,
                known_ahash,
            ) <= 2
        ):
            return (
                False,
                "duplicate of an earlier generated image",
                None,
            )

    return True, "", img


def _save_jpeg(
    img: Image.Image,
    path: Path,
) -> tuple[str, str]:
    """Save a normalized JPEG under the WhatsApp size limit."""
    img = img.convert("RGB")

    img.thumbnail(
        (1600, 1600)
    )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    q = 88

    while True:
        buf = io.BytesIO()

        img.save(
            buf,
            "JPEG",
            quality=q,
            optimize=True,
        )

        # Stay under WhatsApp's 5 MB image limit.
        if (
            buf.tell() < 4_500_000
            or q <= 60
        ):
            break

        q -= 8

    payload = buf.getvalue()

    path.write_bytes(
        payload
    )

    return (
        sha256_hex(payload),
        ahash(img),
    )


def vlm_check(
    gem: GeminiClient,
    jpeg: bytes,
    title: str,
    summary: str,
) -> tuple[bool, str]:
    """Use Gemini structured vision validation as a second quality gate."""
    try:
        r = gem.generate_json(
            (
                f"Story: {title}\n"
                f"{summary}\n\n"
                "Evaluate ONLY the supplied image against the story.\n"
                "Is the image relevant to the story?\n"
                "Is it free of visible text, watermarks and logos?\n"
                "Does it have obvious visual defects such as deformed "
                "hands, faces, anatomy or severe rendering artifacts?\n"
                "Do not reject an image merely because it is an illustration."
            ),
            ImageCheckSchema,
            images=[
                (
                    jpeg,
                    "image/jpeg",
                )
            ],
            temperature=0.0,
            tag="IMGCHECK",
        )

        ok = (
            r.relevant_to_story
            and not r.contains_text_or_watermark
            and not r.obvious_defects
        )

        return ok, r.reason

    except GeminiError:
        # Image generation should not fail solely because the optional
        # secondary VLM check is unavailable.
        return True, "check unavailable (skipped)"


# ------------------------------------------------------------------
# main entry
# ------------------------------------------------------------------
class ImageGenerator:
    def __init__(
        self,
        cfg: Settings,
        gem: GeminiClient,
        provider: ImageProvider | None = None,
    ) -> None:
        self.cfg = cfg
        self.gem = gem
        self.provider = (
            provider
            or GeminiImageProvider(gem)
        )

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
        Attempts:

        1. Full prompt + reference.
        2. Simpler prompt + reference.
        3. Simple prompt without reference.

        If identity-critical generation loses its reference, it falls back
        to editorial illustration and explicitly does NOT claim identity
        preservation.
        """

        plans = (
            [
                (False, ref),
                (True, ref),
                (True, None),
            ]
            if ref
            else [
                (False, None),
                (True, None),
            ]
        )

        max_attempts = max(
            1,
            self.cfg.max_retries + 1,
        )

        for i, (simple, r) in enumerate(
            plans,
            1,
        ):
            if i > max_attempts:
                break

            # A reference_identity strategy without an actual reference
            # must never claim that the generated subject is the same
            # real-world subject.
            if (
                strategy == "reference_identity"
                and r is None
            ):
                strat = "editorial"
                sty = "editorial illustration"
            else:
                strat = strategy
                sty = style

            prompt = build_prompt(
                strat,
                sty,
                v,
                scene_idea,
                title,
                aspect,
                simple=simple,
            )

            try:
                logger.log(
                    "IMAGE",
                    (
                        f"{kind}: generating "
                        f"({strat}/{sty}, "
                        f"attempt {i}, "
                        f"ref={'yes' if r else 'no'})"
                    ),
                )

                data, _ = self.provider.generate(
                    prompt,
                    [r] if r else None,
                    aspect,
                )

            except (
                ImageGenError,
                GeminiError,
            ) as exc:
                logger.warn(
                    "IMAGE",
                    f"{kind}: attempt {i} failed ({exc})",
                )
                continue

            ok, why, img = validate_image(
                data,
                known,
                source_ahash,
            )

            if not ok:
                logger.warn(
                    "IMAGE",
                    f"{kind}: rejected ({why})",
                )
                continue

            if img is None:
                logger.warn(
                    "IMAGE",
                    f"{kind}: rejected (no decoded image)",
                )
                continue

            try:
                sha, ah = _save_jpeg(
                    img,
                    out_path,
                )
            except Exception as exc:
                logger.warn(
                    "IMAGE",
                    f"{kind}: failed to save image ({type(exc).__name__})",
                )
                continue

            if self.cfg.image_vlm_check:
                try:
                    good, reason = vlm_check(
                        self.gem,
                        out_path.read_bytes(),
                        title,
                        summary,
                    )
                except Exception as exc:
                    logger.warn(
                        "IMAGE",
                        f"{kind}: VLM check failed ({type(exc).__name__}); "
                        "continuing",
                    )
                    good = True
                    reason = "check unavailable (skipped)"

                if not good:
                    logger.warn(
                        "IMAGE",
                        f"{kind}: relevance check failed ({reason})",
                    )

                    # Do not leave a rejected image behind.
                    try:
                        out_path.unlink(
                            missing_ok=True
                        )
                    except Exception:
                        pass

                    continue

            logger.log(
                "IMAGE",
                f"{kind}: validation passed",
            )

            return (
                sha,
                ah,
                f"{strat}/{sty}",
            )

        raise ImageGenError(
            f"{kind}: no valid image after attempts"
        )

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
        Generate the main article image and optionally a separate Facebook image.
        """

        strategy, style = choose_strategy(
            v,
            bool(source_ref),
            self.cfg.people_image_style,
        )

        # Identity preservation is claimed only when:
        # 1. strategy is reference_identity,
        # 2. a real reference exists,
        # 3. the analyzer marked the reference as identity-relevant.
        confidence = (
            v.identity_confidence
            if (
                strategy == "reference_identity"
                and source_ref is not None
                and v.reference_required
            )
            else "low"
        )

        out = (
            self.cfg.images_dir
            / f"{story_id}_generated.jpg"
        )

        sha, ah, used = self._one(
            kind="article",
            strategy=strategy,
            style=style,
            v=v,
            scene_idea=article_scene,
            title=title,
            aspect="16:9",
            ref=(
                source_ref
                if strategy == "reference_identity"
                else None
            ),
            known=known,
            source_ahash=source_ahash,
            out_path=out,
            summary=(
                v.summary
                or article.description
            ),
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
                "Identity preservation depends on provider/model "
                f"capabilities; used={used}; "
                f"reference_used={'yes' if source_ref and strategy == 'reference_identity' else 'no'}"
            ),
        )

        # --------------------------------------------------------------
        # Optional separate Facebook image
        # --------------------------------------------------------------
        if self.cfg.facebook_separate_image:
            fb_out = (
                self.cfg.images_dir
                / f"{story_id}_facebook.jpg"
            )

            try:
                self._one(
                    kind="facebook",
                    strategy=strategy,
                    style=style,
                    v=v,
                    scene_idea=facebook_scene,
                    title=title,
                    aspect="1:1",
                    ref=(
                        source_ref
                        if strategy == "reference_identity"
                        else None
                    ),
                    known=[
                        *known,
                        (sha, ah),
                    ],
                    source_ahash=source_ahash,
                    out_path=fb_out,
                    summary=(
                        v.summary
                        or article.description
                    ),
                )

                res.facebook_path = str(
                    fb_out
                )

            except ImageGenError as exc:
                logger.warn(
                    "IMAGE",
                    "facebook image failed; "
                    f"reusing article image ({exc})",
                )

                res.facebook_path = res.path

        else:
            res.facebook_path = res.path

        return res
