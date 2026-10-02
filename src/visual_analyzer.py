"""Source image acquisition + visual analysis (identity vs scene) + safety policy flags."""
from __future__ import annotations

import io

from PIL import Image, UnidentifiedImageError

from . import logger
from .gemini_client import GeminiClient, GeminiError
from .models import SourceArticle, VisualAnalysis, VisualSchema
from .utils import FetchError, PoliteFetcher, sha256_hex

SUBJECT_TYPES = {
    "person",
    "place",
    "animal",
    "vehicle",
    "object",
    "building",
    "event",
    "scene",
    "multiple_subjects",
    "other",
}

REFERENCE_IDENTITY_TYPES = {
    "place",
    "animal",
    "vehicle",
    "object",
    "building",
}

VISUAL_SYSTEM = """You analyse a news/story photo to prepare a NEW illustration of the same story.
Separate IDENTITY features (must stay the same: species/breed/markings, vehicle make/colour/damage, building/place architecture & landmarks, object shape/colour) from SCENE features (pose, angle, lighting, background, moment) that may change.
Be factual: describe only what is visible. Never identify, name, or guess a real person's identity.
Do not invent facial, biographical, or personal attributes.
When the main subject is a real person and reference-preserving generation is permitted, describe only visible non-sensitive visual attributes that are actually supported by the reference image, such as clothing, hairstyle, accessories, pose, approximate presentation, and surrounding context. For people only note that they are present (and whether any appear to be children).
subject_type must be one of: person, place, animal, vehicle, object, building, event, scene, multiple_subjects, other.
contains_real_people=true if any real human is visible. involves_minors=true if any child/teen appears.
identity_confidence: low/medium/high = how well this single reference image supports faithful recreation of the main subject without inventing unsupported details."""


def ahash(img: Image.Image) -> str:
    g = img.convert("L").resize((8, 8), Image.LANCZOS)
    px = list(g.tobytes())
    avg = sum(px) / 64

    return f"{sum((1 << i) for i, v in enumerate(px) if v >= avg):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _prepare(data: bytes) -> tuple[bytes, str, str, str] | None:
    """Validate, cap size (<=1280px), return (jpeg_bytes, mime, sha, ahash)."""
    if not data:
        return None

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError):
        return None

    if min(img.size) < 300:
        return None

    img = img.convert("RGB")
    img.thumbnail((1280, 1280))

    buf = io.BytesIO()

    try:
        img.save(
            buf,
            "JPEG",
            quality=88,
            optimize=True,
        )
    except OSError:
        return None

    return (
        buf.getvalue(),
        "image/jpeg",
        sha256_hex(data),
        ahash(img),
    )


def acquire_source_image(
    article: SourceArticle,
    fetcher: PoliteFetcher,
) -> tuple[bytes, str, str, str, str] | None:
    """Return (bytes, mime, url, sha, ahash) for the best reachable image, or None."""
    urls = [
        u
        for u in [
            article.main_image_url,
            *article.additional_image_urls,
        ]
        if u
    ]

    seen: set[str] = set()

    for u in urls:
        u = str(u).strip()

        if not u or u in seen:
            continue

        seen.add(u)

        try:
            r = fetcher.get(u)
        except FetchError as exc:
            logger.warn(
                "VISUAL",
                f"image not fetched ({exc})",
            )
            continue

        try:
            content_type = (
                r.headers.get("Content-Type", "")
                .split(";", 1)[0]
                .strip()
                .lower()
            )

            if not content_type.startswith("image/"):
                continue

            prepared = _prepare(r.content)

            if prepared:
                b, m, sha, ah = prepared

                return (
                    b,
                    m,
                    u,
                    sha,
                    ah,
                )

        finally:
            r.close()

    return None


def analyze(
    gem: GeminiClient,
    article: SourceArticle,
    image: tuple[bytes, str] | None,
) -> VisualAnalysis:
    """Gemini vision when an image exists; text-only editorial analysis otherwise."""

    ctx = (
        f"STORY TITLE: {article.original_title}\n"
        f"STORY SUMMARY: {article.description[:600]}"
    )

    try:
        res = gem.generate_json(
            ctx
            + (
                "\nAnalyse the attached photo."
                if image
                else
                "\nNo photo is available; infer only from the text."
            ),
            VisualSchema,
            images=[image] if image else None,
            system=VISUAL_SYSTEM,
            temperature=0.2,
            tag="VISUAL",
        )

    except GeminiError as exc:
        logger.warn(
            "VISUAL",
            f"visual analysis failed ({exc}); using editorial defaults",
        )

        return VisualAnalysis(
            subject_type="scene",
            identity_critical=False,
            contains_real_people=False,
            involves_minors=False,
            identity_features=[],
            scene_features=[],
            new_scene_direction=(
                "editorial illustration of the story's scene"
            ),
            reference_required=False,
            identity_confidence="low",
            summary="Visual analysis unavailable; editorial fallback used.",
        )

    st = (
        res.subject_type
        if res.subject_type in SUBJECT_TYPES
        else "other"
    )

    # These are subjects for which preserving the specific source
    # identity can be meaningful without recreating a real person's face.
    non_human_identity = st in REFERENCE_IDENTITY_TYPES

    # People are intentionally excluded from reference_identity here.
    # image_generator.py handles real people through the safe people
    # strategy instead of attempting to recreate a real person's identity.
    identity_critical = (
        bool(res.identity_critical)
        and non_human_identity
    )

    contains_real_people = (
        bool(res.contains_real_people)
        or st in {"person", "multiple_subjects"}
    )

    # A reference image can be required for a specific non-human subject.
    # For real people, the image may still be used as contextual reference,
    # but it must not be treated as permission to recreate their face.
    reference_required = (
        bool(image)
        and non_human_identity
        and not res.involves_minors
    )

    identity_confidence = (
        res.identity_confidence
        if image and non_human_identity
        else "low"
    )

    return VisualAnalysis(
        subject_type=st,
        identity_critical=identity_critical,
        contains_real_people=contains_real_people,
        involves_minors=bool(res.involves_minors),
        identity_features=res.identity_features[:12],
        scene_features=res.scene_features[:10],
        new_scene_direction=res.new_scene_direction,
        reference_required=reference_required,
        identity_confidence=identity_confidence,
        summary=res.summary,
    )
