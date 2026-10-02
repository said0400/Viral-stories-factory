"""Source image acquisition + visual analysis (identity vs scene) + safety policy flags."""
from __future__ import annotations

import io

from PIL import Image, ImageFile, UnidentifiedImageError

from . import logger
from .gemini_client import GeminiClient, GeminiError
from .models import SourceArticle, VisualAnalysis, VisualSchema
from .utils import FetchError, PoliteFetcher, sha256_hex

# Defend against decompression bombs before load().
Image.MAX_IMAGE_PIXELS = 40_000_000
ImageFile.LOAD_TRUNCATED_IMAGES = False

# Reject huge downloads early (bytes), before PIL work.
MAX_SOURCE_IMAGE_BYTES = 12 * 1024 * 1024

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

Separate:
- IDENTITY features that must stay the same when the subject is non-human
  (species/breed/markings, vehicle make/colour/damage, building/place architecture
  and landmarks, object shape/colour)
- SCENE features that may change (pose, angle, lighting, background, moment)

Be factual: describe only what is visible.

REAL PEOPLE:
- Never identify, name, or guess a real person's identity.
- Do not invent facial, biographical, or personal attributes.
- Do not request facial reconstruction or recognition.
- If people are present, note only that they are present and whether any appear
  to be children/minors.
- Optional non-sensitive context only when clearly visible: clothing type,
  approximate pose, accessories, surrounding place — never face identity.

subject_type must be one of:
person, place, animal, vehicle, object, building, event, scene,
multiple_subjects, other.

contains_real_people=true if any real human is visible.
involves_minors=true if any child/teen appears.
identity_confidence: low/medium/high = how well this single reference image
supports faithful recreation of a NON-HUMAN main subject without inventing
unsupported details. For person-primary images use low unless the task is
clearly non-identity scene context.
"""


def ahash(img: Image.Image) -> str:
    g = img.convert("L").resize((8, 8), Image.LANCZOS)
    px = list(g.tobytes())
    avg = sum(px) / 64

    return f"{sum((1 << i) for i, v in enumerate(px) if v >= avg):016x}"


def hamming(a: str, b: str) -> int:
    return bin(int(a, 16) ^ int(b, 16)).count("1")


def _prepare(data: bytes) -> tuple[bytes, str, str, str] | None:
    """
    Validate, cap size (<=1280px), return (jpeg_bytes, mime, sha, ahash).

    sha is computed on the ORIGINAL download bytes for traceability.
    ahash is computed on the normalized RGB thumbnail used as model input.
    """
    if not data:
        return None

    if len(data) > MAX_SOURCE_IMAGE_BYTES:
        return None

    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Image.DecompressionBombError:
        logger.warn(
            "VISUAL",
            "source image rejected (decompression bomb)",
        )
        return None
    except (UnidentifiedImageError, OSError):
        return None

    if min(img.size) < 300:
        return None

    try:
        img = img.convert("RGB")
        img.thumbnail((1280, 1280))

        buf = io.BytesIO()
        img.save(
            buf,
            "JPEG",
            quality=88,
            optimize=True,
        )
    except (Image.DecompressionBombError, OSError):
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

            # Early size gate from headers when available.
            cl = r.headers.get("Content-Length", "").strip()
            if cl.isdigit() and int(cl) > MAX_SOURCE_IMAGE_BYTES:
                logger.warn(
                    "VISUAL",
                    f"image skipped (Content-Length {cl} > cap)",
                )
                continue

            raw = r.content
            if len(raw) > MAX_SOURCE_IMAGE_BYTES:
                logger.warn(
                    "VISUAL",
                    "image skipped (body larger than cap)",
                )
                continue

            prepared = _prepare(raw)

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

    # Non-human subjects may keep reference identity without recreating faces.
    non_human_identity = st in REFERENCE_IDENTITY_TYPES

    # People are intentionally excluded from reference_identity here.
    # image_generator.py handles real people through the safe people strategy.
    identity_critical = (
        bool(res.identity_critical)
        and non_human_identity
    )

    contains_real_people = (
        bool(res.contains_real_people)
        or st in {"person", "multiple_subjects"}
    )

    # CHANGED: Now also requires identity_critical to be True.
    # We only restrict Gemini's creative freedom with a strict reference 
    # if the non-human identity is actually important to the story.
    reference_required = (
        bool(image)
        and identity_critical
        and not bool(res.involves_minors)
    )

    identity_confidence = (
        res.identity_confidence
        if image and non_human_identity
        else "low"
    )

    if identity_confidence not in {"low", "medium", "high"}:
        identity_confidence = "low"

    return VisualAnalysis(
        subject_type=st,
        identity_critical=identity_critical,
        contains_real_people=contains_real_people,
        involves_minors=bool(res.involves_minors),
        identity_features=list(res.identity_features or [])[:12],
        scene_features=list(res.scene_features or [])[:10],
        new_scene_direction=res.new_scene_direction or "",
        reference_required=reference_required,
        identity_confidence=identity_confidence,
        summary=res.summary or "",
    )
