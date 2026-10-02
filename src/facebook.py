"""Facebook = traffic channel to Blogger. Prepares the promotion package only (no auto-post)."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import FacebookPackage, GeneratedContent


class FacebookError(Exception):
    pass


def _contains_url(text: str) -> bool:
    """Return True when text contains an explicit URL or URL-like placeholder."""
    if not text:
        return False

    value = text.strip()

    if "{BLOGGER_URL}" in value:
        return True

    if re.search(
        r"(?:https?://|www\.)\S+",
        value,
        re.IGNORECASE,
    ):
        return True

    return False


def _validate_blogger_url(blogger_url: str) -> None:
    """Validate that the supplied URL is a real HTTPS Blogger destination."""
    u = urlparse(
        blogger_url.strip()
    )

    if u.scheme.lower() != "https":
        raise FacebookError(
            "invalid Blogger URL: HTTPS is required"
        )

    if not u.netloc:
        raise FacebookError(
            "invalid Blogger URL: missing hostname"
        )

    if u.username or u.password:
        raise FacebookError(
            "invalid Blogger URL: credentials are not allowed"
        )

    if u.fragment:
        raise FacebookError(
            "invalid Blogger URL: fragments are not allowed"
        )


def build_package(
    content: GeneratedContent,
    blogger_url: str,
    source_url: str,
    image_path: str = "",
) -> FacebookPackage:
    """
    Only ever called AFTER Blogger returned the real URL.

    Facebook is used only as a traffic/promotion channel.
    This function never publishes anything to Facebook.
    """

    blogger_url = (blogger_url or "").strip()
    source_url = (source_url or "").strip()
    image_path = (image_path or "").strip()

    # ------------------------------------------------------------------
    # Blogger URL validation
    # ------------------------------------------------------------------
    _validate_blogger_url(
        blogger_url
    )

    # The promotion link must point to Blogger, never directly to the
    # original source article.
    if blogger_url.rstrip("/") == source_url.rstrip("/"):
        raise FacebookError(
            "Blogger URL equals the source URL"
        )

    # ------------------------------------------------------------------
    # Generated Facebook content validation
    # ------------------------------------------------------------------
    facebook_post = (
        content.facebook_post or ""
    ).strip()

    facebook_title = (
        content.facebook_title or ""
    ).strip()

    first_comment_hook = (
        content.first_comment_hook or ""
    ).strip()

    if not facebook_title:
        raise FacebookError(
            "Facebook title is empty"
        )

    if len(facebook_post) < 40:
        raise FacebookError(
            "Facebook content incomplete"
        )

    if not first_comment_hook:
        raise FacebookError(
            "Facebook first comment is empty"
        )

    # Gemini must never generate the Blogger URL itself.
    # The application appends the verified URL only after Blogger
    # successfully returns it.
    if _contains_url(facebook_post):
        raise FacebookError(
            "generated Facebook post must not contain URLs/placeholders"
        )

    if _contains_url(first_comment_hook):
        raise FacebookError(
            "generated Facebook comment must not contain URLs/placeholders"
        )

    if _contains_url(facebook_title):
        raise FacebookError(
            "generated Facebook title must not contain URLs"
        )

    # Prevent accidental duplication of the same CTA/link.
    if blogger_url in facebook_post:
        raise FacebookError(
            "Facebook post already contains the Blogger URL"
        )

    if blogger_url in first_comment_hook:
        raise FacebookError(
            "Facebook comment already contains the Blogger URL"
        )

    # ------------------------------------------------------------------
    # Final promotion package
    # ------------------------------------------------------------------
    post = (
        f"{facebook_post}\n\n"
        "التفاصيل الكاملة وما حدث بعد ذلك تجدها في المقال 👇\n"
        f"{blogger_url}"
    )

    comment = (
        f"{first_comment_hook} 👇\n"
        f"{blogger_url}"
    )

    return FacebookPackage(
        title=facebook_title,
        post=post,
        first_comment=comment,
        image_path=image_path,
    )
