"""Facebook = traffic channel to Blogger. Prepares the promotion package only (no auto-post)."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import FacebookPackage, GeneratedContent

CTA_LINE = "التفاصيل الكاملة وما حدث بعد ذلك تجدها في المقال 👇"


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

    try:
        hostname = u.hostname
    except ValueError:
        raise FacebookError("invalid Blogger URL: malformed hostname")

    if not hostname or not u.netloc:
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


def validate_facebook_content(content: GeneratedContent) -> None:
    """Validate raw LLM-generated Facebook content prior to attaching URLs."""
    facebook_post = (content.facebook_post or "").strip()
    facebook_title = (content.facebook_title or "").strip()
    first_comment_hook = (content.first_comment_hook or "").strip()

    if not facebook_title:
        raise FacebookError("Facebook title is empty")

    if len(facebook_post) < 40:
        raise FacebookError("Facebook content incomplete")

    if not first_comment_hook:
        raise FacebookError("Facebook first comment is empty")

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


def build_content_package(
    content: GeneratedContent,
    image_path: str = "",
) -> FacebookPackage:
    """
    Build a raw content package for Export/ZIP before Blogger publishing.
    Does NOT require a real Blogger URL.
    """
    validate_facebook_content(content)

    return FacebookPackage(
        title=content.facebook_title.strip(),
        post=content.facebook_post.strip(),
        first_comment=content.first_comment_hook.strip(),
        image_path=(image_path or "").strip(),
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
    This function never publishes anything directly to Facebook.
    """
    blogger_url = (blogger_url or "").strip()
    source_url = (source_url or "").strip()
    image_path = (image_path or "").strip()

    if not source_url:
        raise FacebookError("source URL is required")

    # ------------------------------------------------------------------
    # Blogger URL validation
    # ------------------------------------------------------------------
    _validate_blogger_url(blogger_url)

    if blogger_url.rstrip("/") == source_url.rstrip("/"):
        raise FacebookError("Blogger URL equals the source URL")

    # ------------------------------------------------------------------
    # Content validation
    # ------------------------------------------------------------------
    validate_facebook_content(content)

    facebook_post = content.facebook_post.strip()
    first_comment_hook = content.first_comment_hook.strip()

    if blogger_url in facebook_post:
        raise FacebookError("Facebook post already contains the Blogger URL")

    if blogger_url in first_comment_hook:
        raise FacebookError("Facebook comment already contains the Blogger URL")

    # ------------------------------------------------------------------
    # Final promotion package with standard CTA
    # ------------------------------------------------------------------
    post = f"{facebook_post}\n\n{CTA_LINE}\n{blogger_url}"
    comment = f"{first_comment_hook} 👇\n{blogger_url}"

    return FacebookPackage(
        title=content.facebook_title.strip(),
        post=post,
        first_comment=comment,
        image_path=image_path,
    )
