"""Facebook = traffic channel to Blogger. Prepares the promotion package only (no auto-post)."""
from __future__ import annotations

import re
from urllib.parse import urlparse

from .models import FacebookPackage, GeneratedContent

CTA_LINE = "التفاصيل الكاملة وما حدث بعد ذلك تجدها في المقال 👇"

MAX_FACEBOOK_POST_LENGTH = 63206
MAX_FACEBOOK_TITLE_LENGTH = 255
MAX_FACEBOOK_COMMENT_LENGTH = 10000


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


def _validate_http_url(
    value: str,
    label: str,
    require_https: bool = False,
) -> None:
    """Validate an absolute HTTP(S) URL."""
    value = (value or "").strip()

    if not value:
        raise FacebookError(
            f"{label} is empty"
        )

    try:
        parsed = urlparse(value)
    except ValueError:
        raise FacebookError(
            f"invalid {label}: malformed URL"
        )

    allowed_schemes = {"https"} if require_https else {"http", "https"}

    if parsed.scheme.lower() not in allowed_schemes:
        required = "HTTPS" if require_https else "HTTP/HTTPS"

        raise FacebookError(
            f"invalid {label}: {required} is required"
        )

    try:
        hostname = parsed.hostname
    except ValueError:
        raise FacebookError(
            f"invalid {label}: malformed hostname"
        )

    if not hostname or not parsed.netloc:
        raise FacebookError(
            f"invalid {label}: missing hostname"
        )

    if parsed.username or parsed.password:
        raise FacebookError(
            f"invalid {label}: credentials are not allowed"
        )


def _validate_blogger_url(blogger_url: str) -> None:
    """Validate that the supplied URL is a real HTTPS Blogger destination."""
    _validate_http_url(
        blogger_url,
        "Blogger URL",
        require_https=True,
    )

    u = urlparse(blogger_url.strip())

    if u.fragment:
        raise FacebookError(
            "invalid Blogger URL: fragments are not allowed"
        )


def _validate_source_url(source_url: str) -> None:
    """Validate that the supplied source URL is a real HTTP(S) URL."""
    _validate_http_url(
        source_url,
        "source URL",
        require_https=False,
    )


def _validate_lengths(content: GeneratedContent) -> None:
    """Protect against unexpectedly large generated Facebook fields."""
    facebook_post = (content.facebook_post or "").strip()
    facebook_title = (content.facebook_title or "").strip()
    first_comment_hook = (content.first_comment_hook or "").strip()

    if len(facebook_title) > MAX_FACEBOOK_TITLE_LENGTH:
        raise FacebookError(
            "Facebook title is too long"
        )

    if len(facebook_post) > MAX_FACEBOOK_POST_LENGTH:
        raise FacebookError(
            "Facebook content is too long"
        )

    if len(first_comment_hook) > MAX_FACEBOOK_COMMENT_LENGTH:
        raise FacebookError(
            "Facebook first comment is too long"
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

    _validate_lengths(content)

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

    # ------------------------------------------------------------------
    # URL validation
    # ------------------------------------------------------------------
    _validate_blogger_url(blogger_url)
    _validate_source_url(source_url)

    if blogger_url.rstrip("/") == source_url.rstrip("/"):
        raise FacebookError("Blogger URL equals the source URL")

    # ------------------------------------------------------------------
    # Content validation
    # ------------------------------------------------------------------
    validate_facebook_content(content)

    facebook_post = content.facebook_post.strip()
    first_comment_hook = content.first_comment_hook.strip()

    if blogger_url in facebook_post:
        raise FacebookError(
            "Facebook post already contains the Blogger URL"
        )

    if blogger_url in first_comment_hook:
        raise FacebookError(
            "Facebook comment already contains the Blogger URL"
        )

    # ------------------------------------------------------------------
    # Final promotion package with standard CTA
    # ------------------------------------------------------------------
    post = (
        f"{facebook_post}\n\n"
        f"{CTA_LINE}\n"
        f"{blogger_url}"
    )

    comment = (
        f"{first_comment_hook} 👇\n"
        f"{blogger_url}"
    )

    return FacebookPackage(
        title=content.facebook_title.strip(),
        post=post,
        first_comment=comment,
        image_path=image_path,
    )
