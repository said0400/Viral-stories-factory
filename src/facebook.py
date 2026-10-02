"""Facebook = traffic channel to Blogger. Prepares the promotion package only (no auto-post)."""
from __future__ import annotations

from urllib.parse import urlparse

from .models import FacebookPackage, GeneratedContent


class FacebookError(Exception):
    pass


def build_package(content: GeneratedContent, blogger_url: str, source_url: str,
                  image_path: str = "") -> FacebookPackage:
    """Only ever called AFTER Blogger returned the real URL."""
    u = urlparse(blogger_url or "")
    if u.scheme != "https" or not u.netloc:
        raise FacebookError("invalid Blogger URL")
    if blogger_url.rstrip("/") == (source_url or "").rstrip("/"):
        raise FacebookError("Blogger URL equals the source URL")
    if "{BLOGGER_URL}" in content.facebook_post or "http" in content.facebook_post:
        raise FacebookError("generated Facebook post must not contain URLs/placeholders")
    post = f"{content.facebook_post.strip()}\n\nالتفاصيل الكاملة وما حدث بعد ذلك تجدها في المقال 👇\n{blogger_url}"
    comment = f"{content.first_comment_hook.strip()} 👇\n{blogger_url}"
    if not content.facebook_title.strip() or len(content.facebook_post.strip()) < 40:
        raise FacebookError("Facebook content incomplete")
    return FacebookPackage(title=content.facebook_title.strip(), post=post,
                           first_comment=comment, image_path=image_path)
