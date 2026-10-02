"""Downloadable Story Package Exporter (non-destructive ZIP bundles)."""
from __future__ import annotations

import html as html_lib
import json
import os
import re
import shutil
import tempfile
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from . import content as editorial
from . import logger
from .facebook import CTA_LINE
from .models import GeneratedContent, SourceArticle, StoryCache, StoryState
from .utils import iso

_SAFE_STORY_ID_RE = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_PLACEHOLDER = "{BLOGGER_URL_NOT_PUBLISHED_YET}"

_CSS = """
body { font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; background: #f4f6f8; color: #222; margin: 0; padding: 20px; }
.container { max-width: 800px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); padding: 30px; }
h1 { color: #1a252f; border-bottom: 2px solid #eee; padding-bottom: 10px; font-size: 1.8rem; }
.badge { display: inline-block; background: #27ae60; color: #fff; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-bottom: 15px; }
.section { margin-bottom: 35px; border-top: 1px solid #eee; padding-top: 20px; }
.section-title { font-size: 1.3rem; color: #2c3e50; margin-bottom: 15px; }
.card { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 18px; white-space: pre-wrap; word-break: break-word; }
.article { background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 18px; word-break: break-word; }
.meta-table { width: 100%; border-collapse: collapse; margin-top: 10px; }
.meta-table td { padding: 8px; border-bottom: 1px solid #eee; font-size: 0.9rem; }
.meta-table td:first-child { font-weight: bold; color: #555; width: 30%; }
img.preview-img { max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }
"""


def _validate_story_id(story_id: str) -> str:
    value = (story_id or "").strip()

    if not value:
        raise ValueError("story_id is empty")

    if not _SAFE_STORY_ID_RE.fullmatch(value):
        raise ValueError("story_id contains unsafe filesystem characters")

    return value


def _safe_http_url(raw_url: str) -> str:
    value = (raw_url or "").strip()

    if not value:
        return ""

    try:
        parsed = urlparse(value)
    except ValueError:
        return ""

    if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
        return ""

    if parsed.username is not None or parsed.password is not None:
        return ""

    return value


def _is_file(path: str | None) -> bool:
    return bool(path) and Path(str(path)).is_file()


def _plain_text_from_html(raw_html: str) -> str:
    if not raw_html:
        return ""

    return BeautifulSoup(raw_html, "lxml").get_text("\n\n", strip=True)


def _facebook_texts(st: StoryState, cache: StoryCache, content: GeneratedContent) -> tuple[str, str]:
    """Publishable Facebook post/comment. The cached package is used only if it already holds the real URL."""
    url = _safe_http_url(st.blogger_url)

    if cache.facebook and url and url in (cache.facebook.post or ""):
        return cache.facebook.post, cache.facebook.first_comment

    target = url or _PLACEHOLDER

    return (
        f"{content.facebook_post}\n\n{CTA_LINE}\n{target}",
        f"{content.first_comment_hook} 👇\n{target}",
    )


def _render_preview_html(
    st: StoryState,
    cache: StoryCache,
    article: SourceArticle,
    content: GeneratedContent,
    has_article_img: bool,
    has_facebook_img: bool,
) -> str:
    esc = html_lib.escape

    blogger_url = _safe_http_url(st.blogger_url)
    blogger_url_display = esc(blogger_url) if blogger_url else "Not published yet (Offline Bundle)"

    post, comment = _facebook_texts(st, cache, content)

    source_url = _safe_http_url(article.original_url)
    source_name = esc(article.source_name or "")

    source_link = (
        f'<a href="{esc(source_url, quote=True)}" target="_blank" rel="noopener noreferrer">{source_name}</a>'
        if source_url
        else source_name
    )

    art_img = '<img src="images/article_image.jpg" class="preview-img" alt="صورة المقال">' if has_article_img else ""
    fb_img = '<img src="images/facebook_image.jpg" class="preview-img" alt="صورة فيسبوك">' if has_facebook_img else ""

    title = esc(content.blogger_title or "")
    body = editorial.sanitize_html(content.blogger_html or "")

    return (
        '<!DOCTYPE html>\n<html lang="ar" dir="rtl">\n<head>\n'
        '  <meta charset="UTF-8">\n'
        '  <meta name="viewport" content="width=device-width, initial-scale=1.0">\n'
        f"  <title>معاينة القصة - {title}</title>\n"
        f"  <style>{_CSS}</style>\n</head>\n<body>\n"
        '  <div class="container">\n'
        f'    <span class="badge">الحالة: {esc(st.status or "generated")}</span>\n'
        f"    <h1>{title}</h1>\n"
        '    <div class="section">\n'
        '      <div class="section-title">📰 محتوى المدونة (Blogger)</div>\n'
        f'      {art_img}\n      <div class="article">{body}</div>\n    </div>\n'
        '    <div class="section">\n'
        '      <div class="section-title">📲 منشور فيسبوك (Facebook)</div>\n'
        f"      {fb_img}\n"
        f'      <div class="card"><strong>العنوان:</strong> {esc(content.facebook_title or "")}\n\n{esc(post)}</div>\n'
        "      <p><strong>التعليق الأول:</strong></p>\n"
        f'      <div class="card">{esc(comment)}</div>\n    </div>\n'
        '    <div class="section">\n'
        '      <div class="section-title">⚙️ البيانات الوصفية (Metadata)</div>\n'
        '      <table class="meta-table">\n'
        f"        <tr><td>المعرف (Story ID)</td><td>{esc(st.story_id)}</td></tr>\n"
        f"        <tr><td>المصدر الأصلي</td><td>{source_link}</td></tr>\n"
        f"        <tr><td>رابط Blogger المنشور</td><td>{blogger_url_display}</td></tr>\n"
        f'        <tr><td>الوصف المخصص (SEO)</td><td>{esc(content.seo_description or "")}</td></tr>\n'
        f'        <tr><td>الوسوم (Labels)</td><td>{esc(", ".join(content.labels or []))}</td></tr>\n'
        "      </table>\n    </div>\n  </div>\n</body>\n</html>\n"
    )


def create_story_package(
    target_dir: Path,
    st: StoryState,
    cache: StoryCache,
    article: SourceArticle,
) -> Path | None:
    """Build the directory structure containing all files for a story."""
    if not cache or not cache.content:
        logger.warn("EXPORTER", f"no content cached for {st.story_id}; skipping export")
        return None

    story_id = _validate_story_id(st.story_id)
    content = cache.content
    image = cache.image

    pkg_dir = target_dir / f"story_{story_id}"
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # 1. images ------------------------------------------------------------
    images_dir = pkg_dir / "images"
    images_dir.mkdir(exist_ok=True)

    has_article_img = False
    has_facebook_img = False

    if image and _is_file(image.path):
        try:
            shutil.copy2(image.path, images_dir / "article_image.jpg")
            has_article_img = True
        except OSError as exc:
            logger.warn("EXPORTER", f"could not copy article image: {exc}")

    fb_source = None

    if image:
        if _is_file(image.facebook_path):
            fb_source = image.facebook_path
        elif _is_file(image.path):
            fb_source = image.path

    if fb_source:
        try:
            shutil.copy2(fb_source, images_dir / "facebook_image.jpg")
            has_facebook_img = True
        except OSError as exc:
            logger.warn("EXPORTER", f"could not copy facebook image: {exc}")

    # 2. blogger -----------------------------------------------------------
    blogger_dir = pkg_dir / "blogger"
    blogger_dir.mkdir(exist_ok=True)

    clean_body = editorial.sanitize_html(content.blogger_html or "")

    (blogger_dir / "title.txt").write_text(content.blogger_title, encoding="utf-8")
    (blogger_dir / "article_body.html").write_text(clean_body, encoding="utf-8")

    public_image_url = _safe_http_url(image.public_url) if image else ""

    if public_image_url:
        image_ref = public_image_url
    elif has_article_img:
        image_ref = "../images/article_image.jpg"
    else:
        image_ref = ""

    (blogger_dir / "article_full.html").write_text(
        editorial.render_blogger_html(content, article, image_ref, story_id),
        encoding="utf-8",
    )
    (blogger_dir / "article_text.txt").write_text(_plain_text_from_html(clean_body), encoding="utf-8")

    # 3. facebook ----------------------------------------------------------
    fb_dir = pkg_dir / "facebook"
    fb_dir.mkdir(exist_ok=True)

    pub_post, pub_comment = _facebook_texts(st, cache, content)

    (fb_dir / "title.txt").write_text(content.facebook_title, encoding="utf-8")
    (fb_dir / "post.txt").write_text(content.facebook_post, encoding="utf-8")
    (fb_dir / "first_comment.txt").write_text(content.first_comment_hook, encoding="utf-8")
    (fb_dir / "publishable_post.txt").write_text(pub_post, encoding="utf-8")
    (fb_dir / "publishable_first_comment.txt").write_text(pub_comment, encoding="utf-8")

    # 4. source ------------------------------------------------------------
    source_dir = pkg_dir / "source"
    source_dir.mkdir(exist_ok=True)

    (source_dir / "source.txt").write_text(
        f"Source: {article.source_name}\n"
        f"Original title: {article.original_title}\n"
        f"Original URL: {article.original_url}\n",
        encoding="utf-8",
    )
    (source_dir / "source_url.txt").write_text(article.original_url or "", encoding="utf-8")

    # 5. preview + metadata ------------------------------------------------
    (pkg_dir / "preview.html").write_text(
        _render_preview_html(st, cache, article, content, has_article_img, has_facebook_img),
        encoding="utf-8",
    )

    meta = {
        "story_id": story_id,
        "source_name": article.source_name,
        "source_url": article.original_url,
        "original_title": article.original_title,
        "blogger_title": content.blogger_title,
        "blogger_url": _safe_http_url(st.blogger_url),
        "status": st.status,
        "export_status": st.export_status or "ready",
        "facebook_status": st.facebook_status or "pending",
        "whatsapp_status": st.whatsapp_status or "pending",
        "image_status": st.image_status or "none",
        "subject_type": st.subject_type or "",
        "identity_confidence": st.identity_confidence or "low",
        "generated_at": st.generated_at or "",
        "image_generated_at": st.image_generated_at or "",
        "published_at": st.published_at or "",
        "export_created_at": iso(),
        "seo_description": content.seo_description,
        "labels": content.labels,
        "hashes": {
            "source_image_hash": image.source_image_hash if image else "",
            "generated_image_hash": image.generated_hash if image else "",
            "generated_image_ahash": image.generated_ahash if image else "",
            "facebook_image_hash": image.facebook_image_hash if image else "",
            "facebook_image_ahash": image.facebook_image_ahash if image else "",
        },
    }

    (pkg_dir / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    return pkg_dir


def export_bundle(
    exports_dir: Path,
    st: StoryState,
    cache: StoryCache,
    article: SourceArticle,
) -> Path | None:
    """Create exports_dir/story_{story_id}.zip. Never raises: failures are logged and None is returned."""
    try:
        exports_dir = Path(exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)

        story_id = _validate_story_id(st.story_id)

        with tempfile.TemporaryDirectory(dir=str(exports_dir)) as tmp:
            staging = Path(tmp)
            pkg_dir = create_story_package(staging, st, cache, article)

            if not pkg_dir or not pkg_dir.exists():
                return None

            zip_name = f"story_{story_id}.zip"
            final_zip = exports_dir / zip_name
            tmp_zip = staging / zip_name

            with zipfile.ZipFile(tmp_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                for file_path in pkg_dir.rglob("*"):
                    if file_path.is_file():
                        zf.write(file_path, file_path.relative_to(pkg_dir))

            os.replace(str(tmp_zip), str(final_zip))

            logger.log("EXPORTER", f"created bundle package: {final_zip.name}")

            return final_zip

    except Exception as exc:
        logger.warn("EXPORTER", f"bundle export failed for {st.story_id} ({type(exc).__name__}: {exc})")
        return None
