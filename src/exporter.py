"""Downloadable Story Package Exporter.

Creates standalone, non-destructive ZIP bundles containing all generated text,
rendered HTML, images, preview HTML, and metadata for offline review/download.
"""
from __future__ import annotations

import html as html_lib
import json
import os
import shutil
import tempfile
import zipfile
from pathlib import Path

from bs4 import BeautifulSoup

from . import content as editorial
from . import logger
from .facebook import CTA_LINE
from .models import GeneratedContent, SourceArticle, StoryCache, StoryState
from .utils import iso


def _plain_text_from_html(raw_html: str) -> str:
    """Extract clean readable plain text from Blogger HTML."""
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "lxml")
    return soup.get_text("\n\n", strip=True)


def _render_preview_html(
    st: StoryState,
    cache: StoryCache,
    article: SourceArticle,
    content: GeneratedContent,
    has_article_img: bool,
    has_facebook_img: bool,
) -> str:
    """Generate a clean, standalone, mobile-responsive, XSS-safe HTML preview file."""
    blogger_url = (st.blogger_url or "").strip()
    blogger_url_display = (
        html_lib.escape(blogger_url)
        if blogger_url
        else "Not published yet (Offline Bundle)"
    )
    pub_status = html_lib.escape(st.status or "generated")

    raw_fb_post = content.facebook_post or ""
    raw_fb_comment = content.first_comment_hook or ""

    if cache.facebook:
        final_fb_post = cache.facebook.post
        final_fb_comment = cache.facebook.first_comment
    elif blogger_url:
        final_fb_post = f"{raw_fb_post}\n\n{CTA_LINE}\n{blogger_url}"
        final_fb_comment = f"{raw_fb_comment} 👇\n{blogger_url}"
    else:
        final_fb_post = f"{raw_fb_post}\n\n{CTA_LINE}\n{{BLOGGER_URL_NOT_PUBLISHED_YET}}"
        final_fb_comment = f"{raw_fb_comment} 👇\n{{BLOGGER_URL_NOT_PUBLISHED_YET}}"

    title_escaped = html_lib.escape(content.blogger_title or "")
    fb_title_escaped = html_lib.escape(content.facebook_title or "")
    source_name_escaped = html_lib.escape(article.source_name or "")
    source_url_escaped = html_lib.escape(article.original_url or "", quote=True)
    seo_desc_escaped = html_lib.escape(content.seo_description or "")
    labels_escaped = html_lib.escape(", ".join(content.labels or []))
    sanitized_blogger_body = editorial.sanitize_html(content.blogger_html or "")

    art_img_tag = (
        '<img src="images/article_image.jpg" class="preview-img" alt="صورة المقال">'
        if has_article_img
        else ''
    )
    fb_img_tag = (
        '<img src="images/facebook_image.jpg" class="preview-img" alt="صورة فيسبوك">'
        if has_facebook_img
        else ''
    )

    return f"""<!DOCTYPE html>
<html lang="ar" dir="rtl">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>معاينة القصة - {title_escaped}</title>
  <style>
    body {{ font-family: system-ui, -apple-system, sans-serif; line-height: 1.6; background: #f4f6f8; color: #222; margin: 0; padding: 20px; }}
    .container {{ max-width: 800px; margin: 0 auto; background: #fff; border-radius: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.08); padding: 30px; }}
    h1 {{ color: #1a252f; border-bottom: 2px solid #eee; padding-bottom: 10px; font-size: 1.8rem; }}
    .badge {{ display: inline-block; background: #27ae60; color: #fff; padding: 4px 10px; border-radius: 20px; font-size: 0.85rem; font-weight: bold; margin-bottom: 15px; }}
    .section {{ margin-bottom: 35px; border-top: 1px solid #eee; padding-top: 20px; }}
    .section-title {{ font-size: 1.3rem; color: #2c3e50; margin-bottom: 15px; display: flex; align-items: center; gap: 8px; }}
    .card {{ background: #f8f9fa; border: 1px solid #e9ecef; border-radius: 8px; padding: 18px; white-space: pre-wrap; font-size: 1rem; word-break: break-word; }}
    .meta-table {{ width: 100%; border-collapse: collapse; margin-top: 10px; }}
    .meta-table td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 0.9rem; }}
    .meta-table td:first-child {{ font-weight: bold; color: #555; width: 30%; }}
    img.preview-img {{ max-width: 100%; height: auto; border-radius: 8px; margin: 10px 0; }}
  </style>
</head>
<body>
  <div class="container">
    <span class="badge">الحالة: {pub_status}</span>
    <h1>{title_escaped}</h1>

    <div class="section">
      <div class="section-title">📰 محتوى المدونة (Blogger)</div>
      {art_img_tag}
      <div class="card">{sanitized_blogger_body}</div>
    </div>

    <div class="section">
      <div class="section-title">📲 منشور فيسبوك (Facebook)</div>
      {fb_img_tag}
      <div class="card"><strong>العنوان:</strong> {fb_title_escaped}\n\n{html_lib.escape(final_fb_post)}</div>
      <p><strong>التعليق الأول:</strong></p>
      <div class="card">{html_lib.escape(final_fb_comment)}</div>
    </div>

    <div class="section">
      <div class="section-title">⚙️ البيانات الوصفية (Metadata)</div>
      <table class="meta-table">
        <tr><td>المعرف (Story ID)</td><td>{html_lib.escape(st.story_id)}</td></tr>
        <tr><td>المصدر الأصلي</td><td><a href="{source_url_escaped}" target="_blank" rel="noopener">{source_name_escaped}</a></td></tr>
        <tr><td>رابط Blogger المنشور</td><td>{blogger_url_display}</td></tr>
        <tr><td>الوصف المخصص (SEO)</td><td>{seo_desc_escaped}</td></tr>
        <tr><td>الوسوم (Labels)</td><td>{labels_escaped}</td></tr>
      </table>
    </div>
  </div>
</body>
</html>
"""


def create_story_package(
    target_dir: Path,
    st: StoryState,
    cache: StoryCache,
    article: SourceArticle,
) -> Path | None:
    """Build a directory structure containing all individual files for a story."""
    if not cache or not cache.content:
        logger.warn("EXPORTER", f"no content cached for {st.story_id}; skipping export")
        return None

    content = cache.content
    folder_name = f"story_{st.story_id}"
    pkg_dir = target_dir / folder_name
    pkg_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1. Images Directory
    # -----------------------------------------------------------------------
    images_dir = pkg_dir / "images"
    images_dir.mkdir(exist_ok=True)

    has_article_img = False
    has_facebook_img = False

    if cache.image and cache.image.path and Path(cache.image.path).exists():
        try:
            shutil.copy2(cache.image.path, images_dir / "article_image.jpg")
            has_article_img = True
        except OSError as exc:
            logger.warn("EXPORTER", f"could not copy article image: {exc}")

    fb_img_source = (
        cache.image.facebook_path
        if (cache.image and cache.image.facebook_path and Path(cache.image.facebook_path).exists())
        else (cache.image.path if (cache.image and cache.image.path and Path(cache.image.path).exists()) else None)
    )

    if fb_img_source:
        try:
            shutil.copy2(fb_img_source, images_dir / "facebook_image.jpg")
            has_facebook_img = True
        except OSError as exc:
            logger.warn("EXPORTER", f"could not copy facebook image: {exc}")

    # -----------------------------------------------------------------------
    # 2. Blogger Directory
    # -----------------------------------------------------------------------
    blogger_dir = pkg_dir / "blogger"
    blogger_dir.mkdir(exist_ok=True)

    (blogger_dir / "title.txt").write_text(content.blogger_title, encoding="utf-8")
    (blogger_dir / "article_body.html").write_text(content.blogger_html, encoding="utf-8")

    # Image URL reference for full HTML: public URL if available, else relative path if image exists, else empty
    if cache.image and cache.image.public_url:
        image_ref_url = cache.image.public_url
    elif has_article_img:
        image_ref_url = "images/article_image.jpg"
    else:
        image_ref_url = ""

    full_html = editorial.render_blogger_html(
        content,
        article,
        image_ref_url,
        st.story_id,
    )
    (blogger_dir / "article_full.html").write_text(full_html, encoding="utf-8")
    (blogger_dir / "article_text.txt").write_text(_plain_text_from_html(content.blogger_html), encoding="utf-8")

    # -----------------------------------------------------------------------
    # 3. Facebook Directory (Separating Raw from Publishable Content)
    # -----------------------------------------------------------------------
    fb_dir = pkg_dir / "facebook"
    fb_dir.mkdir(exist_ok=True)

    (fb_dir / "title.txt").write_text(content.facebook_title, encoding="utf-8")

    # Raw LLM Output
    (fb_dir / "post.txt").write_text(content.facebook_post, encoding="utf-8")
    (fb_dir / "first_comment.txt").write_text(content.first_comment_hook, encoding="utf-8")

    # Final Promotion Copies (with Blogger URL if available)
    blogger_url = (st.blogger_url or "").strip()
    if cache.facebook:
        pub_post = cache.facebook.post
        pub_comment = cache.facebook.first_comment
    elif blogger_url:
        pub_post = f"{content.facebook_post}\n\n{CTA_LINE}\n{blogger_url}"
        pub_comment = f"{content.first_comment_hook} 👇\n{blogger_url}"
    else:
        pub_post = f"{content.facebook_post}\n\n{CTA_LINE}\n{{BLOGGER_URL_NOT_PUBLISHED_YET}}"
        pub_comment = f"{content.first_comment_hook} 👇\n{{BLOGGER_URL_NOT_PUBLISHED_YET}}"

    (fb_dir / "publishable_post.txt").write_text(pub_post, encoding="utf-8")
    (fb_dir / "publishable_first_comment.txt").write_text(pub_comment, encoding="utf-8")

    # -----------------------------------------------------------------------
    # 4. Interactive Preview HTML & Metadata
    # -----------------------------------------------------------------------
    preview_html = _render_preview_html(
        st, cache, article, content, has_article_img, has_facebook_img
    )
    (pkg_dir / "preview.html").write_text(preview_html, encoding="utf-8")

    meta_data = {
        "story_id": st.story_id,
        "source_name": article.source_name,
        "source_url": article.original_url,
        "original_title": article.original_title,
        "blogger_title": content.blogger_title,
        "blogger_url": st.blogger_url or "",
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
            "source_image_hash": cache.image.source_image_hash if cache.image else "",
            "generated_image_hash": cache.image.generated_hash if cache.image else "",
            "generated_image_ahash": cache.image.generated_ahash if cache.image else "",
            "facebook_image_hash": getattr(cache.image, "facebook_image_hash", "") if cache.image else "",
            "facebook_image_ahash": getattr(cache.image, "facebook_image_ahash", "") if cache.image else "",
        },
    }

    (pkg_dir / "metadata.json").write_text(
        json.dumps(meta_data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return pkg_dir


def export_bundle(
    exports_dir: Path,
    st: StoryState,
    cache: StoryCache,
    article: SourceArticle,
) -> Path | None:
    """Create a zipped story package under exports_dir/story_{story_id}.zip.

    Guaranteed non-destructive: catches exceptions and logs warnings without
    breaking the parent workflow execution.
    """
    try:
        exports_dir = Path(exports_dir)
        exports_dir.mkdir(parents=True, exist_ok=True)

        with tempfile.TemporaryDirectory(dir=str(exports_dir)) as tmp_staging:
            staging_path = Path(tmp_staging)
            pkg_dir = create_story_package(staging_path, st, cache, article)

            if not pkg_dir or not pkg_dir.exists():
                return None

            zip_filename = f"story_{st.story_id}.zip"
            final_zip_path = exports_dir / zip_filename
            tmp_zip_path = staging_path / zip_filename

            # Create zip file in staging
            with zipfile.ZipFile(
                tmp_zip_path, "w", compression=zipfile.ZIP_DEFLATED
            ) as zf:
                for file_path in pkg_dir.rglob("*"):
                    if file_path.is_file():
                        arcname = file_path.relative_to(pkg_dir)
                        zf.write(file_path, arcname)

            # True Atomic replace on the same filesystem
            os.replace(str(tmp_zip_path), str(final_zip_path))
            logger.log("EXPORTER", f"created bundle package: {final_zip_path.name}")

            return final_zip_path

    except Exception as exc:
        logger.warn("EXPORTER", f"bundle export failed for {st.story_id} ({type(exc).__name__}: {exc})")
        return None
