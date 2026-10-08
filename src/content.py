"""Groq/Gemini editorial steps: triage, Arabic content generation, and fact checking."""
from __future__ import annotations

import html as html_lib
import re
from typing import Callable
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

from . import logger
from .gemini_client import GeminiClient
from .models import (
    ArticleBodySchema,
    EditorialMetadataSchema,
    FactCheckSchema,
    GeneratedContent,
    SourceArticle,
    ThumbnailBriefSchema,
    TriageItem,
    TriageResult,
)

ALLOWED_TAGS = {
    "h2",
    "h3",
    "p",
    "ul",
    "ol",
    "li",
    "strong",
    "em",
    "blockquote",
    "br",
}

TRIAGE_SYSTEM = """You are an editor selecting REAL, verifiable, curiosity-driven stories for a general Arabic audience.

Prefer:
- strange, surprising, human, animal, mysterious, funny, unusual or rare stories
- bizarre incidents
- odd places
- intriguing history
- unusual discoveries
- stories with a surprising ending

De-prioritise:
- politics
- partisan news
- war
- graphic crime
- graphic violence
- celebrity gossip
- tragedies exploiting victims
- anything sexual
- medical advice
- stories involving minors in distress
- unverifiable claims
- stories whose main claim cannot be supported by the supplied information

Judge ONLY from the title and description provided.
Never invent facts.
Do not infer facts that are not explicitly supported.
Scores are 0-100 and internal only.

IMPORTANT:
- already_published=true only when the candidate describes the same real-world EVENT as a previously published story.
- Do not mark a story as already published merely because it has a similar topic.
- duplicate_of must point only to an EARLIER candidate index describing the same event.
- If there is no genuine duplicate, use -1.
- event_key should identify the same real-world event using a few concise English words.
- If the source is ambiguous or insufficient to verify the core event, prefer suitable=false."""

CONTENT_SYSTEM = """You are a professional Arabic editor-writer. You write ORIGINAL articles from verified source facts.

ABSOLUTE RULES

1. Use ONLY facts found in the SOURCE TEXT.
Never invent names, ages, dates, numbers, places, quotes, statements, outcomes, motives, feelings of real people, medical information, causes, relationships or other factual details.
Do not infer popularity, audience size, virality, an account's quality, a photo's quality, visual details absent from SOURCE TEXT, or why a person/animal acted. Never use unsupported superlatives or claims such as “millions”.
Do not attribute plans, motives, or human-like intentions to animals. Use a clearly marked subjective impression only when it cannot be mistaken for a reported fact.

2. If something is unclear, write:
'بحسب التقرير'
'وفقًا للمصدر'
'لم يتضح'
'لم يذكر المصدر'
or another natural uncertainty marker.
Never convert uncertainty into certainty.

3. Language:
Use simple modern Arabic understood by all Arabs.
Use short, clear, natural sentences.
No literal translation.
No stiff journalese.
No filler.
No repetitive phrasing.

4. ORIGINALITY:
Do NOT copy or line-by-line paraphrase the source.
Change the structure, opening and narrative order while preserving every factual claim.
Original writing must never mean adding new facts.

5. blogger_html:
Return valid HTML using only:
<h2>, <h3>, <p>, <ul>, <li>, <strong>, <em>, <blockquote>.

Do NOT use:
<h1>
<img>
<a>
<div>
<table>
<script>
<style>
or any other HTML element.

Do not include an image.
Do not include a source/attribution line because the system adds it later.

Structure:
- Open with a vivid, highly engaging hook in the first paragraph, but never invent suspense or facts.
- Build a complete narrative: what happened, the verified context, key details, and what is known now.
- Use at least 2 meaningful <h2> subheadings (and <h3> only for useful subsections) when the source contains enough material.
- Use short, connected paragraphs; explain unfamiliar context only when the source supports it.
- End with a satisfying, clear takeaway based only on available information.

Target 450-900 Arabic words when the source contains enough facts.
If the source is thin, write less.
NEVER pad the article to reach a word count.

6. blogger_title (the article's main headline):
- honest
- intriguing
- emotionally compelling and curiosity-driven
- not misleading
- no fabricated information
- <= 90 characters
- do not use sensational claims that the source does not support

7. facebook_title (caption headline, NOT text to render inside the image):
- short
- strong
- mobile friendly
- vivid, punchy, and curiosity-driving like a strong social headline
- does not reveal everything
- does not invent facts
- does not use false clickbait

8. facebook_post (caption body, always separate from the image):
Structure: HOOK -> INTEREST -> PARTIAL CONTEXT -> CURIOSITY. Make it read like a vivid description of the actual story, with a natural open loop and a clear reason to read the article. The publishing system will place facebook_title above this body and append facebook_hashtags so they appear as ONE complete Facebook post.

Do NOT include any URL or website placeholder.
Do NOT reveal the entire story or main twist.
Do NOT fabricate suspense.
Do NOT claim that something happened unless the source confirms it.
Vary openings and avoid stock phrases.

9. first_comment_hook:
One short, natural sentence inviting readers to learn the complete story and what happened next. The system places the real Blogger article URL directly below this sentence.
No URL.
No fabricated claim.

10. facebook_hashtags:
Return 3-5 concise, relevant hashtags (each as one token beginning with #). Use Arabic where natural, and English only for a genuinely useful topic tag. No generic viral/reach claims, unrelated tags, spaces, or URLs.

11. labels:
Return 2-5 Arabic labels genuinely related to the story.
Examples:
قصص غريبة
قصص حقيقية
غرائب
أخبار طريفة
اكتشافات

Do not add unrelated SEO labels.

12. facebook_composition_type MUST be strictly one of these square layouts made only from original source photos:
- "INSET_CIRCLE_RIGHT" / "INSET_CIRCLE_LEFT": one large main photo with a small round secondary photo in the chosen upper corner.
- "INSET_SQUARE_RIGHT" / "INSET_SQUARE_LEFT": same composition with a square inset.
- "DIPTYCH_SPLIT": two distinct photos side by side; best for two people, perspectives, or moments.
- "DIPTYCH_STACK": two photos stacked; best for a clear before/after or sequence.
- "TRIPTYCH": one large vertical panel beside two smaller stacked panels (three images).
- "TRIPTYCH_BOTTOM": two small square panels above one wide lower panel (three images).
- "QUAD_GRID": four story-relevant photos in a balanced 2×2 grid.
- "SINGLE_HERO": only when the story genuinely has one compelling visual; otherwise prefer at least two photos.
Choose the layout that best communicates the verified story at a glance. The Facebook image is assembled from original article images only; do not request newly generated or AI-restyled pictures. The hook and headline belong in facebook_title/facebook_post, outside the image.

13. facebook_scene_idea & facebook_detail_scene_idea:
- Give concise descriptions of the two most visually important, source-supported details that would help select original article photos.
- These descriptions are analysis hints only. Never invent a new scene or use them to request an AI-generated/repainted image.
- If the official article has only one usable photo, the image pipeline may show a magnified crop of that same original photo as a secondary detail.

STRICT VISUAL RULES:
- The square Facebook image must use only original images downloaded from the official article; the image model must not generate, repaint, alter, or replace any photo.
- The compositor may only crop, resize, zoom, and arrange the original pixels. Do not add text or graphics to the image; keep the hook in facebook_post.
- Never remove or rewrite source-native signs, captions, arrows, logos, watermarks, or other marks: they are part of the original photograph. The visual analyzer chooses relevant material; it does not edit pixels.
- Use ONLY factual elements supported by the source text or visible in the original article photos.

14. REAL PEOPLE:
For the AI-generated article hero only, when a source photo is supplied, keep the depicted person's recognizable face, apparent age, clothing, and key identity cues consistent with that photo. Facebook photos are original source pixels and are not generated. Describe only factual/contextual information supported by the source; do not invent identity, biography, motives, or sensitive attributes.

15. PEOPLE OF ANY AGE:
Do not blur, mask, anonymize, or intentionally hide a person's face when it is visible in a reference for the AI-generated article image. The image model may still vary facial details; never claim a perfect identity match.

16. SOURCE IMAGE:
The Facebook visual is assembled from the actual photos found in the official article. Prefer the most story-relevant distinct photos and identify a focal crop for enlargement, while keeping useful surrounding context. Do not synthesize missing image content.

17. NON-HUMAN SUBJECTS:
For animals, vehicles, buildings, places, objects, preserve important identifying characteristics supported by the source or reference image.

18. UNCERTAINTY:
If the source does not establish a detail, omit it or clearly mark it as uncertain.

19. FACTUAL PRIORITY:
Accuracy is more important than drama.

20. SEO:
Derive a primary search phrase from SOURCE TEXT. Use it naturally in blogger_title, the opening or a relevant heading, and seo_description without keyword stuffing. Use clear descriptive H2/H3 headings, useful context, and a natural Arabic meta description of <=155 characters. Never invent search volume, rankings, or facts.

21. FINAL CONSISTENCY:
All fields must remain consistent with the same source facts."""

ARTICLE_STYLE_PROMPT = """اكتب بالعربية البسيطة المفهومة لمعظم القراء العرب، بصوت تحريري طبيعي ودافئ.
نوّع أطوال الجمل والإيقاع، واختر مفردات دقيقة وغير مكررة من دون تعقيد مصطنع أو حشو.
تجنب العبارات المستهلكة مثل: «في الآونة الأخيرة»، «علاوة على ذلك»، «في الختام»، «تجدر الإشارة»، «من الجدير بالذكر»، «يمثل»، و«يعتبر».
اختر نبرة مناسبة للموضوع تلقائيًا: ودية أو تعليمية أو حماسية، من دون افتعال.
اجعل العربية الفصحى المبسطة هي الأساس، مع لمسات خليجية سعودية وقطرية مألوفة وخفيفة جدًا عند ملاءمتها طبيعيًا (مثل: وش، شنو، وايد، مرة)؛ لا تكدّس اللهجات ولا تجعل النص محليًا بحيث يصعب على بقية العرب فهمه.
لا تدّعِ أنك إنسان أو أنك جرّبت شيئًا، ولا تختلق تجربة شخصية بصيغة المتكلم؛ استخدم المتكلم فقط إذا كانت تجربة الكاتب مثبتة في المصدر.
ابدأ بهوك صادق وجذاب من دون اختلاق تشويق، ثم نظّم المقال بعناوين <h2> و<h3> عند الحاجة وفقرات قصيرة واضحة. استخرج عبارة البحث الأهم من المصدر، وأدرجها بسلاسة في العنوان/المقدمة وعنوان فرعي مناسب ووصف SEO، مع مرادفات طبيعية ومن دون تكرار آلي.
قدّم للقارئ قيمة وسياقًا أو خلاصة عملية مما يثبته المصدر، ولا تملأ المقال بمعلومات خارج النص المصدر.
اكتب مقالًا كاملًا ومترابطًا، لكن لا تحشُ الكلام للوصول إلى طول محدد إذا كان المصدر قصيرًا."""

ARTICLE_BODY_SYSTEM = """تصرّف ككاتب محتوى عربي محترف وخبير SEO ذي خبرة تحريرية طويلة. مهمتك الوحيدة كتابة متن المقال داخل blogger_html.
التزم بمعلومات SOURCE TEXT فقط. لا تختلق أسماء أو أرقامًا أو أعمارًا أو اقتباسات أو أسبابًا أو دوافع أو تجربة شخصية. لا تفترض شهرةً أو عدد متابعين أو مشاهدات أو جودة صور أو نوايا بشرية للحيوانات إذا لم يذكرها المصدر صراحة.
استخدم HTML صالحًا بهذه الوسوم فقط: <h2>, <h3>, <p>, <ul>, <ol>, <li>, <strong>, <em>, <blockquote>, <br>.
لا تكتب عنوان المقال أو وصف SEO أو منشور Facebook؛ ستُنشأ هذه الحقول منفصلة.
عندما يحتوي المصدر على مادة كافية، استخدم مقدمة جذابة بلا عنوان «مقدمة»، وعنوانين فرعيين <h2> على الأقل، وفقرات قصيرة، ونهاية واضحة مفيدة.
استهدف 450-900 كلمة فقط إذا كان المصدر غنيًا بما يكفي؛ اكتب أقل عندما تكون الحقائق محدودة ولا تحشو المقال.
إذا كان التفصيل غير مؤكد، انسبه إلى التقرير أو قل إن المصدر لم يوضحه.
""" + ARTICLE_STYLE_PROMPT

EDITORIAL_METADATA_SYSTEM = """أنت محرر عربي ومدقق للمعلومات. أنشئ العنوان الرئيسي ووصف SEO والتصنيفات وعناوين ونصوص Facebook اعتمادًا حصريًا على SOURCE TEXT ومتن المقال المرفق.
لا تضف واقعة أو رقمًا أو اسمًا غير موجود في المصدر، ولا تجعل الفضول تضليلًا أو clickbait كاذبًا.
لا تخترع أرقام المشاهدات أو المتابعين أو الشعبية أو جودة الصور أو دوافع الأشخاص والحيوانات. إذا لم يثبت المصدر التفصيل، احذفه بدل تخمينه.
blogger_title واضح وجذاب ولا يتجاوز 90 حرفًا. seo_description لا يتجاوز 155 حرفًا.
facebook_title قصير وقوي ومناسب للهاتف؛ سيظهر مرة واحدة في بداية النص المنشور. facebook_post هو المتن فقط: هوك ثم اهتمام وسياق جزئي وفضول صادق، ولا يكشف كل القصة ولا يحتوي رابطًا. أعد facebook_hashtags من 3 إلى 5 وسوم دقيقة، ليضمها النظام إلى المنشور نفسه بعد العنوان والمتن.
اكتب العنوان والمتن بعبارة عربية بسيطة مفهومة لمعظم العرب، مع لمسة خليجية طبيعية وخفيفة فقط حين تلائم القصة. اجعل الهوك مؤثرًا ومثيرًا للفضول لكن صادقًا وغير مبالغ فيه.
first_comment_hook جملة قصيرة تدعو بوضوح إلى معرفة تفاصيل القصة كاملة وما حدث بعدها؛ سيضيف النظام رابط المقال الحقيقي مباشرة بعدها. لا تضع رابطًا أو ادعاء غير مسند.
labels من 2 إلى 5 تصنيفات عربية ذات صلة.
اختر facebook_composition_type من: INSET_CIRCLE_RIGHT, INSET_CIRCLE_LEFT, INSET_SQUARE_RIGHT, INSET_SQUARE_LEFT, DIPTYCH_SPLIT, DIPTYCH_STACK, TRIPTYCH, TRIPTYCH_BOTTOM, QUAD_GRID, SINGLE_HERO.
اكتب article_scene_idea وfacebook_scene_idea وfacebook_detail_scene_idea كإشارات تحليلية موجزة مدعومة بالمصدر فقط؛ لا تطلب توليد صورة Facebook.
لا تعِد كتابة blogger_html؛ أعد حقول البيانات التحريرية فقط وفق المخطط."""

FACT_SYSTEM = """You are a strict fact checker.

Compare the article headline/body/headings, SEO description, Facebook title/post/hashtags/first comment, and all scene ideas against the verifiable facts in the SOURCE TEXT.

Flag every checkable factual assertion that is not supported, especially numbers, popularity/reach claims, causal claims, motives, identity, and claims about an account or image.
Do not flag an unmistakable figure of speech or clearly subjective impression as a factual claim unless it also asserts a concrete event or detail. Do not excuse unsupported numbers or factual-sounding claims as “style”.
When flagging a problem, quote or identify the smallest exact claim that needs removal or correction.

Return only claims that genuinely require correction.
If everything is supported, return an empty unsupported_claims list and all_claims_supported=true."""

THUMBNAIL_SYSTEM = """You are an art director writing ONE precise image-generation prompt (in English) for a professional
YouTube-style thumbnail of a real news/curiosity story. The image model also receives the real photos from the article as references.

Rules:
- Base every visual element ONLY on the SOURCE TEXT. Never invent people, objects, places, events or details.
- Describe the single most compelling, factual moment: the main subject (person/animal/object), their visible expression or pose,
  the setting, key supporting objects, camera angle, lens feel, lighting, color mood and depth of field.
- Landscape 16:9 composition, one dominant subject placed large and sharp, uncluttered background, strong focal separation,
  clean contrast, instantly readable at small size, photorealistic editorial photojournalism look.
- Keep real faces, apparent age, clothing and identity cues consistent with the reference photos.
- The image must contain absolutely NO text, letters, numbers, captions, logos, watermarks, arrows, borders or graphic overlays.
- 90-160 words, one paragraph, no markdown, no URLs, no mention of 'thumbnail text'."""

_URL_RE = re.compile(r"(?:https?://|www\.)\S+", re.IGNORECASE)
_PLACEHOLDER_RE = re.compile(r"\{BLOGGER_URL\}")
_DISCLOSURE = "الصورة المرفقة معدَّلة أو مُولَّدة بالذكاء الاصطناعي لأغراض توضيحية."


# ---------------------------------------------------------------- sanitising
def sanitize_html(raw: str) -> str:
    """Allow-list HTML sanitiser (no attributes, no unknown tags, no comments)."""
    if not raw:
        return ""

    soup = BeautifulSoup(str(raw), "lxml")

    for comment in soup.find_all(string=lambda v: isinstance(v, Comment)):
        comment.extract()

    for tag in soup.find_all(
        [
            "script", "style", "iframe", "object", "embed", "form", "img", "a",
            "meta", "link", "base", "svg", "math", "video", "audio", "source", "picture",
        ]
    ):
        if tag.name == "a":
            tag.unwrap()
        else:
            tag.decompose()

    for tag in list(soup.find_all(True)):
        name = str(tag.name or "").lower()

        if name in ("html", "head", "body"):
            tag.unwrap()
            continue

        if name == "h1":
            tag.name = "h2"
            name = "h2"

        if name not in ALLOWED_TAGS:
            tag.unwrap()
            continue

        tag.attrs = {}

    body = soup.body

    if body is not None:
        return "".join(str(child) for child in body.contents).strip()

    return "".join(str(child) for child in soup.contents).strip()


def _strip_urls(text: str) -> str:
    value = _PLACEHOLDER_RE.sub("", str(text or ""))
    value = _URL_RE.sub("", value)
    value = re.sub(r"[ \t]{2,}", " ", value)
    return value.strip()


def _normalize_hashtags(values: list[str] | None) -> list[str]:
    """Keep short, single-token hashtags and normalize their leading #."""
    out: list[str] = []
    for raw in values or []:
        token = re.sub(r"[^\w]", "", str(raw or "").strip().lstrip("#"), flags=re.UNICODE)
        if len(token) < 2:
            continue
        hashtag = "#" + token
        if hashtag not in out:
            out.append(hashtag)
    return out[:5]


def _safe_story_id(story_id: str) -> str:
    value = str(story_id or "").strip()

    if not value:
        raise ValueError("story_id is required")

    if len(value) > 128:
        raise ValueError("story_id is too long")

    if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise ValueError("story_id contains unsupported characters")

    return value


def _safe_source_host(source_url: str) -> str:
    host = (urlparse(source_url or "").hostname or "").strip().lower()

    if not host:
        return "source"

    if host.startswith("www."):
        host = host[4:]

    return html_lib.escape(host, quote=True)


def _safe_source_url(source_url: str) -> str:
    value = str(source_url or "").strip()

    if not value:
        return ""

    try:
        parsed = urlparse(value)
    except ValueError:
        return ""

    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        return ""

    if parsed.username or parsed.password:
        return ""

    return value


def _safe_image_src(value: str) -> str:
    src = str(value or "").strip()

    if not src:
        return ""

    if src.startswith(("data:image/jpeg;base64,", "data:image/png;base64,")):
        return src

    if src.startswith("../images/"):
        return src

    try:
        parsed = urlparse(src)
    except ValueError:
        return ""

    if parsed.scheme in ("http", "https") and parsed.netloc:
        return src

    return ""


def render_blogger_html(
    content: GeneratedContent,
    article: SourceArticle,
    image_url: str,
    story_id: str,
) -> str:
    safe_story_id = _safe_story_id(story_id)
    body = sanitize_html(content.blogger_html)

    img_src = _safe_image_src(image_url)
    img = ""

    if img_src:
        img = (
            '<figure style="margin:0 0 1.2em;text-align:center">'
            f'<img src="{html_lib.escape(img_src, quote=True)}" '
            f'alt="{html_lib.escape(content.blogger_title, quote=True)}" '
            'style="max-width:100%;height:auto" loading="lazy"/></figure>'
        )

    disclosure = f"<p><em>{_DISCLOSURE}</em></p>" if img else ""

    source_url = _safe_source_url(article.original_url)

    if source_url:
        src = (
            "<p><strong>المصدر:</strong> "
            f'<a href="{html_lib.escape(source_url, quote=True)}" '
            'rel="nofollow noopener" target="_blank">'
            f"{html_lib.escape(article.source_name)}</a> "
            f"({_safe_source_host(source_url)})</p>"
            f"{disclosure}"
        )
    else:
        src = (
            f"<p><strong>المصدر:</strong> {html_lib.escape(article.source_name or 'المصدر')}</p>"
            f"{disclosure}"
        )

    return (
        '<div dir="rtl" style="text-align:right">'
        f"{img}{body}{src}"
        "</div>"
        f"<!-- story_id:{html_lib.escape(safe_story_id, quote=False)} -->"
    )


def triage(
    gem: GeminiClient,
    candidates: list[SourceArticle],
    published_titles: list[str],
) -> list[TriageItem]:
    if not candidates:
        return []

    lines = []

    for i, c in enumerate(candidates):
        age = c.publication_date.date().isoformat() if c.publication_date else "unknown"
        lines.append(f"[{i}] ({c.source_name}, {age}) {c.original_title} — {c.description[:220]}")

    pub = "\n".join(f"- {t}" for t in published_titles[:60] if str(t).strip()) or "(none)"

    prompt = (
        "CANDIDATE STORIES:\n"
        + "\n".join(lines)
        + "\n\nALREADY PUBLISHED:\n"
        + pub
        + "\n\n"
        "For each candidate return an item."
    )

    res = gem.generate_json(prompt, TriageResult, system=TRIAGE_SYSTEM, temperature=0.2, tag="TRIAGE")

    out: dict[int, TriageItem] = {}

    for it in res.items:
        if not 0 <= it.index < len(candidates):
            continue

        for field in (
            "viral_score",
            "curiosity_score",
            "share_score",
            "comment_score",
            "originality_score",
            "emotional_score",
        ):
            try:
                value = int(getattr(it, field, 0))
            except (TypeError, ValueError):
                value = 0

            setattr(it, field, max(0, min(100, value)))

        try:
            duplicate_of = int(getattr(it, "duplicate_of", -1))
        except (TypeError, ValueError):
            duplicate_of = -1

        if duplicate_of < 0 or duplicate_of >= len(candidates) or duplicate_of >= it.index:
            duplicate_of = -1

        it.duplicate_of = duplicate_of

        event_key = str(getattr(it, "event_key", "") or "").strip()
        it.event_key = event_key[:160].rstrip()

        out[it.index] = it

    return list(out.values())


def rank_score(item: TriageItem) -> float:
    return (
        0.35 * item.viral_score
        + 0.25 * item.curiosity_score
        + 0.15 * item.share_score
        + 0.10 * item.comment_score
        + 0.10 * item.originality_score
        + 0.05 * item.emotional_score
    )


MAX_FIX_ROUNDS = 3

# Keep the FACTCHECK request small enough for Groq's tokens-per-minute limit (Arabic is token-heavy).
FACT_SOURCE_CHARS = 5500
FACT_ARTICLE_CHARS = 6500


def _clip(text: str, limit: int = 220) -> str:
    return " ".join(str(text or "").split())[:limit]


def _claims(check: FactCheckSchema) -> list[str]:
    return [str(c).strip() for c in (check.unsupported_claims or []) if str(c).strip()]


def _fact_check_passes(check: FactCheckSchema) -> bool:
    """Reject contradictory provider output instead of trusting an empty list alone."""
    return bool(check.all_claims_supported) and not _claims(check)


def _quality_issues(article: SourceArticle, content: GeneratedContent) -> list[str]:
    """Lightweight structure checks; never demand padding from thin source material."""
    issues: list[str] = []
    if not content.blogger_title.strip() or len(content.blogger_title.strip()) > 90:
        issues.append("Keep the main article headline concise (90 characters or fewer).")
    if not content.facebook_title.strip() or len(content.facebook_title.strip()) > 100:
        issues.append("Write a short, compelling Facebook caption headline (100 characters or fewer).")
    if not 3 <= len(content.facebook_hashtags) <= 5:
        issues.append("Provide three to five relevant, concise Facebook hashtags.")

    source_words = len(re.findall(r"\S+", " ".join((article.article_text or "", article.description or ""))))
    body = BeautifulSoup(content.blogger_html or "", "lxml")
    article_words = len(re.findall(r"\S+", body.get_text(" ", strip=True)))
    if source_words >= 350:
        headings = len(body.find_all(["h2", "h3"]))
        paragraphs = len(body.find_all("p"))
        if article_words < 240:
            issues.append("Develop the article into a complete, useful narrative of at least 240 words; do not add unsupported facts.")
        if headings < 2:
            issues.append("Use at least two meaningful <h2> subheadings to organize the substantial article.")
        if paragraphs < 4:
            issues.append("Break the article into at least four readable paragraphs.")
    return issues


def generate_thumbnail_prompt(
    gem: GeminiClient,
    article: SourceArticle,
    content: GeneratedContent,
) -> str:
    """Gemini writes the precise thumbnail image prompt from the verified source facts. Never raises."""
    text = "\n\n".join(
        part for part in (article.article_text.strip(), article.description.strip()) if part
    )[:6000]
    body = BeautifulSoup(content.blogger_html or "", "lxml").get_text(" ", strip=True)[:2500]

    prompt = (
        f"SOURCE TITLE: {article.original_title}\n"
        f"SOURCE TEXT:\n{text}\n\n"
        f"ARTICLE HEADLINE: {content.blogger_title}\n"
        f"ARTICLE SUMMARY TEXT:\n{body}\n\n"
        f"VISUAL HINT: {content.article_scene_idea}\n\n"
        "Write the single thumbnail image prompt now."
    )

    try:
        res = gem.generate_json(
            prompt,
            ThumbnailBriefSchema,
            system=THUMBNAIL_SYSTEM,
            temperature=0.4,
            tag="METADATA",
        )
        return _strip_urls(res.thumbnail_prompt)[:1800]
    except Exception as exc:
        logger.warn("THUMBNAIL", f"thumbnail prompt unavailable ({type(exc).__name__}); using scene idea")
        return ""


def generate_content(
    gem: GeminiClient,
    article: SourceArticle,
    avoid_titles: list[str],
    feedback: str = "",
    previous: GeneratedContent | None = None,
) -> GeneratedContent:
    source_parts = []
    if article.article_text.strip():
        source_parts.append("ARTICLE BODY:\n" + article.article_text.strip())
    if article.description.strip():
        source_parts.append("SOURCE DESCRIPTION:\n" + article.description.strip())
    text = "\n\n".join(source_parts)
    date = article.publication_date.date() if article.publication_date else "unknown"

    prompt = (
        f"SOURCE: {article.source_name}\n"
        f"SOURCE TITLE: {article.original_title}\n"
        f"SOURCE DATE: {date}\n"
        f"SOURCE TEXT:\n{text[:9000]}\n\n"
        "TITLES ALREADY USED:\n"
        + "\n".join(f"- {t}" for t in avoid_titles[:40] if str(t).strip())
    )

    if feedback:
        prompt += (
            "\n\nCORRECTIONS REQUIRED — HIGHEST PRIORITY:\n"
            "Remove each flagged assertion unless SOURCE TEXT explicitly supports it. Do not merely reword it and do not replace it with a new unsupported claim. Recheck every title, paragraph, SEO field, Facebook line, and scene hint against the source.\n"
            + feedback
        )

        if previous is not None:
            prompt += (
                "\n\nPREVIOUS DRAFT (JSON):\n"
                + previous.model_dump_json()
            )

    body_result = gem.generate_json(
        prompt + "\n\nWrite only the complete blogger_html article body.",
        ArticleBodySchema,
        system=ARTICLE_BODY_SYSTEM,
        temperature=0.3 if previous is not None else 0.5,
        tag="ARTICLE",
    )
    body_text = BeautifulSoup(body_result.blogger_html, "lxml").get_text(" ", strip=True)
    metadata_prompt = (
        prompt
        + "\n\nGEMINI ARTICLE BODY (use as context; do not rewrite):\n"
        + body_text[:12000]
        + "\n\nGenerate only the article title, SEO, labels, Facebook copy, and visual-analysis hints."
    )
    metadata_result = gem.generate_json(
        metadata_prompt,
        EditorialMetadataSchema,
        system=EDITORIAL_METADATA_SYSTEM,
        temperature=0.3 if previous is not None else 0.5,
        tag="METADATA",
    )

    data = metadata_result.model_dump()
    data["blogger_html"] = body_result.blogger_html

    data["blogger_title"] = _strip_urls(data["blogger_title"])
    data["facebook_title"] = _strip_urls(data["facebook_title"])
    data["facebook_post"] = _strip_urls(data["facebook_post"])
    data["facebook_hashtags"] = _normalize_hashtags(data.get("facebook_hashtags"))
    data["first_comment_hook"] = _strip_urls(data["first_comment_hook"])
    data["seo_description"] = str(data["seo_description"] or "").strip()[:155].rstrip()

    comp_type = str(data.get("facebook_composition_type", "INSET_CIRCLE_RIGHT")).strip().upper()
    comp_type = comp_type.replace("-", "_").replace(" ", "_")
    comp_type = {
        "INSET_CIRCLE": "INSET_CIRCLE_RIGHT",
        "INSET_SQUARE": "INSET_SQUARE_RIGHT",
        "SPLIT": "DIPTYCH_SPLIT",
    }.get(comp_type, comp_type)
    if comp_type not in {
        "DIPTYCH_SPLIT", "DIPTYCH_STACK", "INSET_CIRCLE_RIGHT", "INSET_CIRCLE_LEFT",
        "INSET_SQUARE_RIGHT", "INSET_SQUARE_LEFT", "TRIPTYCH", "TRIPTYCH_BOTTOM", "QUAD_GRID", "SINGLE_HERO",
    }:
        comp_type = "INSET_CIRCLE_RIGHT"
    data["facebook_composition_type"] = comp_type

    data["facebook_scene_idea"] = _strip_urls(data.get("facebook_scene_idea", ""))
    data["facebook_detail_scene_idea"] = _strip_urls(data.get("facebook_detail_scene_idea", ""))

    labels: list[str] = []

    for label in data.get("labels") or []:
        label = str(label).strip()
        if label and label not in labels:
            labels.append(label)

    data["labels"] = labels[:5] or ["قصص غريبة"]

    return GeneratedContent(**data)


def fact_check(
    gem: GeminiClient,
    article: SourceArticle,
    content: GeneratedContent,
) -> FactCheckSchema:
    plain = BeautifulSoup(content.blogger_html, "lxml").get_text(" ")
    source_text = "\n\n".join(
        part for part in (article.article_text.strip(), article.description.strip()) if part
    )

    prompt = (
        f"SOURCE TEXT:\n{source_text[:FACT_SOURCE_CHARS]}\n\n"
        f"ARTICLE TITLE:\n{content.blogger_title}\n\n"
        f"SEO DESCRIPTION:\n{content.seo_description}\n\n"
        f"ARTICLE BODY:\n{plain[:FACT_ARTICLE_CHARS]}\n\n"
        f"FACEBOOK TITLE:\n{content.facebook_title}\n\n"
        f"FACEBOOK POST:\n{content.facebook_post}\n\n"
        f"FACEBOOK HASHTAGS:\n{' '.join(content.facebook_hashtags)}\n\n"
        f"FIRST COMMENT:\n{content.first_comment_hook}\n\n"
        f"ARTICLE SCENE IDEA:\n{content.article_scene_idea}\n\n"
        f"FACEBOOK SCENE IDEA:\n{content.facebook_scene_idea}\n\n"
        f"FACEBOOK DETAIL SCENE IDEA:\n{content.facebook_detail_scene_idea}\n\n"
        f"FACEBOOK COMPOSITION TYPE:\n{content.facebook_composition_type}\n\n"
        "Check every factual claim against SOURCE TEXT."
    )

    return gem.generate_json(prompt, FactCheckSchema, system=FACT_SYSTEM, temperature=0.0, tag="FACTCHECK")


def generate_verified(
    gem: GeminiClient,
    article: SourceArticle,
    history_titles: list[str],
    is_title_taken: Callable[[str], bool],
) -> GeneratedContent:
    content = generate_content(gem, article, history_titles)

    for round_no in range(MAX_FIX_ROUNDS + 1):
        check = fact_check(gem, article, content)
        claims = _claims(check)
        if not check.all_claims_supported and not claims:
            claims = ["Fact checker reported unsupported claims but returned no details; recheck the complete draft."]
        structure_issues = _quality_issues(article, content)

        if not claims and not structure_issues:
            break

        if claims:
            logger.warn("FACTCHECK", f"{len(claims)} unsupported claim(s) (check {round_no + 1}/{MAX_FIX_ROUNDS + 1})")

        for claim in claims[:6]:
            logger.warn("FACTCHECK", f"  - {_clip(claim)}")

        if round_no >= MAX_FIX_ROUNDS:
            problems = []
            if claims:
                problems.append(f"{len(claims)} unsupported claim(s)")
            if structure_issues:
                problems.append(f"{len(structure_issues)} article structure issue(s)")
            raise ValueError(
                f"content failed quality checks after {MAX_FIX_ROUNDS} correction rounds: "
                + ", ".join(problems)
            )

        feedback = "\n".join(
            [*(f"- Remove or correct this unsupported claim: {c}" for c in claims),
             *(f"- Structure requirement: {issue}" for issue in structure_issues)]
        )
        content = generate_content(gem, article, history_titles, feedback=feedback, previous=content)

    if is_title_taken(content.blogger_title):
        logger.warn("CONTENT", "title too similar to history; asking for a new title")

        content = generate_content(
            gem,
            article,
            history_titles + [content.blogger_title],
            feedback=(
                "Use a clearly different blogger_title and facebook_title while "
                "preserving exactly the same verified source facts."
            ),
            previous=content,
        )

        final_check = fact_check(gem, article, content)
        final_claims = _claims(final_check)
        if not _fact_check_passes(final_check):
            detail = "; ".join(final_claims[:3]) or "the fact-check result was inconsistent"
            raise ValueError(f"title regeneration failed fact-check: {detail}")

        final_structure_issues = _quality_issues(article, content)
        if final_structure_issues:
            raise ValueError("title regeneration failed structure checks: " + "; ".join(final_structure_issues))

        if is_title_taken(content.blogger_title):
            raise ValueError("could not obtain a unique title")

    # Written once, after the article is verified, so the thumbnail brief matches the final facts.
    brief = generate_thumbnail_prompt(gem, article, content)
    if brief:
        content = content.model_copy(update={"thumbnail_prompt": brief})

    return content
