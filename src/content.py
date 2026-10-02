"""Gemini-driven editorial steps: triage/viral selection, Arabic content, fact check."""
from __future__ import annotations

import html as html_lib
import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup, Comment

from . import logger
from .gemini_client import GeminiClient
from .models import (
    ContentSchema,
    FactCheckSchema,
    GeneratedContent,
    SourceArticle,
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
- strong hook introduction
- logical story progression
- necessary context
- important verified details
- clear ending based only on available information

Target 450-900 Arabic words when the source contains enough facts.
If the source is thin, write less.
NEVER pad the article to reach a word count.

6. blogger_title:
- honest
- intriguing
- curiosity-driven
- not misleading
- no fabricated information
- <= 90 characters
- do not use sensational claims that the source does not support

7. facebook_title:
- short
- strong
- mobile friendly
- honest curiosity gap
- does not reveal everything
- does not invent facts
- does not use false clickbait

8. facebook_post:
Structure:
HOOK -> INTEREST -> PARTIAL CONTEXT -> CURIOSITY.

Do NOT include any URL or website placeholder.
Do NOT reveal the entire story or main twist.
Do NOT fabricate suspense.
Do NOT claim that something happened unless the source confirms it.
Vary openings and avoid stock phrases.

9. first_comment_hook:
One short, natural sentence inviting readers to read the full details.
No URL.
No fabricated claim.

10. labels:
Return 2-5 Arabic labels genuinely related to the story.
Examples:
قصص غريبة
قصص حقيقية
غرائب
أخبار طريفة
اكتشافات

Do not add unrelated SEO labels.

11. article_scene_idea / facebook_scene_idea:
Write in English.
Use 1-2 concise sentences.
Describe a NEW visual scene for an illustration that reflects the story.

The visual scene MUST:
- use only facts supported by the source
- avoid invented locations
- avoid invented objects
- avoid invented people
- avoid invented clothing
- avoid invented weather
- avoid invented architecture
- avoid invented actions
- avoid invented emotions
- avoid invented injuries
- avoid invented relationships
- avoid invented background details

Do not describe the face of any real person.
Do not request facial recognition.
Do not request facial reconstruction.
Do not guess a person's identity from appearance.

12. REAL PEOPLE:
When real people are part of the story, describe only factual/contextual information supported by the source.
Never invent facial features, expressions, age appearance, ethnicity, identity or other sensitive characteristics.
The image generation system may use a supplied reference image where permitted, but the editorial text itself must never claim that an exact identity or likeness has been established.

13. MINORS:
If minors are part of the story, do not request identifiable facial depictions.
Prefer a non-identifying, indirect or contextual visual treatment.

14. SOURCE IMAGE:
A source image is a reference for the story and subject only.
It must NOT be treated as a request to reproduce the original image exactly.

When suggesting a new visual:
- preserve supported identity-critical subject characteristics when appropriate
- change composition, framing, camera angle, lighting or moment
- do not invent new factual events
- do not turn an illustrative scene into a claim that the event happened exactly that way

15. NON-HUMAN SUBJECTS:
For animals, vehicles, buildings, places, objects and other specific non-human subjects, preserve important identifying characteristics supported by the source or reference image.
Do not invent markings, colors, damage, architecture, model numbers or other identity-critical details.

16. UNCERTAINTY:
If the source does not establish a detail, omit it or clearly mark it as uncertain.
Never use a plausible assumption as a factual detail.

17. FACTUAL PRIORITY:
Accuracy is more important than drama.
If a dramatic visual or sentence would require inventing a fact, do not use it.

18. SEO:
seo_description must be <= 155 characters in Arabic.
Keep it natural.
No keyword stuffing.
Do not add facts that are not supported by the source.

19. FINAL CONSISTENCY:
blogger_title, facebook_title, facebook_post, first_comment_hook, article_scene_idea and facebook_scene_idea must all remain consistent with the same source facts.
Do not introduce a new factual claim in one field that is absent from the article/source."""

FACT_SYSTEM = """You are a strict fact checker.

Compare the ARTICLE, SEO DESCRIPTION, and FACEBOOK POST against the SOURCE TEXT.

List every concrete claim that is NOT supported by the source.

Concrete claims include:
- names
- ages
- dates
- locations
- numbers
- quotes
- causes
- motives
- outcomes
- relationships
- medical claims
- statements about what a real person thought, felt, intended or knew
- claims about what happened
- claims about specific objects, animals, vehicles, buildings or places

Paraphrase and translation are fine.
Reasonable wording differences are NOT factual errors.

Do NOT flag:
- harmless stylistic wording
- natural Arabic transitions
- clearly marked uncertainty
- statements that accurately summarize the source
- obvious grammatical reformulations

A claim is unsupported when the SOURCE TEXT does not establish it.

Pay special attention to claims that:
- add a new fact
- change a number
- change a date
- change a location
- change who did something
- change the cause of an event
- change the outcome
- turn uncertainty into certainty
- attribute a thought, emotion, motive or intention to a real person

Also check the visual scene ideas for factual additions.
If an image idea describes a specific factual element not supported by the source, include it as an unsupported claim.

Return only claims that genuinely require correction.
If everything is supported, return an empty unsupported_claims list and all_claims_supported=true."""


# ---------------------------------------------------------------- sanitising

def sanitize_html(raw: str) -> str:
    """
    Allow-list HTML sanitiser.

    Removes executable/dangerous elements, comments, attributes and unknown
    tags while preserving the text/content of harmless unknown wrappers.
    """
    if not raw:
        return ""

    soup = BeautifulSoup(str(raw), "lxml")

    # Remove comments explicitly. This prevents model-generated HTML comments
    # from surviving into the article body.
    for comment in soup.find_all(
        string=lambda value: isinstance(value, Comment)
    ):
        comment.extract()

    # Remove dangerous or explicitly forbidden elements.
    for tag in soup.find_all(
        [
            "script",
            "style",
            "iframe",
            "object",
            "embed",
            "form",
            "img",
            "a",
            "meta",
            "link",
            "base",
            "svg",
            "math",
            "video",
            "audio",
            "source",
            "picture",
        ]
    ):
        if tag.name == "a":
            tag.unwrap()
        else:
            tag.decompose()

    # Remove document-level wrappers and enforce the allow-list.
    for tag in list(soup.find_all(True)):
        name = str(tag.name or "").lower()

        if name in ("html", "head", "body"):
            tag.unwrap()
            continue

        if name not in ALLOWED_TAGS:
            tag.unwrap()
            continue

        # No attributes are allowed on article-generated tags.
        tag.attrs = {}

    # BeautifulSoup can leave doctype/document artefacts depending on input.
    # Work only with the resulting body/content nodes.
    body = soup.body

    if body is not None:
        return "".join(str(child) for child in body.contents).strip()

    return "".join(str(child) for child in soup.contents).strip()


def _safe_story_id(story_id: str) -> str:
    """
    Validate and normalise the story ID used in the Blogger idempotency marker.

    The marker is intentionally simple because history.py and blogger.py use
    the exact same `story_id:<id>` convention.
    """
    value = str(story_id or "").strip()

    if not value:
        raise ValueError("story_id is required")

    if len(value) > 128:
        raise ValueError("story_id is too long")

    if not re.fullmatch(r"[A-Za-z0-9._:-]+", value):
        raise ValueError("story_id contains unsupported characters")

    return value


def _safe_source_host(source_url: str) -> str:
    """Return a safe display hostname for the attribution line."""
    parsed_url = urlparse(source_url or "")

    host = (parsed_url.hostname or "").strip().lower()

    if not host:
        return "source"

    if host.startswith("www."):
        host = host[4:]

    return html_lib.escape(host, quote=True)


def _safe_source_url(source_url: str) -> str:
    """
    Return a source URL only when it is a normal HTTP(S) URL.

    The article source URL originates from discovery, but this validation
    prevents malformed schemes from being emitted into the final Blogger HTML.
    """
    value = str(source_url or "").strip()

    if not value:
        return ""

    try:
        parsed = urlparse(value)
    except ValueError:
        return ""

    if parsed.scheme not in ("http", "https"):
        return ""

    if not parsed.netloc:
        return ""

    if parsed.username or parsed.password:
        return ""

    return value


def render_blogger_html(
    content: GeneratedContent,
    article: SourceArticle,
    image_url: str,
    story_id: str,
) -> str:
    """
    Final Blogger HTML:
    image + sanitised body + attribution + hidden idempotency marker.
    """
    safe_story_id = _safe_story_id(story_id)

    body = sanitize_html(content.blogger_html)

    source_url = _safe_source_url(article.original_url)

    if source_url:
        host = _safe_source_host(source_url)

        src = (
            f'<p><strong>المصدر:</strong> '
            f'<a href="{html_lib.escape(source_url, quote=True)}" '
            f'rel="nofollow noopener" target="_blank">'
            f'{html_lib.escape(article.source_name)}</a> '
            f'({host})</p>'
            f'<p><em>'
            f'الصورة المرفقة مُولَّدة بالذكاء الاصطناعي لأغراض توضيحية.'
            f'</em></p>'
        )
    else:
        src = (
            f'<p><strong>المصدر:</strong> '
            f'{html_lib.escape(article.source_name or "المصدر")}</p>'
            f'<p><em>'
            f'الصورة المرفقة مُولَّدة بالذكاء الاصطناعي لأغراض توضيحية.'
            f'</em></p>'
        )

    safe_image_url = str(image_url or "").strip()

    img = ""

    if safe_image_url:
        try:
            parsed_image = urlparse(safe_image_url)

            if parsed_image.scheme in ("http", "https") and parsed_image.netloc:
                img = (
                    f'<figure style="margin:0 0 1.2em;text-align:center">'
                    f'<img src="'
                    f'{html_lib.escape(safe_image_url, quote=True)}" '
                    f'alt="'
                    f'{html_lib.escape(content.blogger_title, quote=True)}" '
                    f'style="max-width:100%;height:auto" '
                    f'loading="lazy"/></figure>'
                )
        except ValueError:
            img = ""

    return (
        f'<div dir="rtl" style="text-align:right">'
        f'{img}'
        f'{body}'
        f'{src}'
        f'</div>'
        f'<!-- story_id:{html_lib.escape(safe_story_id, quote=False)} -->'
    )


# ---------------------------------------------------------------- triage

def triage(
    gem: GeminiClient,
    candidates: list[SourceArticle],
    published_titles: list[str],
) -> list[TriageItem]:
    if not candidates:
        return []

    lines = []

    for i, c in enumerate(candidates):
        age = (
            c.publication_date.date().isoformat()
            if c.publication_date
            else "unknown"
        )

        lines.append(
            f"[{i}] ({c.source_name}, {age}) "
            f"{c.original_title} — {c.description[:220]}"
        )

    pub = (
        "\n".join(
            f"- {t}"
            for t in published_titles[:60]
            if str(t).strip()
        )
        or "(none)"
    )

    prompt = (
        "CANDIDATE STORIES:\n"
        + "\n".join(lines)
        + "\n\nALREADY PUBLISHED "
        "(flag already_published=true if a candidate is the same EVENT):\n"
        + pub
        + "\n\n"
        "For each candidate return an item.\n"
        "Set duplicate_of to the index of an EARLIER candidate "
        "describing the same event/people/place, else -1.\n"
        "Never point duplicate_of to itself or to a later candidate.\n"
        "event_key = people+place+event in a few concise English words.\n"
        "suitable=false for politics, tragedy, graphic, unverifiable "
        "or boring items.\n"
        "is_evergreen=true if the story remains interesting regardless of date."
    )

    res = gem.generate_json(
        prompt,
        TriageResult,
        system=TRIAGE_SYSTEM,
        temperature=0.2,
        tag="TRIAGE",
    )

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
            value = getattr(it, field, 0)

            try:
                value = int(value)
            except (TypeError, ValueError):
                value = 0

            setattr(
                it,
                field,
                max(0, min(100, value)),
            )

        # duplicate_of must be an earlier candidate and cannot refer to
        # itself. Invalid references are converted to "no duplicate".
        duplicate_of = getattr(it, "duplicate_of", -1)

        try:
            duplicate_of = int(duplicate_of)
        except (TypeError, ValueError):
            duplicate_of = -1

        if (
            duplicate_of < 0
            or duplicate_of >= len(candidates)
            or duplicate_of >= it.index
        ):
            duplicate_of = -1

        it.duplicate_of = duplicate_of

        # Keep event_key compact and safe for downstream logs/selection.
        event_key = str(getattr(it, "event_key", "") or "").strip()

        if len(event_key) > 160:
            event_key = event_key[:160].rstrip()

        it.event_key = event_key

        out[it.index] = it

    return list(out.values())


def rank_score(item: TriageItem) -> float:
    """Internal ordering only; never shown to readers."""
    return (
        0.35 * item.viral_score
        + 0.25 * item.curiosity_score
        + 0.15 * item.share_score
        + 0.10 * item.comment_score
        + 0.10 * item.originality_score
        + 0.05 * item.emotional_score
    )


# ---------------------------------------------------------------- generation

def generate_content(
    gem: GeminiClient,
    article: SourceArticle,
    avoid_titles: list[str],
    feedback: str = "",
) -> GeneratedContent:
    text = article.article_text or article.description

    prompt = (
        f"SOURCE: {article.source_name}\n"
        f"SOURCE TITLE: {article.original_title}\n"
        f"SOURCE DATE: "
        f"{article.publication_date.date() if article.publication_date else 'unknown'}\n"
        f"SOURCE TEXT:\n{text[:9000]}\n\n"
        "TITLES ALREADY USED "
        "(do not repeat or closely resemble):\n"
        + "\n".join(
            f"- {t}"
            for t in avoid_titles[:40]
            if str(t).strip()
        )
        + (
            "\n\nCORRECTIONS REQUIRED FROM THE FACT CHECKER "
            "(remove or fix these claims):\n"
            + feedback
            if feedback
            else ""
        )
    )

    # Temperature set to 0.5 for better factual accuracy.
    res = gem.generate_json(
        prompt,
        ContentSchema,
        system=CONTENT_SYSTEM,
        temperature=0.5,
        tag="CONTENT",
    )

    return GeneratedContent(
        **res.model_dump()
    )


def fact_check(
    gem: GeminiClient,
    article: SourceArticle,
    content: GeneratedContent,
) -> FactCheckSchema:
    plain = BeautifulSoup(
        content.blogger_html,
        "lxml",
    ).get_text(" ")

    prompt = (
        f"SOURCE TEXT:\n"
        f"{(article.article_text or article.description)[:9000]}\n\n"
        f"ARTICLE TITLE:\n"
        f"{content.blogger_title}\n\n"
        f"SEO DESCRIPTION:\n"
        f"{content.seo_description}\n\n"
        f"ARTICLE BODY:\n"
        f"{plain[:12000]}\n\n"
        f"FACEBOOK TITLE:\n"
        f"{content.facebook_title}\n\n"
        f"FACEBOOK POST:\n"
        f"{content.facebook_post}\n\n"
        f"FIRST COMMENT:\n"
        f"{content.first_comment_hook}\n\n"
        f"ARTICLE SCENE IDEA:\n"
        f"{content.article_scene_idea}\n\n"
        f"FACEBOOK SCENE IDEA:\n"
        f"{content.facebook_scene_idea}\n\n"
        "Check every factual claim against SOURCE TEXT."
    )

    return gem.generate_json(
        prompt,
        FactCheckSchema,
        system=FACT_SYSTEM,
        temperature=0.0,
        tag="FACTCHECK",
    )


def generate_verified(
    gem: GeminiClient,
    article: SourceArticle,
    history_titles: list[str],
    is_title_taken,
) -> GeneratedContent:
    """Generate -> fact-check -> corrective regeneration -> title-dedup check."""

    content = generate_content(
        gem,
        article,
        history_titles,
    )

    for attempt in range(2):
        check = fact_check(
            gem,
            article,
            content,
        )

        if (
            check.all_claims_supported
            and not check.unsupported_claims
        ):
            break

        logger.warn(
            "FACTCHECK",
            f"{len(check.unsupported_claims)} unsupported claim(s); regenerating",
        )

        if attempt == 1:
            raise ValueError(
                "content still contains unsupported claims"
            )

        feedback = "\n".join(
            f"- {c}"
            for c in check.unsupported_claims
        )

        content = generate_content(
            gem,
            article,
            history_titles,
            feedback=feedback,
        )

    if is_title_taken(content.blogger_title):
        logger.warn(
            "CONTENT",
            "title too similar to history; asking for a new title",
        )

        content = generate_content(
            gem,
            article,
            history_titles + [content.blogger_title],
            feedback=(
                "Use a clearly different blogger_title "
                "and facebook_title while preserving exactly "
                "the same verified source facts. "
                "Do not introduce any new factual claim."
            ),
        )

        final_check = fact_check(
            gem,
            article,
            content,
        )

        if (
            not final_check.all_claims_supported
            or final_check.unsupported_claims
        ):
            logger.warn(
                "FACTCHECK",
                "new title/content regeneration introduced "
                f"{len(final_check.unsupported_claims)} unsupported claim(s)",
            )
            raise ValueError(
                "title regeneration produced unsupported claims"
            )

        if is_title_taken(content.blogger_title):
            raise ValueError(
                "could not obtain a unique title"
            )

    return content
