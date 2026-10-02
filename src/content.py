"""Gemini-driven editorial steps: triage/viral selection, Arabic content, fact check."""
from __future__ import annotations

import html as html_lib
from urllib.parse import urlparse

from bs4 import BeautifulSoup

from . import logger
from .gemini_client import GeminiClient
from .models import (ContentSchema, FactCheckSchema, GeneratedContent, SourceArticle,
                     TriageItem, TriageResult)

ALLOWED_TAGS = {"h2", "h3", "p", "ul", "ol", "li", "strong", "em", "blockquote", "br"}

TRIAGE_SYSTEM = """You are an editor selecting REAL, verifiable, curiosity-driven stories for a general Arabic audience.
Prefer: strange, surprising, human, animal, mysterious, funny, unusual, rare discoveries, bizarre incidents, odd places, intriguing history, stories with a surprising ending.
De-prioritise: politics, partisan news, war, crime with graphic detail, celebrity gossip, tragedies exploiting victims, anything sexual, medical advice, stories of minors in distress.
Judge ONLY from the title/description provided. Never invent facts. Scores are 0-100 and internal only."""

CONTENT_SYSTEM = """You are a professional Arabic editor-writer. You write ORIGINAL articles from verified source facts.
ABSOLUTE RULES
1. Use ONLY facts found in the SOURCE TEXT. Never invent names, ages, dates, numbers, places, quotes, statements, outcomes, motives, feelings of real people, medical info or unproven causes.
2. If something is unclear write 'بحسب التقرير', 'وفقًا للمصدر', 'لم يتضح' or 'لم يذكر المصدر'.
3. Language: simple modern Arabic understood by all Arabs. Short clear sentences. No literal translation, no stiff journalese, no filler or repetition.
4. Do NOT copy or line-by-line paraphrase the source. Change structure, opening and narrative order while keeping the facts.
5. blogger_html: valid HTML using only <h2>,<h3>,<p>,<ul>,<li>,<strong>,<em>,<blockquote>. No <h1>, no images, no links, no source line (the system adds it). Sections: strong hook intro, story told in logical order, necessary context, important details, clear ending based only on available info. 450-900 Arabic words when the source allows; shorter if facts are thin (never pad).
6. blogger_title: honest, intriguing, not misleading, <= 90 chars.
7. facebook_title: short, strong, mobile friendly, honest curiosity gap, does not reveal everything. facebook_post: HOOK -> INTEREST -> PARTIAL CONTEXT -> CURIOSITY -> a CTA sentence pointing to the full article. Do NOT include any URL. Do not give away the main twist. No false clickbait. Vary openings; never reuse stock phrases.
8. first_comment_hook: one short natural sentence inviting to read the full details (no URL).
9. labels: 2-5 Arabic labels truly related to the story (e.g. قصص غريبة، قصص حقيقية، غرائب).
10. article_scene_idea / facebook_scene_idea: English, 1-2 sentences, a NEW visual scene for an illustration that reflects the story WITHOUT inventing facts and WITHOUT describing any real person's face.
11. seo_description: <= 155 chars Arabic, natural, no keyword stuffing."""

FACT_SYSTEM = """You are a strict fact checker. Compare the ARTICLE (Arabic) against the SOURCE TEXT.
List every concrete claim in the article (names, numbers, dates, places, quotes, causes, outcomes) that is NOT supported by the source. Paraphrase and translation are fine; invented or altered facts are not."""


# ---------------------------------------------------------------- sanitising
def sanitize_html(raw: str) -> str:
    """Allow-list sanitiser: strips scripts, attributes, unknown tags (keeps their text)."""
    soup = BeautifulSoup(raw or "", "lxml")
    for t in soup(["script", "style", "iframe", "object", "embed", "form", "img", "a"]):
        if t.name == "a":
            t.unwrap()
        else:
            t.decompose()
    for t in soup.find_all(True):
        if t.name in ("html", "body"):
            t.unwrap()
        elif t.name not in ALLOWED_TAGS:
            t.unwrap()
        else:
            t.attrs = {}
    return "".join(str(c) for c in (soup.body or soup).contents).strip()


def render_blogger_html(content: GeneratedContent, article: SourceArticle,
                        image_url: str, story_id: str) -> str:
    """Final post HTML: image + sanitised body + attribution + hidden idempotency marker."""
    body = sanitize_html(content.blogger_html)
    host = urlparse(article.original_url).netloc.removeprefix("www.")
    img = (f'<figure style="margin:0 0 1.2em;text-align:center">'
           f'<img src="{html_lib.escape(image_url, quote=True)}" '
           f'alt="{html_lib.escape(content.blogger_title, quote=True)}" '
           f'style="max-width:100%;height:auto" loading="lazy"/></figure>') if image_url else ""
    src = (f'<p><strong>المصدر:</strong> <a href="{html_lib.escape(article.original_url, quote=True)}" '
           f'rel="nofollow noopener" target="_blank">{html_lib.escape(article.source_name)}</a> '
           f'({html_lib.escape(host)})</p>'
           f'<p><em>الصورة المرفقة مُولَّدة بالذكاء الاصطناعي لأغراض توضيحية.</em></p>')
    return (f'<div dir="rtl" style="text-align:right">{img}{body}{src}</div>'
            f'<!-- story_id:{story_id} -->')


# ---------------------------------------------------------------- triage
def triage(gem: GeminiClient, candidates: list[SourceArticle], published_titles: list[str]) -> list[TriageItem]:
    lines = []
    for i, c in enumerate(candidates):
        age = c.publication_date.date().isoformat() if c.publication_date else "unknown"
        lines.append(f"[{i}] ({c.source_name}, {age}) {c.original_title} — {c.description[:220]}")
    pub = "\n".join(f"- {t}" for t in published_titles[:60]) or "(none)"
    prompt = (f"CANDIDATE STORIES:\n" + "\n".join(lines) +
              f"\n\nALREADY PUBLISHED (flag already_published=true if a candidate is the same EVENT):\n{pub}\n\n"
              "For each candidate return an item. Set duplicate_of to the index of an EARLIER candidate "
              "describing the same event/people/place (else -1). event_key = people+place+event in a few English words. "
              "suitable=false for politics, tragedy, graphic, unverifiable or boring items. is_evergreen=true if still "
              "interesting regardless of date.")
    res = gem.generate_json(prompt, TriageResult, system=TRIAGE_SYSTEM, temperature=0.2, tag="TRIAGE")
    out: dict[int, TriageItem] = {}
    for it in res.items:
        if 0 <= it.index < len(candidates):
            for f in ("viral_score", "curiosity_score", "share_score", "comment_score",
                      "originality_score", "emotional_score"):
                setattr(it, f, max(0, min(100, getattr(it, f))))
            out[it.index] = it
    return list(out.values())


def rank_score(item: TriageItem) -> float:   # internal ordering only; never shown to readers
    return (0.35 * item.viral_score + 0.25 * item.curiosity_score + 0.15 * item.share_score
            + 0.10 * item.comment_score + 0.10 * item.originality_score + 0.05 * item.emotional_score)


# ---------------------------------------------------------------- generation
def generate_content(gem: GeminiClient, article: SourceArticle, avoid_titles: list[str],
                     feedback: str = "") -> GeneratedContent:
    text = article.article_text or article.description
    prompt = (f"SOURCE: {article.source_name}\nSOURCE TITLE: {article.original_title}\n"
              f"SOURCE DATE: {article.publication_date.date() if article.publication_date else 'unknown'}\n"
              f"SOURCE TEXT:\n{text[:9000]}\n\n"
              f"TITLES ALREADY USED (do not repeat or closely resemble):\n"
              + "\n".join(f"- {t}" for t in avoid_titles[:40]) +
              (f"\n\nCORRECTIONS REQUIRED FROM THE FACT CHECKER (remove or fix these claims):\n{feedback}"
               if feedback else ""))
    res = gem.generate_json(prompt, ContentSchema, system=CONTENT_SYSTEM, temperature=0.8, tag="CONTENT")
    return GeneratedContent(**res.model_dump())


def fact_check(gem: GeminiClient, article: SourceArticle, content: GeneratedContent) -> FactCheckSchema:
    plain = BeautifulSoup(content.blogger_html, "lxml").get_text(" ")
    prompt = (f"SOURCE TEXT:\n{(article.article_text or article.description)[:9000]}\n\n"
              f"ARTICLE:\n{content.blogger_title}\n{plain[:7000]}\n\nFACEBOOK POST:\n{content.facebook_post}")
    return gem.generate_json(prompt, FactCheckSchema, system=FACT_SYSTEM, temperature=0.0, tag="FACTCHECK")


def generate_verified(gem: GeminiClient, article: SourceArticle, history_titles: list[str],
                      is_title_taken) -> GeneratedContent:
    """Generate -> fact-check -> (one corrective regeneration) -> title-dedup check."""
    content = generate_content(gem, article, history_titles)
    for attempt in range(2):
        check = fact_check(gem, article, content)
        if check.all_claims_supported and not check.unsupported_claims:
            break
        logger.warn("FACTCHECK", f"{len(check.unsupported_claims)} unsupported claim(s); regenerating")
        if attempt == 1:
            raise ValueError("content still contains unsupported claims")
        content = generate_content(gem, article, history_titles,
                                   feedback="\n".join(f"- {c}" for c in check.unsupported_claims))
    if is_title_taken(content.blogger_title):
        logger.warn("CONTENT", "title too similar to history; asking for a new title")
        content = generate_content(gem, article, history_titles + [content.blogger_title],
                                   feedback="Use a clearly different blogger_title and facebook_title.")
        if is_title_taken(content.blogger_title):
            raise ValueError("could not obtain a unique title")
    return content
