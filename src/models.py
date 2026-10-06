"""Pydantic models: pipeline data + Gemini response schemas."""
from __future__ import annotations

from datetime import datetime
from typing import Optional

from pydantic import BaseModel, Field


# =======================================================================
# pipeline
class SourceArticle(BaseModel):
    source_name: str
    original_title: str
    original_url: str
    normalized_url: str
    canonical_url: str = ""
    publication_date: Optional[datetime] = None
    author: str = ""
    description: str = ""
    article_text: str = ""
    main_image_url: str = ""
    additional_image_urls: list[str] = Field(default_factory=list)
    source_section: str = ""
    discovered_at: datetime
    age_exception: str = ""  # why an old story was accepted (logged)


class VisualAnalysis(BaseModel):
    subject_type: str = "other"
    identity_critical: bool = False
    contains_real_people: bool = False
    involves_minors: bool = False
    identity_features: list[str] = Field(default_factory=list)
    scene_features: list[str] = Field(default_factory=list)
    new_scene_direction: str = ""
    reference_required: bool = False
    identity_confidence: str = "low"  # low | medium | high
    summary: str = ""


class GeneratedContent(BaseModel):
    blogger_title: str
    blogger_html: str
    seo_description: str
    labels: list[str]
    facebook_title: str
    facebook_post: str          # no URL inside; URL is added after Blogger succeeds
    first_comment_hook: str
    article_scene_idea: str
    facebook_scene_idea: str          # Main Photo (Background)
    facebook_detail_scene_idea: str = "" # Secondary Inset Photo (Circle/Square)
    facebook_composition_type: str = "INSET_CIRCLE_RIGHT"


class ImageResult(BaseModel):
    path: str = ""
    facebook_path: str = ""
    mime_type: str = "image/jpeg"
    strategy: str = ""
    style: str = ""
    identity_confidence: str = "low"
    
    generated_hash: str = ""
    generated_ahash: str = ""
    
    # Traceability for the separate Facebook image
    facebook_image_hash: str = ""
    facebook_image_ahash: str = ""
    
    source_image_url: str = ""
    source_image_hash: str = ""
    source_image_ahash: str = ""
    
    public_url: str = ""
    facebook_public_url: str = ""
    notes: str = ""


class BloggerResult(BaseModel):
    post_id: str
    url: str
    published_at: str


class FacebookPackage(BaseModel):
    title: str
    post: str
    first_comment: str
    image_path: str = ""


class WhatsAppResult(BaseModel):
    message_ids: list[str] = Field(default_factory=list)
    used_template: bool = False
    media_sent: bool = False
    # Will hold tags like "header", "post_1", "post_2", "comment" for partial tracking
    sent_parts: list[str] = Field(default_factory=list)


class StoryState(BaseModel):
    """One row of data/history.json (superset of the specification)."""
    story_id: str
    source: str
    original_url: str
    normalized_url: str
    original_title: str
    discovered_at: str = ""
    selected_at: str = ""
    generated_at: str = ""
    image_generated_at: str = ""
    blogger_post_id: str = ""
    blogger_url: str = ""
    published_at: str = ""
    
    # State tracking
    status: str = "discovered"
    image_status: str = ""
    facebook_status: str = ""
    
    # WhatsApp tracking
    whatsapp_status: str = ""
    whatsapp_message_id: str = ""  # Kept for backward compatibility
    whatsapp_message_ids: list[str] = Field(default_factory=list)
    whatsapp_sent_parts: list[str] = Field(default_factory=list)
    
    # Export / Offline bundle tracking
    export_status: str = "pending"  # pending | ready | failed
    bundle_path: str = ""
    export_created_at: str = ""
    export_error: str = ""
    
    error: str = ""

    # extras: traceability / recovery
    blogger_title: str = ""
    subject_type: str = ""
    identity_confidence: str = ""
    source_image_url: str = ""
    source_image_hash: str = ""
    source_image_ahash: str = ""
    generated_image_path: str = ""
    generated_image_hash: str = ""
    generated_image_ahash: str = ""
    
    attempts: int = 0
    failed_stage: str = ""
    updated_at: str = ""


class StoryCache(BaseModel):
    """Per-story artefacts so recovery never re-pays for Gemini calls."""
    article: Optional[SourceArticle] = None
    visual: Optional[VisualAnalysis] = None
    content: Optional[GeneratedContent] = None
    image: Optional[ImageResult] = None
    facebook: Optional[FacebookPackage] = None
    analysis_reason: str = ""


# =======================================================================
# Gemini schemas
# (kept free of defaults/validators for maximum JSON-Schema compatibility)
class TriageItem(BaseModel):
    index: int
    suitable: bool
    viral_score: int
    curiosity_score: int
    share_score: int
    comment_score: int
    originality_score: int
    emotional_score: int
    is_evergreen: bool

    # index of the same event in this batch, or -1
    duplicate_of: int

    # same event as one in the "already published" list
    already_published: bool

    # people / place / event in a few words (English)
    event_key: str
    reason: str


class TriageResult(BaseModel):
    items: list[TriageItem]


class VisualSchema(BaseModel):
    subject_type: str
    # person|place|animal|vehicle|object|building|
    # event|scene|multiple_subjects|other
    identity_critical: bool
    contains_real_people: bool
    involves_minors: bool
    identity_features: list[str]
    scene_features: list[str]
    new_scene_direction: str
    identity_confidence: str  # low|medium|high
    summary: str


class ContentSchema(BaseModel):
    blogger_title: str
    blogger_html: str
    seo_description: str
    labels: list[str]
    facebook_title: str
    facebook_post: str
    first_comment_hook: str
    article_scene_idea: str
    facebook_scene_idea: str
    facebook_detail_scene_idea: str
    facebook_composition_type: str


class FactCheckSchema(BaseModel):
    all_claims_supported: bool
    unsupported_claims: list[str]


class ImageCheckSchema(BaseModel):
    relevant_to_story: bool
    contains_text_or_watermark: bool
    obvious_defects: bool
    reason: str
