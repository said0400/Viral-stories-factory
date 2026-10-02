"""Twilio WhatsApp notifications. Free-form inside the 24h window; approved template outside it."""
from __future__ import annotations

import json

from twilio.base.exceptions import TwilioRestException
from twilio.rest import Client

from . import logger
from .config import Settings
from .models import FacebookPackage, WhatsAppResult

OUTSIDE_WINDOW_CODES = {63016, 63018, 63049}   # free-form not allowed / template required
MAX_BODY = 1500


class WhatsAppError(Exception):
    def __init__(self, msg: str, partial_result: WhatsAppResult | None = None) -> None:
        super().__init__(msg)
        self.partial_result = partial_result


def _wa(n: str) -> str:
    return n if n.startswith("whatsapp:") else f"whatsapp:{n}"


def _chunks(text: str, limit: int = MAX_BODY) -> list[str]:
    """Split text into chunks below the Twilio limit, preserving paragraph breaks.
    If a single paragraph exceeds the limit, it is hard-split."""
    out, cur = [], ""
    for para in text.split("\n"):
        # Hard split super long paragraphs to ensure no chunk ever exceeds MAX_BODY
        while len(para) > limit:
            if cur:
                out.append(cur.strip())
                cur = ""
            out.append(para[:limit].strip())
            para = para[limit:]

        if len(cur) + len(para) + 1 > limit and cur:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n"
        
    if cur.strip():
        out.append(cur.strip())
    return out


class WhatsAppClient:
    def __init__(self, cfg: Settings) -> None:
        if not cfg.twilio_configured():
            raise WhatsAppError("Twilio WhatsApp credentials are incomplete")
        self.cfg = cfg
        self.client = Client(cfg.twilio_account_sid, cfg.twilio_auth_token)
        self.sender, self.to = _wa(cfg.twilio_whatsapp_number), _wa(cfg.your_personal_number)

    def _send(self, body: str, media_url: str | None = None) -> str:
        kw = dict(from_=self.sender, to=self.to, body=body)
        if media_url:
            kw["media_url"] = [media_url]
        return self.client.messages.create(**kw).sid

    def send_package(
        self,
        *,
        title: str,
        blogger_url: str,
        source_name: str,
        source_url: str,
        fb: FacebookPackage,
        image_public_url: str,
        status: str,
        already_sent_parts: list[str] | None = None,
    ) -> WhatsAppResult:
        """
        Sends the package in parts (header, post_chunks, comment_chunks).
        Skips any part explicitly listed in already_sent_parts (from a previous partial failure).
        """
        sent = set(already_sent_parts or [])
        res = WhatsAppResult()

        header = (f"📰 Story\n\nTitle:\n{title}\n\nSource:\n{source_name}\n{source_url}\n\n"
                  f"Blogger:\n{blogger_url}\n\nStatus: {status}\nImage: "
                  f"{'attached (AI generated)' if image_public_url else 'not attached (no public URL)'}")
        
        fb_msg = f"Facebook Title:\n{fb.title}\n\nFacebook Post:\n{fb.post}"
        comment = f"First Comment:\n{fb.first_comment}"
        
        post_parts = _chunks(fb_msg)
        comment_parts = _chunks(comment)

        try:
            # 1. Header
            if "header" not in sent:
                res.message_ids.append(self._send(header, image_public_url or None))
                res.sent_parts.append("header")
                res.media_sent = bool(image_public_url)

            # 2. Post Chunks
            for i, part in enumerate(post_parts):
                tag = f"post_{i}"
                if tag not in sent:
                    res.message_ids.append(self._send(part))
                    res.sent_parts.append(tag)

            # 3. Comment Chunks
            for i, part in enumerate(comment_parts):
                tag = f"comment_{i}"
                if tag not in sent:
                    res.message_ids.append(self._send(part))
                    res.sent_parts.append(tag)

            return res

        except TwilioRestException as exc:
            # Check if this was a 24-hour window restriction on the FIRST message (header)
            if exc.code in OUTSIDE_WINDOW_CODES and not res.sent_parts:
                if self.cfg.whatsapp_content_sid:
                    logger.log("WHATSAPP", "outside 24h window -> using approved template")
                    return self._send_template(title, blogger_url, res)
                raise WhatsAppError(
                    "outside the 24h window and WHATSAPP_CONTENT_SID is not set", 
                    partial_result=res
                ) from exc
            
            # Any other failure mid-flight or general Twilio error
            raise WhatsAppError(f"Twilio error {exc.code}", partial_result=res) from exc
        
        except Exception as exc:
            raise WhatsAppError(f"Unexpected WhatsApp error: {type(exc).__name__}", partial_result=res) from exc

    def _send_template(self, title: str, blogger_url: str, res: WhatsAppResult) -> WhatsAppResult:
        try:
            m = self.client.messages.create(
                from_=self.sender,
                to=self.to,
                content_sid=self.cfg.whatsapp_content_sid,
                content_variables=json.dumps({"1": title[:200], "2": blogger_url})
            )
            res.message_ids.append(m.sid)
            res.sent_parts.append("template")
            res.used_template = True
            return res
        except TwilioRestException as exc:
            raise WhatsAppError(f"template send failed: {exc.code}", partial_result=res) from exc
