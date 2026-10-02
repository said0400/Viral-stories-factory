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
    pass


def _wa(n: str) -> str:
    return n if n.startswith("whatsapp:") else f"whatsapp:{n}"


def _chunks(text: str, limit: int = MAX_BODY) -> list[str]:
    out, cur = [], ""
    for para in text.split("\n"):
        if len(cur) + len(para) + 1 > limit and cur:
            out.append(cur.strip())
            cur = ""
        cur += para + "\n"
    if cur.strip():
        out.append(cur.strip())
    return out


class WhatsAppClient:
    def __init__(self, cfg: Settings) -> None:
        if not (cfg.twilio_account_sid and cfg.twilio_auth_token and cfg.twilio_whatsapp_number
                and cfg.your_personal_number):
            raise WhatsAppError("Twilio WhatsApp credentials are incomplete")
        self.cfg = cfg
        self.client = Client(cfg.twilio_account_sid, cfg.twilio_auth_token)
        self.sender, self.to = _wa(cfg.twilio_whatsapp_number), _wa(cfg.your_personal_number)

    def _send(self, body: str, media_url: str | None = None) -> str:
        kw = dict(from_=self.sender, to=self.to, body=body)
        if media_url:
            kw["media_url"] = [media_url]
        return self.client.messages.create(**kw).sid

    def send_package(self, *, title: str, blogger_url: str, source_name: str, source_url: str,
                     fb: FacebookPackage, image_public_url: str, status: str) -> WhatsAppResult:
        header = (f"📰 Story\n\nTitle:\n{title}\n\nSource:\n{source_name}\n{source_url}\n\n"
                  f"Blogger:\n{blogger_url}\n\nStatus: {status}\nImage: "
                  f"{'attached (AI generated)' if image_public_url else 'not attached (no public URL)'}")
        fb_msg = f"Facebook Title:\n{fb.title}\n\nFacebook Post:\n{fb.post}"
        comment = f"First Comment:\n{fb.first_comment}"
        res = WhatsAppResult()
        try:
            res.message_ids.append(self._send(header, image_public_url or None))
            res.media_sent = bool(image_public_url)
            for part in [*_chunks(fb_msg), *_chunks(comment)]:
                res.message_ids.append(self._send(part))
            return res
        except TwilioRestException as exc:
            if exc.code in OUTSIDE_WINDOW_CODES and self.cfg.whatsapp_content_sid:
                logger.log("WHATSAPP", "outside 24h window -> using approved template")
                return self._send_template(title, blogger_url, res)
            if exc.code in OUTSIDE_WINDOW_CODES:
                raise WhatsAppError("outside the 24h window and WHATSAPP_CONTENT_SID is not set") from exc
            raise WhatsAppError(f"Twilio error {exc.code}") from exc

    def _send_template(self, title: str, blogger_url: str, res: WhatsAppResult) -> WhatsAppResult:
        try:
            m = self.client.messages.create(
                from_=self.sender, to=self.to, content_sid=self.cfg.whatsapp_content_sid,
                content_variables=json.dumps({"1": title[:200], "2": blogger_url}))
            res.message_ids.append(m.sid)
            res.used_template = True
            return res
        except TwilioRestException as exc:
            raise WhatsAppError(f"template send failed: {exc.code}") from exc
