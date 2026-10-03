#!/usr/bin/env python3
"""Standalone Cloudflare Workers AI image probe. One request per case, no retries."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests

OUT = Path("probe_out")
OUT.mkdir(exist_ok=True)

ACCOUNT = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "120"))

BASE = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

PROMPT = (
    "A photorealistic photo of a red apple on a wooden table, soft daylight, "
    "no text, no watermark."
)
REF_PROMPT = (
    "Use the reference image: the same apple, but photographed from above on a blue plate, "
    "soft daylight, no text, no watermark."
)

RESULTS: list[dict] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)


def extract_image(resp: requests.Response) -> bytes:
    ctype = (resp.headers.get("Content-Type") or "").lower()

    if ctype.startswith("image/"):
        return resp.content

    data = resp.json()
    result = data.get("result", data) if isinstance(data, dict) else {}
    raw = result.get("image") if isinstance(result, dict) else None

    if not raw:
        raise RuntimeError(f"no image in response: {json.dumps(data)[:300]}")

    return base64.b64decode(raw)


def classify(status: int, body: str) -> str:
    low = body.lower()

    if status in (401, 403):
        return "AUTH_FAILED (token/permissions)"
    if status == 404:
        return "NOT_FOUND (model name or account id)"
    if status == 429 or "limit" in low or "quota" in low or "neuron" in low:
        return f"QUOTA/RATE ({status})"
    if status == 400:
        return "BAD_REQUEST (400)"
    if status >= 500:
        return f"SERVER ({status})"
    return f"HTTP_{status}"


def save_and_describe(data: bytes, name: str) -> str:
    from PIL import Image

    img = Image.open(io.BytesIO(data))
    img.load()
    ext = "png" if (img.format or "").upper() == "PNG" else "jpg"
    (OUT / f"{name}.{ext}").write_bytes(data)
    return f"{img.size[0]}x{img.size[1]} {img.format}, {len(data) // 1024}KB"


def run_case(name: str, model: str, *, json_body=None, form=None, files=None):
    say(f"[{name}] {model}")
    start = time.time()

    try:
        if json_body is not None:
            resp = requests.post(BASE + model, headers=HEADERS, json=json_body, timeout=TIMEOUT)
        else:
            multipart = {k: (None, str(v)) for k, v in (form or {}).items()}
            multipart.update(files or {})
            resp = requests.post(BASE + model, headers=HEADERS, files=multipart, timeout=TIMEOUT)

        elapsed = time.time() - start

        if resp.status_code >= 400:
            status = classify(resp.status_code, resp.text)
            detail = resp.text[:250].replace("\n", " ")
            RESULTS.append({"case": name, "model": model, "status": status, "seconds": round(elapsed, 1), "detail": detail})
            say(f"  -> {status} ({elapsed:.1f}s) {detail}")
            return None

        data = extract_image(resp)
        info = save_and_describe(data, name)
        RESULTS.append({"case": name, "model": model, "status": "OK", "seconds": round(elapsed, 1), "detail": info})
        say(f"  -> OK ({elapsed:.1f}s) {info}")
        return data

    except requests.Timeout:
        elapsed = time.time() - start
        RESULTS.append({"case": name, "model": model, "status": "TIMEOUT", "seconds": round(elapsed, 1), "detail": ""})
        say(f"  -> TIMEOUT ({elapsed:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        RESULTS.append({"case": name, "model": model, "status": f"ERROR ({type(exc).__name__})", "seconds": round(elapsed, 1), "detail": str(exc)[:250]})
        say(f"  -> ERROR {type(exc).__name__}: {str(exc)[:250]}")

    return None


def synthetic_reference() -> bytes:
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (768, 768), (235, 228, 215))
    draw = ImageDraw.Draw(img)
    draw.ellipse((200, 220, 568, 590), fill=(190, 30, 35))
    draw.rectangle((375, 150, 395, 230), fill=(80, 50, 25))
    buf = io.BytesIO()
    img.save(buf, "PNG")
    return buf.getvalue()


def main() -> int:
    if not ACCOUNT or not TOKEN:
        say("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN is missing")
        return 2

    say("=" * 70)
    say("CLOUDFLARE WORKERS AI IMAGE PROBE")
    say("=" * 70)

    schnell = run_case(
        "schnell_text",
        "@cf/black-forest-labs/flux-1-schnell",
        json_body={"prompt": PROMPT, "steps": 4},
    )

    run_case(
        "klein4b_text",
        "@cf/black-forest-labs/flux-2-klein-4b",
        form={"prompt": PROMPT, "width": 1024, "height": 576},
    )

    reference = schnell or synthetic_reference()

    run_case(
        "klein4b_reference",
        "@cf/black-forest-labs/flux-2-klein-4b",
        form={"prompt": REF_PROMPT, "width": 1024, "height": 576},
        files={"input_image_0": ("reference.png", reference, "image/png")},
    )

    say("")
    say("=" * 70)
    say("SUMMARY")
    say("=" * 70)
    say(f"{'CASE':20} {'SEC':>6}  STATUS")

    for r in RESULTS:
        say(f"{r['case']:20} {r['seconds']:>6}  {r['status']}  {r['detail'][:60] if r['status'] == 'OK' else ''}")

    (OUT / "cloudflare_report.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")

    ok = [r for r in RESULTS if r["status"] == "OK"]
    say("")
    say(f"WORKING CASES: {len(ok)}/{len(RESULTS)}")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.exit(code)
