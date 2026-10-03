#!/usr/bin/env python3
"""Cloudflare flux-2-klein-4b reference-image probe: warm-up, then small and large reference."""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from pathlib import Path

import requests
from PIL import Image, ImageDraw

OUT = Path("probe_out")
OUT.mkdir(exist_ok=True)

ACCOUNT = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "").strip()
TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "").strip()
TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "300"))

URL = f"https://api.cloudflare.com/client/v4/accounts/{ACCOUNT}/ai/run/@cf/black-forest-labs/flux-2-klein-4b"
HEADERS = {"Authorization": f"Bearer {TOKEN}"}

MODEL_ID = "@cf/black-forest-labs/flux-2-klein-4b"
RESULTS: list[dict] = []


def say(msg: str = "") -> None:
    print(msg, flush=True)


def make_reference(size: int) -> bytes:
    """Distinctive subject: a green teapot with a yellow stripe (easy to see if identity is kept)."""
    img = Image.new("RGB", (size, size), (225, 225, 230))
    d = ImageDraw.Draw(img)
    s = size / 768
    d.ellipse((190 * s, 260 * s, 580 * s, 600 * s), fill=(20, 120, 60))
    d.rectangle((190 * s, 400 * s, 580 * s, 440 * s), fill=(240, 200, 30))
    d.rectangle((330 * s, 215 * s, 440 * s, 275 * s), fill=(20, 120, 60))
    d.arc((540 * s, 330 * s, 680 * s, 500 * s), 270, 90, fill=(20, 120, 60), width=int(22 * s))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=88)
    return buf.getvalue()


def call(name: str, prompt: str, ref: bytes | None) -> None:
    say(f"[{name}]")
    start = time.time()

    form = {k: (None, str(v)) for k, v in {"prompt": prompt, "width": 1024, "height": 576}.items()}

    if ref:
        form["input_image_0"] = ("reference.jpg", ref, "image/jpeg")

    try:
        resp = requests.post(URL, headers=HEADERS, files=form, timeout=TIMEOUT)
        elapsed = time.time() - start

        if resp.status_code >= 400:
            detail = resp.text[:250].replace("\n", " ")
            RESULTS.append({"case": name, "status": f"HTTP_{resp.status_code}", "seconds": round(elapsed, 1), "detail": detail})
            say(f"  -> HTTP {resp.status_code} ({elapsed:.1f}s) {detail}")
            return

        ctype = (resp.headers.get("Content-Type") or "").lower()

        if ctype.startswith("image/"):
            data = resp.content
        else:
            body = resp.json()
            result = body.get("result", body)
            data = base64.b64decode(result["image"])

        img = Image.open(io.BytesIO(data))
        img.load()
        (OUT / f"{name}.jpg").write_bytes(data)
        info = f"{img.size[0]}x{img.size[1]}, {len(data) // 1024}KB"
        RESULTS.append({"case": name, "status": "OK", "seconds": round(elapsed, 1), "detail": info})
        say(f"  -> OK ({elapsed:.1f}s) {info}")

    except requests.Timeout:
        elapsed = time.time() - start
        RESULTS.append({"case": name, "status": "TIMEOUT", "seconds": round(elapsed, 1), "detail": ""})
        say(f"  -> TIMEOUT ({elapsed:.1f}s)")
    except Exception as exc:  # noqa: BLE001
        elapsed = time.time() - start
        RESULTS.append({"case": name, "status": f"ERROR ({type(exc).__name__})", "seconds": round(elapsed, 1), "detail": str(exc)[:200]})
        say(f"  -> ERROR {type(exc).__name__}: {str(exc)[:200]}")


def main() -> int:
    if not ACCOUNT or not TOKEN:
        say("CLOUDFLARE_ACCOUNT_ID / CLOUDFLARE_API_TOKEN is missing")
        return 2

    scene = (
        "Use the reference image: keep the same green teapot with the yellow stripe, "
        "but place it on a kitchen counter in warm morning light, camera from a low angle, no text."
    )

    (OUT / "reference_512.jpg").write_bytes(make_reference(512))

    call("warmup_text", "A cup of tea on a table, soft daylight, no text.", None)
    call("ref_512", scene, make_reference(512))
    call("ref_1024", scene, make_reference(1024))
    call("ref_512_again", scene, make_reference(512))

    say("")
    say("=" * 60)
    say("SUMMARY")
    say("=" * 60)

    for r in RESULTS:
        say(f"{r['case']:16} {r['seconds']:>7}s  {r['status']}  {r['detail'][:60] if r['status'] == 'OK' else ''}")

    (OUT / "ref_report.json").write_text(json.dumps(RESULTS, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    sys.exit(code)
