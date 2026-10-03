#!/usr/bin/env python3
"""Ask Gemini which image models it knows, cross-check with the real model list, then test each one."""
from __future__ import annotations

import io
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path("probe_out")
OUT.mkdir(exist_ok=True)

ASK_MODEL = os.environ.get("ASK_MODEL", "gemini-3.5-flash-lite")
TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "45"))

EXCLUDE = (
    "tts", "embedding", "live", "transcribe", "robotics", "computer-use",
    "native-audio", "translate", "aqa", "veo", "lyria",
)


def say(msg: str = "") -> None:
    print(msg, flush=True)


def with_timeout(fn, seconds: float, label: str):
    box: dict = {}

    def runner() -> None:
        try:
            box["value"] = fn()
        except BaseException as exc:  # noqa: BLE001
            box["error"] = exc

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join(seconds)

    if worker.is_alive():
        raise TimeoutError(f"no answer within {int(seconds)}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def classify(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    if isinstance(exc, TimeoutError):
        return "TIMEOUT"
    if "limit: 0" in low:
        return "NO_FREE_QUOTA (limit 0)"
    if code == 429 or "resource_exhausted" in low:
        return "QUOTA/RATE (429)"
    if code == 503 or "unavailable" in low:
        return "OVERLOADED (503)"
    if code == 404 or "not found" in low:
        return "NOT_FOUND (404)"
    if code in (400, 403):
        return f"REJECTED ({code})"
    return f"ERROR ({type(exc).__name__})"


def short(name: str) -> str:
    return name[len("models/"):] if name.startswith("models/") else name


# ---------------------------------------------------------------- step 1
def ask_gemini(client, real_names: set[str]) -> list[str]:
    say("=" * 70)
    say(f"1) ASKING {ASK_MODEL} (answers may be outdated or invented)")
    say("=" * 70)

    prompt = (
        "I use the Gemini Developer API with an AI Studio API key (free tier). "
        "List every Google model name (exact API model IDs) that can generate images "
        "through generateContent or any other Gemini API method, including previews. "
        "Return ONLY a JSON array of strings, no explanation."
    )

    try:
        response = with_timeout(
            lambda: client.models.generate_content(model=ASK_MODEL, contents=prompt),
            60,
            "ask",
        )
        text = (getattr(response, "text", "") or "").strip()
        say(f"raw answer: {text[:600]}")

        match = re.search(r"\[.*\]", text, re.S)
        names = json.loads(match.group(0)) if match else []
        names = [short(str(n).strip()) for n in names if str(n).strip()]
    except Exception as exc:  # noqa: BLE001
        say(f"could not ask Gemini: {classify(exc)} {str(exc)[:200]}")
        return []

    say("")
    for n in names:
        mark = "exists in your list" if n in real_names else "NOT in your list (likely invented/outdated)"
        say(f"   {n:45} {mark}")

    return names


# ---------------------------------------------------------------- step 2
def list_models(client) -> dict[str, dict]:
    say("")
    say("=" * 70)
    say("2) REAL MODEL LIST FOR THIS KEY")
    say("=" * 70)

    rows: dict[str, dict] = {}

    for m in client.models.list():
        name = short(getattr(m, "name", "") or "")
        actions = [str(a) for a in (getattr(m, "supported_actions", None) or [])]
        rows[name] = {
            "actions": actions,
            "description": (getattr(m, "description", "") or "")[:140],
        }

    say(f"total models: {len(rows)}")
    return rows


def pick_candidates(rows: dict[str, dict], suggested: list[str]) -> list[str]:
    out: list[str] = []

    for name, info in rows.items():
        low = name.lower()

        if "generateContent" not in info["actions"]:
            continue
        if any(word in low for word in EXCLUDE):
            continue
        if low.startswith("gemma"):
            continue

        out.append(name)

    for name in suggested:
        if name in rows and name not in out:
            out.append(name)

    return out


# ---------------------------------------------------------------- step 3
def try_image(client, model: str):
    from google.genai import types

    def run(modalities):
        config = types.GenerateContentConfig(response_modalities=modalities)
        response = client.models.generate_content(
            model=model,
            contents="A photorealistic photo of a red apple on a wooden table, soft daylight, no text.",
            config=config,
        )

        parts = getattr(response, "parts", None)

        if not parts:
            for cand in getattr(response, "candidates", None) or []:
                parts = getattr(getattr(cand, "content", None), "parts", None)
                if parts:
                    break

        for part in parts or []:
            inline = getattr(part, "inline_data", None)
            data = getattr(inline, "data", None) if inline is not None else None
            if data:
                return bytes(data), getattr(inline, "mime_type", None) or "image/png"

        raise RuntimeError("model answered but returned no image")

    try:
        return run(["IMAGE"])
    except Exception as exc:  # noqa: BLE001
        low = str(exc).lower()
        if "modalit" in low or "invalid_argument" in low:
            return run(["TEXT", "IMAGE"])
        raise


def test_one(client, model: str) -> dict:
    start = time.time()

    try:
        data, mime = with_timeout(lambda: try_image(client, model), TIMEOUT, model)

        info = "saved"
        try:
            from PIL import Image

            img = Image.open(io.BytesIO(data))
            img.load()
            ext = "png" if "png" in mime else "jpg"
            (OUT / f"{model}.{ext}").write_bytes(data)
            info = f"{img.size[0]}x{img.size[1]}, {len(data) // 1024}KB"
        except Exception as exc:  # noqa: BLE001
            info = f"returned bytes but unreadable ({type(exc).__name__})"

        return {"model": model, "status": "OK", "seconds": round(time.time() - start, 1), "detail": info}

    except Exception as exc:  # noqa: BLE001
        return {
            "model": model,
            "status": classify(exc),
            "seconds": round(time.time() - start, 1),
            "detail": str(exc)[:160].replace("\n", " "),
        }


def main() -> int:
    key = os.environ.get("IMAGE_API_KEY") or os.environ.get("GEMINI_API_KEY")

    if not key:
        say("GEMINI_API_KEY is missing")
        return 2

    from google import genai

    client = genai.Client(api_key=key)

    rows = list_models(client)
    suggested = ask_gemini(client, set(rows))
    candidates = pick_candidates(rows, suggested)

    say("")
    say("=" * 70)
    say(f"3) TESTING {len(candidates)} CANDIDATES (1 request each, parallel)")
    say("=" * 70)

    with ThreadPoolExecutor(max_workers=6) as pool:
        results = list(pool.map(lambda m: test_one(client, m), candidates))

    say("")
    say("=" * 70)
    say("SUMMARY")
    say("=" * 70)
    say(f"{'MODEL':48} {'SEC':>6}  STATUS")

    for r in sorted(results, key=lambda r: (r["status"] != "OK", r["model"])):
        say(f"{r['model'][:48]:48} {r['seconds']:>6}  {r['status']}  {r['detail'][:70] if r['status'] == 'OK' else ''}")

    working = [r for r in results if r["status"] == "OK"]
    say("")
    say(f"WORKING IMAGE MODELS: {len(working)}")

    for r in working:
        say(f"   {r['model']}  ({r['seconds']}s, {r['detail']})")

    if not working:
        say("No Gemini model can generate images for this key on the current plan.")

    (OUT / "models_report.json").write_text(
        json.dumps({"suggested": suggested, "results": results}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    os._exit(code)
