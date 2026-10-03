#!/usr/bin/env python3
"""Standalone Gemini probe: which image/text models and call styles really work for this key.

One request per (model, method). No retries. Never prints the API key.
"""
from __future__ import annotations

import base64
import io
import json
import os
import sys
import threading
import time
from pathlib import Path

OUT = Path("probe_out")
OUT.mkdir(exist_ok=True)

TIMEOUT = int(os.environ.get("PROBE_TIMEOUT", "90"))

IMAGE_MODELS = [
    m.strip()
    for m in os.environ.get(
        "PROBE_IMAGE_MODELS",
        "gemini-3.1-flash-lite-image,gemini-3.1-flash-image,imagen-3.0-generate-002",
    ).split(",")
    if m.strip()
]
TEXT_MODELS = [
    m.strip()
    for m in os.environ.get("PROBE_TEXT_MODELS", "gemini-3.5-flash-lite,gemini-2.5-flash").split(",")
    if m.strip()
]
METHODS = [
    m.strip()
    for m in os.environ.get("PROBE_METHODS", "generate_content,interactions").split(",")
    if m.strip()
]

PROMPT = (
    "A photorealistic photo of a red apple on a wooden table, soft daylight, "
    "no text, no watermark."
)

RESULTS: list[dict] = []


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
        raise TimeoutError(f"{label} gave no answer within {int(seconds)}s")
    if "error" in box:
        raise box["error"]
    return box.get("value")


def classify(exc: BaseException) -> str:
    msg = str(exc)
    low = msg.lower()
    code = getattr(exc, "code", None) or getattr(exc, "status_code", None)

    if isinstance(exc, TimeoutError):
        return "TIMEOUT (no answer)"
    if "limit: 0" in low:
        return "NO_FREE_QUOTA (limit 0 -> needs billing)"
    if code == 429 or "resource_exhausted" in low:
        return "QUOTA/RATE (429)"
    if code == 503 or "unavailable" in low:
        return "OVERLOADED (503)"
    if code == 404 or "not found" in low:
        return "MODEL_NOT_FOUND (404)"
    if code in (400, 403) or "permission" in low or "invalid" in low:
        return f"REJECTED ({code})"
    if isinstance(exc, AttributeError):
        return "NOT_AVAILABLE_IN_SDK"
    return f"ERROR ({type(exc).__name__})"


def save_image(data: bytes, mime: str, name: str) -> str:
    try:
        from PIL import Image

        img = Image.open(io.BytesIO(data))
        img.load()
        ext = "png" if "png" in (mime or "") else "jpg"
        path = OUT / f"{name}.{ext}"
        path.write_bytes(data)
        return f"{img.size[0]}x{img.size[1]}, {len(data) // 1024}KB"
    except Exception as exc:  # noqa: BLE001
        return f"saved but unreadable ({type(exc).__name__})"


def record(kind: str, model: str, method: str, status: str, seconds: float, detail: str) -> None:
    RESULTS.append(
        {
            "kind": kind,
            "model": model,
            "method": method,
            "status": status,
            "seconds": round(seconds, 1),
            "detail": detail[:300],
        }
    )
    say(f"  -> {status} ({seconds:.1f}s) {detail[:200]}")


# ---------------------------------------------------------------- calls
def call_generate_content_image(client, model: str):
    from google.genai import types

    kwargs = {"response_modalities": ["IMAGE"]}
    try:
        kwargs["image_config"] = types.ImageConfig(aspect_ratio="16:9")
        config = types.GenerateContentConfig(**kwargs)
    except Exception:  # noqa: BLE001
        kwargs.pop("image_config", None)
        config = types.GenerateContentConfig(**kwargs)

    response = client.models.generate_content(model=model, contents=PROMPT, config=config)

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

    text = (getattr(response, "text", "") or "")[:150]
    raise RuntimeError(f"no image in response; text={text!r}")


def call_interactions_image(client, model: str):
    interactions = getattr(client, "interactions", None)
    if interactions is None or not hasattr(interactions, "create"):
        raise AttributeError("client.interactions missing in this google-genai version")

    interaction = interactions.create(
        model=model if model.startswith("models/") else f"models/{model}",
        input=PROMPT,
        generation_config={"temperature": 1, "top_p": 0.95, "thinking_level": "minimal"},
        response_modalities=["image", "text"],
    )

    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", "") != "model_output" or not getattr(step, "content", None):
            continue
        for part in step.content:
            if getattr(part, "type", "") == "image":
                raw = getattr(part, "data", None)
                if raw:
                    data = base64.b64decode(raw) if isinstance(raw, str) else bytes(raw)
                    return data, getattr(part, "mime_type", None) or "image/png"

    raise RuntimeError("no image part in interaction steps")


def call_imagen(client, model: str):
    from google.genai import types

    try:
        config = types.GenerateImagesConfig(
            number_of_images=1, aspect_ratio="16:9", output_mime_type="image/jpeg"
        )
    except Exception:  # noqa: BLE001
        config = types.GenerateImagesConfig(number_of_images=1, output_mime_type="image/jpeg")

    response = client.models.generate_images(model=model, prompt=PROMPT, config=config)
    images = getattr(response, "generated_images", None) or []
    if images:
        data = getattr(getattr(images[0], "image", None), "image_bytes", None)
        if data:
            return bytes(data), "image/jpeg"
    raise RuntimeError("no generated_images in response")


def call_text(client, model: str):
    response = client.models.generate_content(model=model, contents="Reply with the single word: OK")
    return (getattr(response, "text", "") or "").strip()[:60]


# ---------------------------------------------------------------- runners
def probe_models_list(client) -> None:
    say("=" * 70)
    say("1) MODELS AVAILABLE TO THIS KEY (image-related first)")
    say("=" * 70)
    try:
        rows = []
        for m in client.models.list():
            name = getattr(m, "name", "") or ""
            actions = getattr(m, "supported_actions", None) or getattr(
                m, "supported_generation_methods", None
            ) or []
            rows.append((name, ",".join(str(a) for a in actions)))

        image_rows = [r for r in rows if "image" in r[0].lower() or "imagen" in r[0].lower()]
        gemini_rows = [
            r for r in rows if "gemini" in r[0].lower() and r not in image_rows
        ]

        say(f"total models: {len(rows)}")
        say("-- image models:")
        for name, actions in image_rows:
            say(f"   {name}   [{actions}]")
        say("-- gemini text models:")
        for name, _ in gemini_rows[:40]:
            say(f"   {name}")
    except Exception as exc:  # noqa: BLE001
        say(f"could not list models: {classify(exc)} {str(exc)[:200]}")


def probe_text(client) -> None:
    say("")
    say("=" * 70)
    say("2) TEXT MODELS")
    say("=" * 70)
    for model in TEXT_MODELS:
        say(f"[text] {model}")
        start = time.time()
        try:
            out = with_timeout(lambda m=model: call_text(client, m), 60, "text")
            record("text", model, "generate_content", "OK", time.time() - start, f"reply={out!r}")
        except Exception as exc:  # noqa: BLE001
            record("text", model, "generate_content", classify(exc), time.time() - start, str(exc))


def probe_images(client) -> None:
    say("")
    say("=" * 70)
    say("3) IMAGE MODELS (one request each, no retries)")
    say("=" * 70)

    for model in IMAGE_MODELS:
        is_imagen = "imagen" in model.lower()
        methods = ["generate_images"] if is_imagen else METHODS

        for method in methods:
            say(f"[image] {model} via {method}")
            start = time.time()
            try:
                if method == "generate_images":
                    fn = lambda m=model: call_imagen(client, m)  # noqa: E731
                elif method == "interactions":
                    fn = lambda m=model: call_interactions_image(client, m)  # noqa: E731
                else:
                    fn = lambda m=model: call_generate_content_image(client, m)  # noqa: E731

                data, mime = with_timeout(fn, TIMEOUT, method)
                safe = f"{model.replace('/', '_')}__{method}"
                info = save_image(data, mime, safe)
                record("image", model, method, "OK", time.time() - start, info)
            except Exception as exc:  # noqa: BLE001
                record("image", model, method, classify(exc), time.time() - start, str(exc))


def print_summary() -> None:
    say("")
    say("=" * 70)
    say("SUMMARY")
    say("=" * 70)
    say(f"{'KIND':6} {'MODEL':36} {'METHOD':17} {'SEC':>6}  STATUS")
    for r in RESULTS:
        say(f"{r['kind']:6} {r['model'][:36]:36} {r['method']:17} {r['seconds']:>6}  {r['status']}")

    working = [r for r in RESULTS if r["kind"] == "image" and r["status"] == "OK"]
    say("")
    if working:
        best = min(working, key=lambda r: r["seconds"])
        say(f"WORKING IMAGE ROUTES: {len(working)}")
        say(f"FASTEST: {best['model']} via {best['method']} ({best['seconds']}s)")
    else:
        say("NO IMAGE ROUTE WORKED FOR THIS KEY.")

    (OUT / "report.json").write_text(json.dumps(RESULTS, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    key = os.environ.get("IMAGE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not key:
        say("GEMINI_API_KEY is missing")
        return 2

    from google import genai

    try:
        import importlib.metadata as md

        say(f"google-genai version: {md.version('google-genai')}")
    except Exception:  # noqa: BLE001
        pass

    client = genai.Client(api_key=key)

    probe_models_list(client)
    probe_text(client)
    probe_images(client)
    print_summary()
    return 0


if __name__ == "__main__":
    code = main()
    sys.stdout.flush()
    # Stuck worker threads are daemons; force exit so the job never hangs.
    os._exit(code)
