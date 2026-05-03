import base64
import re
import time
from datetime import datetime
from io import BytesIO

import yaml
from openai import OpenAI
from PIL import Image

from sentinel.models import Note, NoteAnalysis
from sentinel.settings import settings


SYSTEM_PROMPT = """You are observing a screenshot of someone's laptop screen.
Extract structured information about what is currently happening on screen.

Return ONLY a YAML document inside a ```yaml fenced block. No prose before or after.

The YAML document must have exactly these top-level keys, in this order:
- app:     a string — the app, website, or window in focus
- ocr:     a multi-line string using the `|` block scalar — verbatim text on screen
- summary: a multi-line string using the `|` block scalar — 3-6 sentence interpretation
- people:  a YAML list of strings — names of people visible (or [] if none)
- urgency: one of: low, medium, high

Example of the exact format expected:

```yaml
app: Slack
ocr: |
  #general
  John: Hey team, status update?
  Sarah: All good on the API rollout.
  Notifications (3)
summary: |
  The user is reading the #general channel in Slack. John asked for a
  status update and Sarah replied that the API rollout is on track.
  Three unread notifications are visible in the sidebar.
people:
  - John
  - Sarah
urgency: low
```

Rules for `ocr` — verbatim text extraction:
- Transcribe every readable piece of text on screen exactly as written.
- Read top-to-bottom, left-to-right. Preserve rough grouping with newlines
  (e.g. menu items on separate lines, message bubbles separated, columns flowing).
- Keep numbers, dates, urls, code, prices, button labels, badge counts EXACT.
- Do not paraphrase, translate, summarise, or correct typos.
- Skip purely decorative elements (icons with no text). Skip text that is too blurry to read.

Rules for `summary` — interpretive 3-6 sentences:
- Higher-level than `ocr`: explain what the user is doing, with whom, and why it matters.
- State who is interacting with whom, in which channel/thread/document.
- Mention notable UI state (modal open, error banner, unread badge count).
- Note what action (if any) the user appears to be taking
  (composing, scrolling, reading, debugging).
- Do NOT speculate about what is off-screen or what the user might do next.

Other rules:
- Only describe what is actually visible. Do not invent details.
- `people` is `[]` if no names are visible.
- `urgency` is 'high' only if something clearly demands action soon
  (urgent message, error, deadline visible). Otherwise 'low'.
"""


_client = OpenAI(base_url=settings.GEMMA_BASE_URL, api_key="not-needed")


def _detect_mime(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
        return "image/webp"
    return "image/png"


def _to_data_url(image_bytes: bytes) -> str:
    mime = _detect_mime(image_bytes)
    b64 = base64.b64encode(image_bytes).decode()
    return f"data:{mime};base64,{b64}"


def _downscale(image_bytes: bytes, max_width: int = settings.MAX_IMAGE_WIDTH) -> bytes:
    """Resize a screenshot down to max_width preserving aspect ratio.

    Native-res Mac screenshots can be 5-6K wide; the VLM has to encode every
    pixel into image tokens, which dominates prefill time. Downscaling to
    ~1280px gives a large speed-up with negligible OCR-quality loss because
    Gemma's image encoder downsamples internally anyway.
    """
    img = Image.open(BytesIO(image_bytes))
    if img.width <= max_width:
        return image_bytes
    ratio = max_width / img.width
    new_size = (max_width, int(img.height * ratio))
    img = img.resize(new_size, Image.LANCZOS)
    if img.mode != "RGB":
        img = img.convert("RGB")
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


def _extract_yaml(text: str) -> str:
    """Pull YAML body out of a ```yaml ... ``` fenced block. Falls back to raw text."""
    match = re.search(r"```(?:yaml|yml)?\s*\n(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1)
    return text.strip()


def analyze_screenshot(image_bytes: bytes) -> Note:
    """Send a screenshot to Gemma 4 and return a stamped Note. Streams to stdout."""
    t0 = time.perf_counter()
    original_kb = len(image_bytes) / 1024
    image_bytes = _downscale(image_bytes)
    t1 = time.perf_counter()
    print(
        f"[debug] downscale {original_kb:.0f}KB -> {len(image_bytes) / 1024:.0f}KB "
        f"in {(t1 - t0) * 1000:.0f}ms",
        flush=True,
    )

    print(f"[debug] sending to {settings.GEMMA_BASE_URL} ...", flush=True)
    print(f"[debug] system prompt is {len(SYSTEM_PROMPT)} chars", flush=True)
    print(f"[debug] model name is {settings.GEMMA_MODEL}", flush=True)
    t_request = time.perf_counter()
    stream = _client.chat.completions.create(
        model=settings.GEMMA_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _to_data_url(image_bytes)}},
                    {"type": "text", "text": "Analyze this screenshot."},
                ],
            },
        ],
        max_tokens=2048,
        temperature=0.1,
        stream=True,
    )

    first_token_at: float | None = None
    parts: list[str] = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_at is None:
                first_token_at = time.perf_counter()
                print(
                    f"[debug] first token after {first_token_at - t_request:.1f}s\n",
                    flush=True,
                )
            print(delta, end="", flush=True)
            parts.append(delta)
    t_done = time.perf_counter()
    print()
    total_chars = sum(len(p) for p in parts)
    gen_time = t_done - (first_token_at or t_request)
    print(
        f"[debug] generated {total_chars} chars in {gen_time:.1f}s "
        f"(~{total_chars / gen_time:.0f} chars/s, total {t_done - t_request:.1f}s)",
        flush=True,
    )

    content = "".join(parts)
    if not content.strip():
        raise RuntimeError(
            "Gemma returned 0 content tokens. The MLX server likely crashed silently "
            "(known cache bug). Restart it with: "
            "uv run mlx_vlm.server --model <name> --host 0.0.0.0 --port 7500"
        )

    yaml_str = _extract_yaml(content)
    parsed = yaml.safe_load(yaml_str)
    if not isinstance(parsed, dict):
        raise RuntimeError(
            f"Gemma response did not parse to a YAML mapping. Raw output:\n{content[:500]!r}"
        )
    analysis = NoteAnalysis.model_validate(parsed)
    return Note(ts=datetime.now(), **analysis.model_dump())
