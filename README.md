# Gemma 4 on MLX — Local Server Guide

Run Google's Gemma 4 (26B-A4B MoE, 4B active) locally on Apple Silicon with an OpenAI-compatible HTTP endpoint. Supports text, vision (image input), streaming, and TurboQuant KV-cache compression.

---

## 1. Prerequisites

- Apple Silicon Mac (M1 or newer; M3/M4 best for `mxfp8`)
- ≥ 36 GB unified memory (for the `mxfp8` weights)
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) installed

## 2. Install dependencies

From this project directory:

```bash
uv add mlx mlx-lm mlx-vlm
```

This pins them in `pyproject.toml` and installs into `.venv/`.

## 3. Pick a model

Default: `mlx-community/gemma-4-26b-a4b-it-mxfp8` (~27 GB, best quality on M3/M4).

Alternatives if RAM-constrained:

| Model ID | Size | Min RAM |
|---|---|---|
| `mlx-community/gemma-4-26b-a4b-it-mxfp8` | ~27 GB | 36 GB |
| `mlx-community/gemma-4-26b-a4b-it-4bit` | ~14 GB | 24 GB |
| `mlx-community/gemma-4-26B-A4B-it-heretic-msq-2.6bit` | ~9 GB | 16 GB |

First run auto-downloads to `~/.cache/huggingface/hub/`.

---

## 4. Start the server

### Text-only (faster startup)

```bash
uv run mlx_lm.server \
  --model mlx-community/gemma-4-26b-a4b-it-mxfp8 \
  --host 0.0.0.0 \
  --port 7500
```

### Vision + text (use this for images)

```bash
uv run mlx_vlm.server \
  --model mlx-community/gemma-4-26b-a4b-it-mxfp8 \
  --host 0.0.0.0 \
  --port 7500
```

### With TurboQuant (compresses KV cache → fit longer context in same RAM)

Add these flags to either server:

```bash
  --kv-bits 3.5 \
  --kv-quant-scheme turboquant
```

Server is ready when `curl http://localhost:7500/v1/models` returns JSON.

---

## 5. Inference with curl

The endpoint is OpenAI-compatible (`/v1/chat/completions`).

### Plain text

```bash
curl http://localhost:7500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-mxfp8",
    "messages": [{"role":"user","content":"Write a haiku about the ocean."}],
    "max_tokens": 100
  }'
```

### Streaming (token-by-token)

Add `"stream": true` and use `curl -N` to disable buffering:

```bash
curl -N http://localhost:7500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-mxfp8",
    "messages": [{"role":"user","content":"Tell me a story in 3 paragraphs."}],
    "max_tokens": 500,
    "stream": true
  }'
```

### Image input (requires `mlx_vlm.server`)

**From a URL:**

```bash
curl -N http://localhost:7500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "mlx-community/gemma-4-26b-a4b-it-mxfp8",
    "stream": true,
    "max_tokens": 400,
    "messages": [{
      "role": "user",
      "content": [
        {"type": "text", "text": "What is in this image?"},
        {"type": "image_url", "image_url": {"url": "https://images.unsplash.com/photo-1514888286974-6c03e2ca1dba"}}
      ]
    }]
  }'
```

**From a local file** (base64-encoded inline):

```bash
IMG=/Users/keshav/Pictures/your-image.jpg
B64=$(base64 -i "$IMG")

curl -N http://localhost:7500/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"mlx-community/gemma-4-26b-a4b-it-mxfp8\",
    \"stream\": true,
    \"max_tokens\": 400,
    \"messages\": [{
      \"role\": \"user\",
      \"content\": [
        {\"type\": \"text\", \"text\": \"Describe this image.\"},
        {\"type\": \"image_url\", \"image_url\": {\"url\": \"data:image/jpeg;base64,$B64\"}}
      ]
    }]
  }"
```

(Use `image/png` instead of `image/jpeg` for PNGs.)

---

## 6. Python client (optional)

Any OpenAI SDK works — just point `base_url` at the local server:

```bash
uv run --with openai python -c '
from openai import OpenAI
client = OpenAI(base_url="http://localhost:7500/v1", api_key="x")
stream = client.chat.completions.create(
    model="mlx-community/gemma-4-26b-a4b-it-mxfp8",
    messages=[{"role":"user","content":"hi"}],
    stream=True,
)
for chunk in stream:
    print(chunk.choices[0].delta.content or "", end="", flush=True)
print()
'
```

The `api_key` value is ignored by the local server but the SDK requires the field.

---

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `mlx_lm.server` errors on image input | Switch to `mlx_vlm.server` — text server can't handle images |
| Streaming fails with generic `APIError` | Run the same request **without** `stream` to see the real server error |
| Wikimedia image URL gives `403 Forbidden` | Wikimedia blocks requests without a User-Agent — use a different host or base64 a local file |
| Local file `FileNotFoundError` despite correct-looking path | macOS screenshots use a **narrow no-break space** (U+202F) in filenames. Use a glob (`glob.glob("Screenshot*12.59.07*.jpg")[0]`) or rename the file |
| Server hangs after "Fetching N files" | Loading 27 GB into memory takes 30–90s. Test with `curl /v1/models` to check readiness |
| Out of memory / heavy swapping | Drop to `mlx-community/gemma-4-26b-a4b-it-4bit` (~14 GB) |
| `--kv-quant-scheme` unrecognized | `uv add --upgrade mlx-lm mlx-vlm` — TurboQuant landed in early 2026 |

---

## 8. Quick reference

| Command | Purpose |
|---|---|
| `uv run mlx_lm.server ...` | Text-only server |
| `uv run mlx_vlm.server ...` | Vision + text server |
| `curl http://localhost:7500/v1/models` | Health check |
| `curl http://localhost:7500/v1/chat/completions` | Chat endpoint (OpenAI-compatible) |
| `--kv-bits 3.5 --kv-quant-scheme turboquant` | Enable TurboQuant KV compression |
| `"stream": true` + `curl -N` | Token-by-token streaming |
