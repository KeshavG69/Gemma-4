# Sentinel

**Your day, remembered locally — and only locally.**

Sentinel watches your screens every few minutes, builds a private memory of what you actually did, and lets you ask it anything later. Everything runs on your laptop. No screenshot ever touches disk. No data ever leaves your machine.

Built with **Gemma 4** (Apple Silicon via MLX), **Agno** for the agent, and **NiceGUI** for the desktop window.

---

## Why?

Tools like Loom AI, Rewind, and Notion AI promise to "remember everything you did" — but they ship your screen contents to a cloud they own. For anyone with sensitive code, customer data, NDAs, or just an aversion to handing over their digital life, that's a non-starter.

Sentinel does the same thing, except:

- **The screenshot lives in RAM for ~2 seconds** while Gemma 4 reads it, then it's gone.
- **Only structured text notes** are persisted to a local SQLite file.
- If your laptop is stolen, there are zero screenshots to recover.

The whole pipeline — vision, reasoning, search — runs on the same machine as your work.

---

## What it does

Every N minutes (default 5), Sentinel:

1. Captures **all your monitors** (in memory only).
2. Sends each screenshot to a local **Gemma 4 vision model**.
3. Extracts a structured note:
   - `app` — app or window in focus
   - `ocr` — verbatim text on screen
   - `summary` — a 3–6 sentence interpretive description
   - `people` — names visible in DMs / emails / video calls
   - `urgency` — `low | medium | high`
4. Stores the note in a local **SQLite** database.
5. Drops the image. Forever.

Then, in the desktop UI, you can:

- **Ask anything** in plain English — *"when did I last open Slack?"*, *"what was I doing at 1 AM?"*, *"any high-urgency moments today?"* — answered by an Agno agent that writes its own SQL against the notes table.
- **Generate a briefing** — a streamed 2–3 paragraph summary of everything Sentinel saw today.

---

## Architecture

```
                      ┌──────────────────────────────┐
                      │  Screen capture (mss)        │
                      │  All monitors, every 5 min   │
                      └────────────────┬─────────────┘
                                       │ PNG bytes (in RAM only)
                                       ▼
                      ┌──────────────────────────────┐
                      │  Gemma 4 VLM via MLX server  │
                      │  Extracts structured note    │
                      │  (YAML, then validated)      │
                      └────────────────┬─────────────┘
                                       │ Note (pydantic)
                                       ▼
                      ┌──────────────────────────────┐
                      │  SQLite — notes table        │
                      │  No images, no embeddings,   │
                      │  no third-party services     │
                      └────────────────┬─────────────┘
                                       │
              ┌────────────────────────┼────────────────────────┐
              ▼                        ▼                        ▼
   ┌──────────────────┐     ┌──────────────────┐     ┌──────────────────┐
   │  Agno SQL agent  │     │  Digest module   │     │  Stats / charts  │
   │  Q&A in plain    │     │  Daily briefing  │     │  (UI live cards) │
   │  English         │     │  via Gemma 4     │     │                  │
   └────────┬─────────┘     └────────┬─────────┘     └────────┬─────────┘
            │                        │                        │
            └────────────────────────┴────────────────────────┘
                                     │
                                     ▼
                      ┌──────────────────────────────┐
                      │  NiceGUI desktop window      │
                      │  Streamed responses, status, │
                      │  suggested questions         │
                      └──────────────────────────────┘
```

---

## Project layout

```
sentinel/
├── settings.py    # env-driven config
├── models.py      # Note + NoteAnalysis pydantic schemas
├── storage.py     # SQLite CRUD (init_db, insert_note, recent_notes, notes_on_date)
├── analyze.py     # screenshot → MLX server → Note (streaming, downscaled)
├── capture.py     # main capture loop, multi-monitor
├── agent.py       # Agno agent with a single read-only SQL tool
├── digest.py      # streamed end-of-day briefing
├── streaming.py   # forwards Agno events as JSON dict events
└── ui.py          # NiceGUI desktop window
```

---

## Tech stack

| Layer | Tool | Why |
|---|---|---|
| Vision model | **Gemma 4 (E4B-it, MXFP8)** | Open, multimodal, ~5 GB on disk, near-FP16 quality |
| Inference | **MLX + mlx-vlm** | Apple Silicon native, 15-30× faster than CPU |
| Agent | **Agno** | Native tool calling, OpenAI-compatible model adapter |
| UI | **NiceGUI** | Modern Material Design (Quasar), runs as a real desktop window |
| Storage | **SQLite** | Bundled with macOS, zero setup, single file |
| Capture | **mss + Pillow** | ~50ms grab, in-memory PNG, downscales to 512px before inference |
| Config | **pydantic-settings** | Single `.env` for all knobs |

---

## Setup

**Prerequisites**
- macOS with Apple Silicon (M1 / M2 / M3 / M4)
- Python 3.11
- [`uv`](https://docs.astral.sh/uv/) for dep management
- ~10 GB disk for the Gemma 4 model

```bash
git clone <this repo>
cd Gemma4
uv sync
```

Create a `.env` at the project root:

```bash
GEMMA_BASE_URL=http://localhost:7500/v1
GEMMA_MODEL=mlx-community/gemma-4-e4b-it-mxfp8
DB_PATH=sentinel.db
CAPTURE_INTERVAL_SECONDS=300
MAX_IMAGE_WIDTH=512
```

> The MLX server reports model names lowercase. Match the case exactly or requests silently fail.

---

## Run

You need three terminals.

### 1. Start the local Gemma 4 server

```bash
uv run mlx_vlm.server \
  --model mlx-community/gemma-4-e4b-it-mxfp8 \
  --host 0.0.0.0 \
  --port 7500
```

First run downloads ~5 GB. Subsequent runs spin up in seconds.

Verify it's healthy:

```bash
curl -s http://localhost:7500/v1/models
```

### 2. Start screen capture

```bash
cd sentinel
uv run python -m sentinel.capture
```

You'll see one log line per cycle: `[01:34:46] screen=1 #105 Code Editor (low) — Editing settings.py...`

Leave it running. Ctrl+C to stop cleanly.

### 3. Open the desktop UI

```bash
uv run python -m sentinel.ui
```

A native desktop window opens with:
- **Stats** — notes today, unique apps, first note time (live, refreshes every 10s)
- **Ask** — text box + suggestion chips, streamed answer
- **Today's briefing** — one click, streamed paragraph

---

## Inspecting the data

Browse the SQLite file with anything you like. A nice option:

```bash
uvx sqlite-web sentinel/sentinel.db
```

Open `http://localhost:8080`, browse the `notes` table.

---

## What it deliberately doesn't do

- ✗ Upload anything to a cloud
- ✗ Save screenshots to disk
- ✗ Run face recognition or person tracking
- ✗ Notify you in real time (out of scope for v1)
- ✗ Work on Windows / Linux (MLX is Apple Silicon only)



---

## Roadmap

Things that would obviously be next:

- Real-time nudges ("Sarah from Sequoia just emailed")
- Voice queries (talk instead of type)


---

Built solo by [Keshav Garg](https://www.linkedin.com/in/keshavcodes). If you'd like to discuss building privacy-first AI tools or have ideas for what to add, reach out on LinkedIn.
