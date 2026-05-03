"""Sentinel desktop UI — header, ask box, digest button, live notes feed.

Run:
    uv run python -m sentinel.ui
"""

import asyncio
from datetime import datetime, timedelta

from nicegui import ui

from sentinel.agent import make_agent
from sentinel.digest import stream_digest
from sentinel.storage import init_db, recent_notes  # recent_notes still used for status chip
from sentinel.streaming import stream_agent_response


_agent = None


def _get_agent():
    """Lazy — building the agent triggers an HTTP probe to the MLX server."""
    global _agent
    if _agent is None:
        _agent = make_agent()
    return _agent


def _capture_status_text() -> tuple[str, str]:
    """Heuristic: if the most recent note is younger than 2 capture intervals, we're live."""
    from sentinel.settings import settings

    notes = recent_notes(limit=1)
    if not notes:
        return "Idle", "grey"
    age = datetime.now() - notes[0].ts
    if age <= timedelta(seconds=settings.CAPTURE_INTERVAL_SECONDS * 2):
        return "Capturing", "green"
    return f"Last note {int(age.total_seconds() // 60)}m ago", "orange"


def build_ui() -> None:
    init_db()

    ui.colors(primary="#5D5CDE")

    with ui.column().classes("w-full max-w-3xl mx-auto p-6 gap-6"):
        # ── Header ────────────────────────────────────────────
        with ui.row().classes("items-center w-full"):
            ui.label("Sentinel").classes("text-3xl font-bold")
            ui.space()
            status_chip = ui.chip("Idle", color="grey", text_color="white")

        def refresh_status():
            text, color = _capture_status_text()
            status_chip.text = text
            status_chip.props(f"color={color}")

        ui.timer(5.0, refresh_status, immediate=True)

        # ── Ask ───────────────────────────────────────────────
        with ui.card().classes("w-full"):
            ui.label("Ask anything").classes("text-lg font-semibold")
            with ui.row().classes("w-full items-center gap-2"):
                question_input = (
                    ui.input(placeholder="what was I doing at 1 AM?")
                    .classes("flex-grow")
                    .props("outlined dense")
                )
                ask_button = ui.button("Ask", icon="search")
            answer = ui.markdown("").classes("mt-3")

            async def on_ask():
                q = (question_input.value or "").strip()
                if not q:
                    return
                ask_button.disable()
                buffer: list[str] = []
                answer.set_content("_Thinking..._")
                try:
                    async for ev in stream_agent_response(q, _get_agent()):
                        kind = ev.get("event")
                        if kind == "message.delta":
                            buffer.append(ev["content"])
                            answer.set_content("".join(buffer))
                        elif kind == "tool.started":
                            buffer.append(f"\n\n_calling `{ev['tool_name']}` ..._\n")
                            answer.set_content("".join(buffer))
                        elif kind == "tool.completed":
                            buffer.append("_(tool done)_\n\n")
                            answer.set_content("".join(buffer))
                        elif kind == "message.completed":
                            # Deltas already built the full text; only fall back if empty.
                            if not buffer and ev.get("content"):
                                answer.set_content(ev["content"])
                        elif kind == "error":
                            answer.set_content(f"**Error:** `{ev.get('error')}`")
                            break
                except Exception as e:
                    answer.set_content(f"**Error:** `{e}`")
                finally:
                    ask_button.enable()

            ask_button.on_click(on_ask)
            question_input.on("keydown.enter", on_ask)

        # ── Digest ────────────────────────────────────────────
        with ui.card().classes("w-full"):
            with ui.row().classes("items-center w-full"):
                ui.label("Today's briefing").classes("text-lg font-semibold")
                ui.space()
                digest_button = ui.button("Generate", icon="article")
            digest_text = ui.markdown("").classes("mt-3")

            async def on_digest():
                digest_button.disable()
                buffer: list[str] = []
                digest_text.set_content("_Reading today's notes..._")
                try:
                    async for chunk in stream_digest():
                        buffer.append(chunk)
                        digest_text.set_content("".join(buffer))
                except Exception as e:
                    digest_text.set_content(f"**Error:** `{e}`")
                finally:
                    digest_button.enable()

            digest_button.on_click(on_digest)


build_ui()


def main() -> None:
    ui.run(
        title="Sentinel",
        native=True,
        window_size=(1100, 800),
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
