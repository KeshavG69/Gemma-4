"""Sentinel desktop UI — polished demo layout.

Run:
    uv run python -m sentinel.ui
"""

import atexit
import signal
import subprocess
import sys
from datetime import date, datetime, timedelta

from nicegui import ui

from sentinel.agent import make_agent
from sentinel.digest import stream_digest
from sentinel.settings import PROJECT_ROOT
from sentinel.storage import init_db, notes_on_date, recent_notes
from sentinel.streaming import stream_agent_response


SUGGESTED_QUESTIONS = [
    "What apps did I use today?",
    "What was I doing at 1 AM?",
    "Any high urgency moments today?",
    "Summarise the last hour",
]


_agent = None
_capture_proc: subprocess.Popen | None = None


def _get_agent():
    global _agent
    if _agent is None:
        _agent = make_agent()
    return _agent


def is_capturing() -> bool:
    """True if a capture subprocess is alive."""
    return _capture_proc is not None and _capture_proc.poll() is None


def start_capture() -> None:
    """Spawn `python -m sentinel.capture` as a subprocess from the project root."""
    global _capture_proc
    if is_capturing():
        return
    _capture_proc = subprocess.Popen(
        [sys.executable, "-m", "sentinel.capture"],
        cwd=PROJECT_ROOT,
    )


def stop_capture() -> None:
    """Send SIGINT so capture's KeyboardInterrupt handler exits cleanly."""
    global _capture_proc
    if _capture_proc is None:
        return
    if _capture_proc.poll() is None:
        try:
            _capture_proc.send_signal(signal.SIGINT)
            _capture_proc.wait(timeout=5)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            _capture_proc.kill()
    _capture_proc = None


# Make sure capture doesn't outlive the UI on a clean exit
atexit.register(stop_capture)


def _capture_status() -> tuple[str, str, str]:
    """Returns (label, color, icon) for the status chip."""
    from sentinel.settings import settings

    notes = recent_notes(limit=1)
    if not notes:
        return ("Capturing", "green", "fiber_manual_record") if is_capturing() else ("Idle", "grey-7", "circle")
    age = datetime.now() - notes[0].ts
    if age <= timedelta(seconds=settings.CAPTURE_INTERVAL_SECONDS * 2):
        return "Capturing", "green", "fiber_manual_record"
    minutes = int(age.total_seconds() // 60)
    if is_capturing():
        return f"Capturing (last {minutes}m ago)", "green", "fiber_manual_record"
    return f"Last note {minutes}m ago", "orange", "schedule"


def build_ui() -> None:
    init_db()

    # Theme
    ui.colors(
        primary="#5D5CDE",
        secondary="#10B981",
        accent="#F59E0B",
        positive="#10B981",
        negative="#EF4444",
    )

    # Page background
    ui.query("body").style("background-color: #F8FAFC")

    with ui.column().classes("w-full max-w-4xl mx-auto p-8 gap-6"):

        # ── HEADER ─────────────────────────────────────────────────
        with ui.row().classes("w-full items-start justify-between"):
            with ui.column().classes("gap-1"):
                with ui.row().classes("items-center gap-3"):
                    ui.icon("shield", size="2.25rem").classes("text-primary")
                    ui.label("Sentinel").classes("text-4xl font-bold tracking-tight")
                ui.label("Your day, remembered locally.").classes(
                    "text-lg text-gray-600"
                )
                with ui.row().classes("items-center gap-1.5 mt-1"):
                    ui.icon("lock", size="0.875rem").classes("text-emerald-600")
                    ui.label("100% on this laptop · Powered by Gemma 4").classes(
                        "text-xs text-gray-500"
                    )
            with ui.column().classes("items-end gap-2"):
                status_chip = (
                    ui.chip("Idle", icon="circle", color="grey-7")
                    .props("text-color=white outline")
                )
                capture_button = ui.button("Start Capture", icon="play_arrow").props(
                    "rounded color=positive"
                )

        def toggle_capture():
            if is_capturing():
                stop_capture()
            else:
                start_capture()
            refresh_status()

        capture_button.on_click(toggle_capture)

        def refresh_status():
            label, color, icon = _capture_status()
            status_chip.text = label
            status_chip.props(f"color={color} icon={icon}")
            if is_capturing():
                capture_button.text = "Stop Capture"
                capture_button.props("icon=stop color=negative")
            else:
                capture_button.text = "Start Capture"
                capture_button.props("icon=play_arrow color=positive")

        ui.timer(5.0, refresh_status, immediate=True)

        # ── STATS STRIP ────────────────────────────────────────────
        with ui.row().classes("w-full gap-4"):
            stat_cards: dict[str, ui.label] = {}
            for key, label in [
                ("notes", "Notes today"),
                ("apps", "Unique apps"),
                ("first", "First note today"),
            ]:
                with ui.card().classes(
                    "flex-1 p-4 shadow-sm border border-gray-100"
                ):
                    ui.label(label).classes(
                        "text-xs text-gray-500 uppercase tracking-wider"
                    )
                    stat_cards[key] = ui.label("—").classes(
                        "text-3xl font-bold mt-1 text-gray-900"
                    )

        def refresh_stats():
            today = notes_on_date(date.today())
            stat_cards["notes"].text = str(len(today))
            stat_cards["apps"].text = str(len({n.app for n in today}))
            stat_cards["first"].text = today[0].ts.strftime("%H:%M") if today else "—"

        ui.timer(10.0, refresh_stats, immediate=True)

        # ── ASK CARD ───────────────────────────────────────────────
        with ui.card().classes("w-full p-6 shadow-sm border border-gray-100"):
            with ui.row().classes("items-center gap-2 mb-3"):
                ui.icon("question_answer", size="1.5rem").classes("text-primary")
                ui.label("Ask anything").classes("text-xl font-semibold text-gray-900")

            with ui.row().classes("w-full items-center gap-2"):
                question_input = (
                    ui.input(placeholder="What was I doing at 1 AM last night?")
                    .classes("flex-grow")
                    .props("outlined dense rounded")
                )
                ask_button = (
                    ui.button("Ask", icon="send")
                    .props("rounded color=primary")
                )

            answer = ui.markdown("").classes(
                "mt-4 p-4 bg-slate-50 rounded-lg min-h-[60px] prose prose-sm max-w-none"
            )

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
                            buffer.append(
                                f"\n\n🔎 _Querying: `{ev['tool_name']}`_\n\n"
                            )
                            answer.set_content("".join(buffer))
                        elif kind == "tool.completed":
                            if ev.get("error"):
                                buffer.append(f"⚠ _Tool error_\n\n")
                            else:
                                buffer.append("✓ _Got results_\n\n")
                            answer.set_content("".join(buffer))
                        elif kind == "message.completed":
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

            # Suggestion chips
            with ui.row().classes("items-center gap-2 mt-4 flex-wrap"):
                ui.label("Try:").classes("text-sm text-gray-500")
                for q in SUGGESTED_QUESTIONS:
                    def make_handler(question_text=q):
                        async def handler():
                            question_input.value = question_text
                            await on_ask()
                        return handler
                    chip = (
                        ui.chip(q, color="primary")
                        .props("outline clickable")
                        .classes("text-sm")
                    )
                    chip.on("click", make_handler())

        # ── DIGEST CARD ────────────────────────────────────────────
        with ui.card().classes("w-full p-6 shadow-sm border border-gray-100"):
            with ui.row().classes("items-center w-full mb-3"):
                ui.icon("auto_awesome", size="1.5rem").classes("text-amber-500")
                ui.label("Today's briefing").classes(
                    "text-xl font-semibold text-gray-900"
                )
                ui.space()
                digest_button = (
                    ui.button("Generate", icon="auto_fix_high")
                    .props("rounded color=primary outline")
                )
            digest_text = ui.markdown(
                "_Click **Generate** to summarise everything Sentinel saw today._"
            ).classes(
                "mt-2 p-4 bg-slate-50 rounded-lg min-h-[60px] prose prose-sm max-w-none text-gray-700"
            )

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

        # ── FOOTER ─────────────────────────────────────────────────
        with ui.row().classes("w-full justify-center mt-2"):
            ui.label(
                "Built with Gemma 4 · Agno · NiceGUI · MLX. No data leaves this machine."
            ).classes("text-xs text-gray-400")


build_ui()


def main() -> None:
    ui.run(
        title="Sentinel",
        native=True,
        window_size=(1100, 900),
        reload=False,
    )


if __name__ in {"__main__", "__mp_main__"}:
    main()
