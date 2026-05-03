import time
from io import BytesIO

import mss
from PIL import Image

from sentinel.analyze import analyze_screenshot
from sentinel.settings import settings
from sentinel.storage import init_db, insert_note


def _grab_monitor(sct: mss.mss, monitor: dict) -> bytes:
    """Grab one monitor and return PNG bytes (in memory only)."""
    raw = sct.grab(monitor)
    img = Image.frombytes("RGB", raw.size, raw.rgb)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def run_loop() -> None:
    init_db()
    interval = settings.CAPTURE_INTERVAL_SECONDS

    with mss.mss() as sct:
        # monitors[0] is the virtual "all monitors" rect; [1:] are individual displays.
        screens = sct.monitors[1:]
        print(
            f"Sentinel started. Detected {len(screens)} screen(s). "
            f"Capturing every {interval}s. Ctrl+C to stop.\n"
        )

        cycle = 0
        while True:
            cycle += 1
            cycle_started = time.perf_counter()
            try:
                print(f"--- cycle {cycle} ---")
                for idx, monitor in enumerate(screens, start=1):
                    print(f"\n[screen {idx}] grabbing + analysing...")
                    img_bytes = _grab_monitor(sct, monitor)
                    note = analyze_screenshot(img_bytes)
                    note.screen = idx
                    note_id = insert_note(note)
                    print(
                        f"[{note.ts:%H:%M:%S}] screen={idx} #{note_id} "
                        f"{note.app} ({note.urgency}) — {note.summary[:90]}..."
                    )
                elapsed = time.perf_counter() - cycle_started
                print(f"cycle {cycle} done in {elapsed:.1f}s")
            except KeyboardInterrupt:
                print("\nSentinel stopped.")
                return
            except Exception as e:
                elapsed = time.perf_counter() - cycle_started
                print(f"[!] cycle {cycle} failed after {elapsed:.1f}s: {e}")

            # sleep relative to wall clock so a slow cycle doesn't compound
            time.sleep(max(0, interval - (time.perf_counter() - cycle_started)))


if __name__ == "__main__":
    try:
        run_loop()
    except KeyboardInterrupt:
        print("\nSentinel stopped.")
