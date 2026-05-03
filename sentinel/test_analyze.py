"""Run sentinel.analyze on a single image file and print the Note.

Usage:
    uv run python scripts/test_analyze.py <image_path>

Example:
    uv run python scripts/test_analyze.py /Users/keshav/Pictures/Screenshots/test.jpg
"""

import sys
import time
from pathlib import Path

from analyze import analyze_screenshot


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    path = Path(sys.argv[1])
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    img_bytes = path.read_bytes()
    print(f"Image: {path.name} ({len(img_bytes) / 1024:.1f} KB)")
    print("Calling Gemma 4...")

    started = time.perf_counter()
    note = analyze_screenshot(img_bytes)
    elapsed = time.perf_counter() - started

    print(f"Done in {elapsed:.1f}s")
    print()
    print("=== Note ===")
    print(note.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
