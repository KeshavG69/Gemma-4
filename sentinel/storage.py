import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime

from sentinel.models import Note
from sentinel.settings import settings


SCHEMA = """
CREATE TABLE IF NOT EXISTS notes (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    ts       TEXT    NOT NULL,
    screen   INTEGER NOT NULL DEFAULT 1,
    app      TEXT    NOT NULL,
    ocr      TEXT    NOT NULL DEFAULT '',
    summary  TEXT    NOT NULL,
    people   TEXT    NOT NULL DEFAULT '[]',
    urgency  TEXT    NOT NULL DEFAULT 'low'
);

CREATE INDEX IF NOT EXISTS idx_notes_ts ON notes(ts);
"""


@contextmanager
def _connect():
    conn = sqlite3.connect(settings.db_path)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db() -> None:
    with _connect() as conn:
        conn.executescript(SCHEMA)


def insert_note(note: Note) -> int:
    with _connect() as conn:
        cur = conn.execute(
            "INSERT INTO notes (ts, screen, app, ocr, summary, people, urgency) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                note.ts.isoformat(),
                note.screen,
                note.app,
                note.ocr,
                note.summary,
                json.dumps(note.people),
                note.urgency,
            ),
        )
        return cur.lastrowid


def _row_to_note(row) -> Note:
    return Note(
        ts=datetime.fromisoformat(row["ts"]),
        screen=row["screen"],
        app=row["app"],
        ocr=row["ocr"],
        summary=row["summary"],
        people=json.loads(row["people"]),
        urgency=row["urgency"],
    )


def recent_notes(limit: int = 10) -> list[Note]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, screen, app, ocr, summary, people, urgency FROM notes ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [_row_to_note(r) for r in rows]


def notes_on_date(d) -> list[Note]:
    """All notes for a given local date, oldest first. No limit."""
    iso = d.isoformat() if hasattr(d, "isoformat") else str(d)
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, screen, app, ocr, summary, people, urgency FROM notes "
            "WHERE date(ts) = ? ORDER BY ts ASC",
            (iso,),
        ).fetchall()
    return [_row_to_note(r) for r in rows]
