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


def recent_notes(limit: int = 10) -> list[Note]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT ts, screen, app, ocr, summary, people, urgency FROM notes ORDER BY ts DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [
        Note(
            ts=datetime.fromisoformat(r["ts"]),
            screen=r["screen"],
            app=r["app"],
            ocr=r["ocr"],
            summary=r["summary"],
            people=json.loads(r["people"]),
            urgency=r["urgency"],
        )
        for r in rows
    ]
