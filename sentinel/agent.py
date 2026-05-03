"""Q&A agent over Sentinel notes.

Single tool: query_notes(sql) — agent writes its own read-only SELECT against the
notes table. Schema lives in the system prompt so the agent doesn't have to
discover it. Errors are returned as text so Agno's agent loop retries naturally.

Run:
    uv run python -m sentinel.agent "when did I last look at Slack?"
"""

import json
import re
import sqlite3
import sys
from datetime import datetime

from agno.agent import Agent
from agno.models.openai import OpenAILike
from agno.tools import tool

from sentinel.settings import settings


SYSTEM_PROMPT = """You answer questions about the user's screen activity by querying a notes table.

Each row is one screenshot the user's laptop took every {interval}s while they worked.

Schema:
```sql
CREATE TABLE notes (
    id       INTEGER PRIMARY KEY,
    ts       TEXT,         -- ISO 8601 local timestamp
    screen   INTEGER,      -- monitor index (1, 2, ...)
    app      TEXT,         -- app or website in focus, e.g. 'Slack', 'VS Code', 'Gmail'
    ocr      TEXT,         -- verbatim text on screen at that moment
    summary  TEXT,         -- 3-6 sentence interpretive description
    people   TEXT,         -- JSON array of names like '["Sarah","John"]' — use LIKE '%name%'
    urgency  TEXT          -- 'low' | 'medium' | 'high'
);
```

SQL tips for this schema:
- Use LIKE '%text%' for simple fuzzy matches on app, summary, ocr.
- REGEXP is also available — uses Python re.search semantics. Use it when LIKE is not enough,
  e.g. for digits, word boundaries, or alternation.
- For people, treat the column as text: WHERE people LIKE '%Sarah%'.
- For "today": WHERE date(ts) = date('now', 'localtime').
- For "last hour": WHERE ts >= datetime('now', '-1 hour', 'localtime').
- Order by ts DESC for recency.
- Always LIMIT 50 unless the question is a count or aggregate.
- The tool only accepts a single SELECT statement (no semicolons, no DDL/DML).

Examples:

Q: "When did I look at Slack today?"
SQL: SELECT ts, summary FROM notes WHERE app LIKE '%Slack%' AND date(ts) = date('now', 'localtime') ORDER BY ts LIMIT 20

Q: "What did Sarah say recently?"
SQL: SELECT ts, app, summary FROM notes WHERE people LIKE '%Sarah%' ORDER BY ts DESC LIMIT 20

Q: "How many high urgency notes today?"
SQL: SELECT COUNT(*) AS cnt FROM notes WHERE urgency = 'high' AND date(ts) = date('now', 'localtime')

Q: "What apps did I use most this week?"
SQL: SELECT app, COUNT(*) AS n FROM notes WHERE ts >= datetime('now', '-7 days') GROUP BY app ORDER BY n DESC LIMIT 10

Q: "Show me notes mentioning a dollar amount"
SQL: SELECT ts, app, summary FROM notes WHERE ocr REGEXP '\\$\\d+' ORDER BY ts DESC LIMIT 20

Q: "Find notes about Q2 or Q3 numbers"
SQL: SELECT ts, app, summary FROM notes WHERE summary REGEXP '\\bQ[23]\\b' ORDER BY ts DESC LIMIT 20

Current local time: {now}

After running a query, write a natural, concise answer for the user. Cite specific
timestamps and apps when relevant. If the result is empty, say so honestly."""


@tool
def query_notes(sql: str) -> str:
    """Execute a single read-only SELECT against the notes table.

    Args:
        sql: A single SELECT statement. No DDL, DML, or multiple statements.

    Returns:
        JSON-encoded list of result rows, or a string starting with 'ERROR:'.
    """
    sql = sql.strip().rstrip(";")
    if not sql.lower().startswith("select"):
        return "ERROR: Only SELECT queries allowed."
    if ";" in sql:
        return "ERROR: Only one statement allowed."

    try:
        with sqlite3.connect(f"file:{settings.db_path}?mode=ro", uri=True) as conn:
            conn.row_factory = sqlite3.Row
            conn.create_function(
                "REGEXP",
                2,
                lambda pattern, value: 1 if value is not None and re.search(pattern, value) else 0,
                deterministic=True,
            )
            rows = conn.execute(sql).fetchall()
            return json.dumps([dict(row) for row in rows], default=str)
    except (sqlite3.Error, re.error) as e:
        return f"SQL ERROR: {e}"


def make_agent() -> Agent:
    return Agent(
        model=OpenAILike(
            id=settings.GEMMA_MODEL,
            base_url=settings.GEMMA_BASE_URL,
            api_key="not-needed",
        ),
        tools=[query_notes],
        instructions=SYSTEM_PROMPT.format(
            interval=settings.CAPTURE_INTERVAL_SECONDS,
            now=datetime.now().isoformat(timespec="seconds"),
        ),
        markdown=True,
        debug_mode=True
    )


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: uv run python -m sentinel.agent "<question>"')
        sys.exit(1)

    question = " ".join(sys.argv[1:])
    agent = make_agent()
    response = agent.run(question)
    print(response.content)


if __name__ == "__main__":
    main()
