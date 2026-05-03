"""End-of-day digest over today's notes — one Gemma call, streamed."""

import asyncio
from datetime import date
from typing import AsyncIterator

from openai import AsyncOpenAI

from sentinel.settings import settings
from sentinel.storage import notes_on_date


DIGEST_PROMPT = """You are summarising someone's day from a chronological list of structured notes.
Each note describes what was on their laptop screen at one moment in time.

Write a natural, friendly briefing in 2-3 short paragraphs. Cover:
- What they spent the most time on (apps, topics)
- Notable events: people they engaged with, urgent moments, decisions
- Any patterns or shifts during the day (focused stretch, fragmented, late night)

Be specific — cite times (HH:MM) and apps. If their day was unfocused, say so honestly.
Don't editorialise. Don't preach. Just describe what actually happened."""


_client = AsyncOpenAI(base_url=settings.GEMMA_BASE_URL, api_key="not-needed")


def _today_notes_text() -> str:
    notes = notes_on_date(date.today())
    return "\n".join(
        f"[{n.ts.strftime('%H:%M')}] {n.app} ({n.urgency}): {n.summary}"
        for n in notes
    )


async def stream_digest() -> AsyncIterator[str]:
    """Stream today's digest as token chunks. Yields '' if no notes today."""
    notes_text = _today_notes_text()
    if not notes_text:
        yield "No activity recorded today yet."
        return

    stream = await _client.chat.completions.create(
        model=settings.GEMMA_MODEL,
        messages=[
            {"role": "system", "content": DIGEST_PROMPT},
            {"role": "user", "content": f"Today's notes:\n\n{notes_text}\n\nWrite the briefing."},
        ],
        max_tokens=1500,
        temperature=0.5,
        stream=True,
    )

    async for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta.content
        if delta:
            yield delta


async def _collect_digest() -> str:
    parts: list[str] = []
    async for chunk in stream_digest():
        parts.append(chunk)
    return "".join(parts)


def generate_digest() -> str:
    """Sync wrapper around the streamed digest (handy for tests / non-async callers)."""
    return asyncio.run(_collect_digest())


async def _cli_stream() -> None:
    async for chunk in stream_digest():
        print(chunk, end="", flush=True)
    print()


if __name__ == "__main__":
    asyncio.run(_cli_stream())
