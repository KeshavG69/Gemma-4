from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class NoteAnalysis(BaseModel):
    """Visual content extracted by the VLM from a single screenshot."""

    app: str = Field(description="Active app or website visible on screen")
    ocr: str = Field(description="Verbatim transcription of all readable text on screen, top-to-bottom, left-to-right, preserving rough grouping with newlines")
    summary: str = Field(description="A 3-6 sentence interpretive overview of what the user is doing and seeing")
    people: list[str] = Field(default_factory=list, description="Names of people visible or mentioned")
    urgency: Literal["low", "medium", "high"] = "low"


class Note(NoteAnalysis):
    """A NoteAnalysis stamped with capture metadata (when, which screen)."""

    ts: datetime
    screen: int = 1
