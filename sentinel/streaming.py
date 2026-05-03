"""Stream Agno agent events as simple dict events.

Same event-stream pattern as kroolo's enterprise-fastapi/services/ai/agent_streaming.py,
trimmed to only what Sentinel needs (single agent, single tool, no sub-agents,
no cost tracking, no DB logging).

Event shape:
    {"event": "message.delta",     "content": "<token chunk>"}
    {"event": "tool.started",      "tool_name": "...", "tool_args": {...}}
    {"event": "tool.completed",    "tool_name": "...", "result": "..."}
    {"event": "message.completed", "content": "<final text>", "metrics": {...}}
    {"event": "error",             "error": "..."}
"""

import logging
from typing import Any, AsyncGenerator, Dict

from agno.agent import Agent
from agno.run.agent import RunEvent, RunOutput, RunOutputEvent


logger = logging.getLogger(__name__)


def _extract_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if hasattr(content, "content"):
        return str(content.content or "")
    return str(content)


async def stream_agent_response(
    query: str,
    agent: Agent,
) -> AsyncGenerator[Dict[str, Any], None]:
    """Yield Sentinel-shaped events as the agent runs."""
    if not query or not query.strip():
        yield {"event": "error", "error": "Empty query"}
        return

    try:
        async for chunk in agent.arun(query, stream=True, stream_events=True):
            if isinstance(chunk, RunOutputEvent):
                payload: Dict[str, Any] = chunk.to_dict()
            elif isinstance(chunk, RunOutput):
                payload = chunk.to_dict()
                payload.setdefault("event", RunEvent.run_completed.value)
            else:
                payload = {
                    "event": getattr(chunk, "event", RunEvent.run_content.value),
                    "content": str(chunk),
                }

            event = payload.get("event", "")

            if event in {
                RunEvent.run_content.value,
                RunEvent.run_intermediate_content.value,
            }:
                text = _extract_text(payload.get("content"))
                if text:
                    yield {"event": "message.delta", "content": text}

            elif event == RunEvent.tool_call_started.value:
                tool = payload.get("tool") or {}
                yield {
                    "event": "tool.started",
                    "tool_name": tool.get("tool_name", ""),
                    "tool_args": tool.get("tool_args"),
                }

            elif event == RunEvent.tool_call_completed.value:
                tool = payload.get("tool") or {}
                yield {
                    "event": "tool.completed",
                    "tool_name": tool.get("tool_name", ""),
                    "result": tool.get("result"),
                    "error": tool.get("tool_call_error"),
                }

            elif event == RunEvent.run_completed.value:
                yield {
                    "event": "message.completed",
                    "content": _extract_text(payload.get("content")),
                    "metrics": payload.get("metrics"),
                }

            elif event == RunEvent.run_error.value:
                yield {
                    "event": "error",
                    "error": payload.get("content") or payload.get("message") or "agent error",
                }

    except Exception as exc:
        logger.exception("agent stream error")
        yield {"event": "error", "error": str(exc)}
