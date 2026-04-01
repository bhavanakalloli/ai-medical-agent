"""Tool layer for summarizing fetched conversation history."""

from __future__ import annotations

from typing import Iterable


def summarise_convo(
    messages: Iterable[dict[str, str]],
    previous_summary: str | None = None,
    *,
    max_items: int = 30,
    max_chars_per_item: int = 220,
) -> str:
    """Summarize role/content message records into compact context text."""
    chunks: list[str] = []
    if previous_summary:
        chunks.append(f"Previous summary: {previous_summary}")

    window = list(messages)[-max_items:]
    for item in window:
        role = item.get("role", "unknown")
        content = " ".join(str(item.get("content", "")).split())
        if len(content) > max_chars_per_item:
            content = content[: max_chars_per_item - 3] + "..."
        chunks.append(f"{role}: {content}")

    return "\n".join(chunks)
