"""Conversation memory management for long-running chat sessions."""

from __future__ import annotations

from typing import Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from tools.summarise_convo import summarise_convo

SUMMARY_EVERY_TURNS = 20
RECENT_MESSAGES_WINDOW = 8


def records_to_messages(records: list[dict[str, str]]) -> list[BaseMessage]:
    """Convert persisted role/content records into LangChain message objects."""
    converted: list[BaseMessage] = []
    for record in records:
        role = record.get("role", "")
        content = str(record.get("content", ""))
        if role == "assistant":
            converted.append(AIMessage(content=content))
        else:
            converted.append(HumanMessage(content=content))
    return converted


def build_context_from_db_fetch(
    *,
    fetched_records: list[dict[str, str]],
    existing_summary: str | None,
    recent_window: int = RECENT_MESSAGES_WINDOW,
) -> dict:
    """Create memory context from DB with mandatory summary layer to reduce bloat."""
    db_summary = summarise_convo(fetched_records, existing_summary)
    recent_records = fetched_records[-recent_window:] if recent_window > 0 else []
    return {
        "conversation_summary": db_summary if db_summary else existing_summary,
        "history_messages": records_to_messages(recent_records),
    }


def default_summary_function(messages: list[BaseMessage], previous_summary: str | None = None) -> str:
    """Build a compact textual summary from historical messages."""
    chunks: list[str] = []
    if previous_summary:
        chunks.append(f"Previous summary: {previous_summary}")

    for message in messages[-24:]:
        role = "user" if isinstance(message, HumanMessage) else "assistant"
        content = message.content if isinstance(message.content, str) else str(message.content)
        content = " ".join(content.split())
        if len(content) > 180:
            content = content[:177] + "..."
        chunks.append(f"{role}: {content}")

    return "\n".join(chunks)


def build_messages_for_turn(
    query: str,
    history_messages: list[BaseMessage] | None,
    conversation_summary: str | None,
    *,
    recent_window: int = RECENT_MESSAGES_WINDOW,
) -> list[BaseMessage]:
    """Prepare graph input messages using summary + recent raw context + latest query."""
    history = history_messages or []
    windowed_history = history[-recent_window:] if recent_window > 0 else []

    payload: list[BaseMessage] = []
    if conversation_summary:
        payload.append(
            SystemMessage(
                content=(
                    "Conversation summary from earlier turns. Use this as prior context:\n"
                    f"{conversation_summary}"
                )
            )
        )

    payload.extend(windowed_history)
    payload.append(HumanMessage(content=query))
    return payload


def update_conversation_memory(
    *,
    history_messages: list[BaseMessage] | None,
    conversation_summary: str | None,
    turn_count: int,
    query: str,
    final_response: str,
    summary_every: int = SUMMARY_EVERY_TURNS,
    recent_window: int = RECENT_MESSAGES_WINDOW,
    summary_function: Callable[[list[BaseMessage], str | None], str] = default_summary_function,
) -> dict:
    """Update memory state and summarize every N turns while trimming raw message history."""
    history = list(history_messages or [])
    history.append(HumanMessage(content=query))
    history.append(AIMessage(content=final_response))

    updated_turn_count = turn_count + 1
    summary_updated = False
    updated_summary = conversation_summary

    if summary_every > 0 and updated_turn_count % summary_every == 0:
        updated_summary = summary_function(history, conversation_summary)
        summary_updated = True
        if recent_window > 0:
            history = history[-recent_window:]
        else:
            history = []

    return {
        "history_messages": history,
        "conversation_summary": updated_summary,
        "turn_count": updated_turn_count,
        "summary_updated": summary_updated,
    }
