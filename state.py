"""Shared LangGraph state for the Phase 1 architecture."""

from __future__ import annotations

from typing import Annotated, Literal, TypedDict

from langgraph.graph.message import add_messages

AgentName = Literal[
    "supervisor",
    "cardiovascular",
    "sleep",
    "drug",
    "writing",
    "finish",
]


class MedicalAgentState(TypedDict, total=False):
    """State container passed between all graph nodes."""

    messages: Annotated[list, add_messages]
    next_agent: AgentName | None
    route_history: list[str]
    loop_count: int
    final_response: str | None
    execution_trace: list[dict]
    token_usage: dict[str, int]
    conversation_summary: str | None
    conversation_turn_count: int
