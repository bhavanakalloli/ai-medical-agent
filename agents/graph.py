"""LangGraph assembly for the Phase 1 medical multi-agent system."""

from __future__ import annotations

from functools import partial
from typing import Any

from langgraph.graph import END, START, StateGraph

from agents.model import DEFAULT_MODEL, build_gemini_model
from agents.nodes import (
    cardiovascular_node,
    drug_node,
    sleep_node,
    supervisor_node,
    writing_node,
)
from state import MedicalAgentState


def _route_from_supervisor(state: MedicalAgentState) -> str:
    next_agent = state.get("next_agent", "finish")
    if next_agent in {"cardiovascular", "sleep", "drug", "writing", "finish"}:
        return next_agent
    return "finish"


def build_graph(llm: Any | None = None, model_name: str = DEFAULT_MODEL):
    """Build and compile the LangGraph graph for Phase 1."""
    model = llm or build_gemini_model(model_name=model_name)

    graph = StateGraph(MedicalAgentState)

    graph.add_node("supervisor", partial(supervisor_node, llm=model))
    graph.add_node("cardiovascular", partial(cardiovascular_node, llm=model))
    graph.add_node("sleep", partial(sleep_node, llm=model))
    graph.add_node("drug", partial(drug_node, llm=model))
    graph.add_node("writing", partial(writing_node, llm=model))

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        _route_from_supervisor,
        {
            "cardiovascular": "cardiovascular",
            "sleep": "sleep",
            "drug": "drug",
            "writing": "writing",
            "finish": END,
        },
    )

    graph.add_edge("cardiovascular", "supervisor")
    graph.add_edge("sleep", "supervisor")
    graph.add_edge("drug", "supervisor")
    graph.add_edge("writing", "supervisor")

    return graph.compile()
