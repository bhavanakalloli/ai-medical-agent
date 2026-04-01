"""FastAPI application for the medical supervisor multi-agent system."""

from __future__ import annotations

import time
import uuid
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agents.context_memory import (
    build_context_from_db_fetch,
    build_messages_for_turn,
    update_conversation_memory,
)
from agents.graph import build_graph
from agents.model import DEFAULT_MODEL
from database.database import SQLiteMemoryStore
from utils.telemetry import (
    annotate_trace_with_token_estimates,
    estimate_cost_usd,
    estimate_usage_from_trace,
)

app = FastAPI(title="Medical Supervisor API", version="1.0.0")
store = SQLiteMemoryStore(db_path="database/medical_agent.db")
_compiled_graph = None


class ExecuteRequest(BaseModel):
    query: str = Field(min_length=1)
    session_id: str | None = None


class ExecuteResponse(BaseModel):
    session_id: str
    final_report: str
    route_history: list[str]
    execution_trace: list[dict[str, Any]]
    token_usage: dict[str, Any]
    retrieval: dict[str, Any]
    conversation_context: dict[str, Any]
    time_taken_ms: int


def _graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = build_graph()
    return _compiled_graph

def run_once(
    query: str,
    *,
    session_id: str,
    store: SQLiteMemoryStore,
    context: dict | None = None,
) -> dict:
    """Execute one graph run for a user query with persisted memory context."""

    if context is None:
        fetched = store.load_session_context(session_id)
        db_context = build_context_from_db_fetch(
            fetched_records=fetched.get("messages", []),
            existing_summary=fetched.get("conversation_summary"),
        )
        memory = {
            "history_messages": db_context.get("history_messages", []),
            "conversation_summary": db_context.get("conversation_summary"),
            "turn_count": fetched.get("turn_count", 0),
        }
    else:
        memory = context

    prepared_messages = build_messages_for_turn(
        query,
        memory.get("history_messages", []),
        memory.get("conversation_summary"),
    )

    initial_state = {
        "messages": prepared_messages,
        "next_agent": None,
        "route_history": [],
        "loop_count": 0,
        "final_response": None,
        "execution_trace": [],
        "token_usage": {},
        "conversation_summary": memory.get("conversation_summary"),
        "conversation_turn_count": memory.get("turn_count", 0),
    }

    result = _graph().invoke(initial_state)
    final_response = result.get("final_response") or "No final response produced."
    updated_memory = update_conversation_memory(
        history_messages=memory.get("history_messages", []),
        conversation_summary=memory.get("conversation_summary"),
        turn_count=memory.get("turn_count", 0),
        query=query,
        final_response=final_response,
    )

    return {
        "final_response": final_response,
        "route_history": result.get("route_history", []),
        "execution_trace": result.get("execution_trace", []),
        "memory": updated_memory,
    }


@app.get("/v1/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/v1/session/{session_id}")
def get_session(session_id: str) -> dict[str, Any]:
    return store.load_session_context(session_id)


@app.get("/v1/sessions")
def list_sessions(limit: int = 200) -> dict[str, Any]:
    return {"sessions": store.list_sessions(limit=limit)}


@app.post("/v1/execute", response_model=ExecuteResponse)
def execute(req: ExecuteRequest) -> ExecuteResponse:
    started = time.perf_counter()
    session_id = req.session_id or str(uuid.uuid4())
    query = req.query.strip()
    if not query:
        raise HTTPException(status_code=400, detail="query cannot be empty")

    # Session-based retrieval first: if a close prior query exists, reuse answer.
    retrieval_match = store.find_similar_query_response(session_id=session_id, query=query)
    if retrieval_match:
        final_report = retrieval_match["matched_response"]
        execution_trace = [
            {
                "agent": "session_retrieval",
                "input": query,
                "output": final_report,
                "note": f"matched_query={retrieval_match['matched_query']}",
            }
        ]
        execution_trace, tokens_by_agent = annotate_trace_with_token_estimates(execution_trace)
        token_usage = estimate_usage_from_trace(
            query=query,
            execution_trace=execution_trace,
            final_response=final_report,
        )
        token_usage["estimated_cost_usd"] = estimate_cost_usd(token_usage, DEFAULT_MODEL)
        token_usage["model"] = DEFAULT_MODEL
        token_usage["is_estimated"] = True
        token_usage["by_agent"] = tokens_by_agent

        loaded_context = store.load_session_context(session_id)
        db_context = build_context_from_db_fetch(
            fetched_records=loaded_context.get("messages", []),
            existing_summary=loaded_context.get("conversation_summary"),
        )
        memory = {
            "history_messages": db_context.get("history_messages", []),
            "conversation_summary": db_context.get("conversation_summary"),
            "turn_count": loaded_context.get("turn_count", 0),
            "summary_updated": False,
        }
        memory = update_conversation_memory(
            history_messages=memory.get("history_messages", []),
            conversation_summary=memory.get("conversation_summary"),
            turn_count=memory.get("turn_count", 0),
            query=query,
            final_response=final_report,
        )
        store.save_turn(
            session_id=session_id,
            query=query,
            response=final_report,
            turn_count=memory.get("turn_count", 0),
            conversation_summary=memory.get("conversation_summary"),
            execution_trace=execution_trace,
        )

        elapsed_ms = int((time.perf_counter() - started) * 1000)
        return ExecuteResponse(
            session_id=session_id,
            final_report=final_report,
            route_history=["retrieval_hit", "finish"],
            execution_trace=execution_trace,
            token_usage=token_usage,
            retrieval={"hit": True, **retrieval_match},
            conversation_context={
                "turn_count": memory.get("turn_count", 0),
                "conversation_summary": memory.get("conversation_summary"),
                "summary_updated": memory.get("summary_updated", False),
            },
            time_taken_ms=elapsed_ms,
        )

    result = run_once(query, session_id=session_id, store=store, context=None)
    memory = result["memory"]
    execution_trace, tokens_by_agent = annotate_trace_with_token_estimates(result.get("execution_trace", []))

    store.save_turn(
        session_id=session_id,
        query=query,
        response=result["final_response"],
        turn_count=memory.get("turn_count", 0),
        conversation_summary=memory.get("conversation_summary"),
        execution_trace=execution_trace,
    )

    token_usage = estimate_usage_from_trace(
        query=query,
        execution_trace=execution_trace,
        final_response=result["final_response"],
    )
    token_usage["estimated_cost_usd"] = estimate_cost_usd(token_usage, DEFAULT_MODEL)
    token_usage["model"] = DEFAULT_MODEL
    token_usage["is_estimated"] = True
    token_usage["by_agent"] = tokens_by_agent

    elapsed_ms = int((time.perf_counter() - started) * 1000)
    return ExecuteResponse(
        session_id=session_id,
        final_report=result["final_response"],
        route_history=result.get("route_history", []),
        execution_trace=execution_trace,
        token_usage=token_usage,
        retrieval={"hit": False},
        conversation_context={
            "turn_count": memory.get("turn_count", 0),
            "conversation_summary": memory.get("conversation_summary"),
            "summary_updated": memory.get("summary_updated", False),
        },
        time_taken_ms=elapsed_ms,
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
