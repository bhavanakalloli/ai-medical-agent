"""Token and cost estimation utilities for supervisor execution telemetry."""

from __future__ import annotations

from typing import Any

DEFAULT_PRICING_USD_PER_1K = {
    "gemini-3.1-flash-lite-preview": {"input": 0.00035, "output": 0.00053},
    "gemini-1.5-flash": {"input": 0.00035, "output": 0.00053},
}


def estimate_tokens_from_text(text: str) -> int:
    """Estimate token count from text length with a simple heuristic."""
    if not text:
        return 0
    return max(1, len(text) // 4)


def estimate_usage_from_trace(
    *,
    query: str,
    execution_trace: list[dict[str, Any]],
    final_response: str,
) -> dict[str, int]:
    """Estimate prompt/completion/total tokens for one supervisor execution loop."""
    prompt_tokens = estimate_tokens_from_text(query)
    completion_tokens = estimate_tokens_from_text(final_response)

    for event in execution_trace:
        prompt_tokens += estimate_tokens_from_text(str(event.get("input", "")))
        prompt_tokens += estimate_tokens_from_text(str(event.get("note", "")))
        completion_tokens += estimate_tokens_from_text(str(event.get("output", "")))

    return {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


def annotate_trace_with_token_estimates(
    execution_trace: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, int]]]:
    """Annotate each trace event with estimated tokens and aggregate by agent."""
    annotated: list[dict[str, Any]] = []
    by_agent: dict[str, dict[str, int]] = {}

    for event in execution_trace:
        agent = str(event.get("agent", "unknown"))
        prompt_tokens = estimate_tokens_from_text(str(event.get("input", ""))) + estimate_tokens_from_text(
            str(event.get("note", ""))
        )
        completion_tokens = estimate_tokens_from_text(str(event.get("output", "")))

        cloned = dict(event)
        cloned["token_estimate"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }
        annotated.append(cloned)

        if agent not in by_agent:
            by_agent[agent] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

        by_agent[agent]["prompt_tokens"] += prompt_tokens
        by_agent[agent]["completion_tokens"] += completion_tokens
        by_agent[agent]["total_tokens"] += prompt_tokens + completion_tokens

    return annotated, by_agent


def estimate_cost_usd(token_usage: dict[str, int], model_name: str) -> float:
    """Estimate execution cost in USD based on per-1K token pricing."""
    pricing = DEFAULT_PRICING_USD_PER_1K.get(model_name)
    if not pricing:
        return 0.0

    prompt = token_usage.get("prompt_tokens", 0)
    completion = token_usage.get("completion_tokens", 0)
    cost = (prompt / 1000.0) * pricing["input"] + (completion / 1000.0) * pricing["output"]
    return round(cost, 8)
