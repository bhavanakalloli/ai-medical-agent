"""Node implementations for supervisor and specialist agents."""

from __future__ import annotations

import json
import re
from typing import Any, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from agents.prompts import (
    CARDIOVASCULAR_PROMPT,
    DRUG_PROMPT,
    SLEEP_PROMPT,
    SUPERVISOR_PROMPT,
    WRITING_PROMPT,
)
from state import AgentName, MedicalAgentState
from tools.cardio_mock_data import cardio_mock_lookup, get_mock_cardiovascular_context
from tools.drug_wikipedia_tools import (
    extract_symptoms_and_problems_from_conversation,
    extract_symptoms_from_conversation,
    search_wikipedia_drug_info,
    wikipedia_drug_lookup,
)

MAX_SUPERVISOR_LOOPS = 6
SPECIALIST_AGENTS = ["cardiovascular", "sleep", "drug"]
MAX_TRACE_NOTE_LEN = 180
MAX_AGENT_OUTPUT_CHARS = {
    "sleep": 650,
    "cardiovascular": 700,
    "drug": 700,
    "writing": 1000,
}


class RouteDecision(BaseModel):
    """Structured routing output produced by the supervisor."""

    next_agent: Literal["cardiovascular", "sleep", "drug", "writing", "finish"] = Field(
        description="The next agent the supervisor should invoke."
    )
    reason: str = Field(description="Short reason for this routing choice.")


def _extract_latest_user_text(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return msg.content if isinstance(msg.content, str) else str(msg.content)
    return ""


def _trace_event(agent: str, input_text: str, output_text: str, note: str = "") -> dict[str, str]:
    if note and len(note) > MAX_TRACE_NOTE_LEN:
        note = note[: MAX_TRACE_NOTE_LEN - 3] + "..."
    return {
        "agent": agent,
        "input": input_text,
        "output": output_text,
        "note": note,
    }


def _is_greeting_only(user_text: str) -> bool:
    text = user_text.strip().lower()
    if not text:
        return False
    greetings = {
        "hi",
        "hey",
        "hello",
        "yo",
        "hola",
        "hii",
        "heyy",
        "good morning",
        "good evening",
    }
    return text in greetings


def _normalize_model_output(content: Any) -> str:
    """Normalize model output into clean display text across provider response shapes."""
    if isinstance(content, str):
        normalized = content.strip()
        return normalized

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text" and item.get("text"):
                    chunks.append(str(item["text"]))
                elif item.get("text"):
                    chunks.append(str(item["text"]))
                elif item.get("content"):
                    chunks.append(str(item["content"]))
            else:
                text_attr = getattr(item, "text", None)
                if text_attr:
                    chunks.append(str(text_attr))
        return "\n".join(chunks).strip()

    if isinstance(content, dict):
        if "text" in content:
            return str(content["text"]).strip()
        if "content" in content:
            return str(content["content"]).strip()

    return str(content).strip()


def _serialize_tool_result(result: Any) -> str:
    if isinstance(result, (dict, list)):
        return json.dumps(result)
    return _normalize_model_output(result)


def _invoke_with_bound_tools(
    *,
    llm: Any,
    base_messages: list[BaseMessage],
    tools: list[BaseTool],
) -> tuple[Any, list[dict[str, str]]]:
    """Run one bind_tools cycle and return final model response plus tool trace events."""
    if not hasattr(llm, "bind_tools"):
        return llm.invoke(base_messages), []

    try:
        bound = llm.bind_tools(tools)
        first = bound.invoke(base_messages)
        tool_calls = getattr(first, "tool_calls", None) or []
        if not tool_calls:
            return first, []

        tool_by_name = {tool.name: tool for tool in tools}
        tool_messages: list[ToolMessage] = []
        traces: list[dict[str, str]] = []

        for call in tool_calls:
            name = str(call.get("name", ""))
            args = call.get("args", {}) or {}
            tool = tool_by_name.get(name)
            if not tool:
                continue

            result = tool.invoke(args)
            result_text = _serialize_tool_result(result)
            tool_call_id = str(call.get("id", ""))
            tool_messages.append(ToolMessage(content=result_text, tool_call_id=tool_call_id, name=name))
            traces.append(
                _trace_event(
                    name,
                    _normalize_model_output(args),
                    result_text,
                    "bind_tools_call",
                )
            )

        final = bound.invoke([*base_messages, first, *tool_messages])
        return final, traces
    except Exception:
        return llm.invoke(base_messages), []


def _writing_fallback_from_messages(messages: list[BaseMessage]) -> str:
    """Create a minimal safe final report when the writing model returns empty output."""
    specialist_lines: list[str] = []
    for msg in messages:
        if isinstance(msg, AIMessage) and isinstance(msg.content, str):
            text = msg.content.strip()
            if text.startswith("[sleep]") or text.startswith("[cardiovascular]") or text.startswith("[drug]"):
                specialist_lines.append(text)

    if not specialist_lines:
        specialist_lines.append("No specialist details were available in this turn.")

    joined = "\n".join(specialist_lines[-3:])
    return (
        "Summary: Preliminary synthesis based on available agent context.\n"
        f"Key concerns: {joined}\n"
        "Suggested next steps: Continue symptom monitoring and seek clinician review for persistent symptoms.\n"
        "Safety note: Seek urgent care for severe worsening pain, breathing trouble, confusion, or sudden neurologic changes."
    )


def _truncate_text(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _apply_agent_output_limit(agent_name: str, output_text: str) -> str:
    max_chars = MAX_AGENT_OUTPUT_CHARS.get(agent_name)
    if not max_chars:
        return output_text
    return _truncate_text(output_text, max_chars)


def _required_agents_from_query(user_text: str) -> list[str]:
    """Infer required specialist coverage for multi-intent triage from user text."""
    text = user_text.lower()
    required: list[str] = []

    sleep_patterns = [r"sleep", r"insomnia", r"haven't slept", r"cannot sleep", r"2 days"]
    cardio_patterns = [
        r"bp",
        r"blood pressure",
        r"heart",
        r"chest",
        r"palpitation",
        r"uneasy",
        r"dizziness",
    ]
    drug_patterns = [
        r"medicine",
        r"medication",
        r"tablet",
        r"drug",
        r"side effect",
        r"dose",
        r"cough",
        r"bloat",
        r"bloated",
        r"bloating",
    ]

    if any(re.search(pattern, text) for pattern in sleep_patterns):
        required.append("sleep")
    if any(re.search(pattern, text) for pattern in cardio_patterns):
        required.append("cardiovascular")
    if any(re.search(pattern, text) for pattern in drug_patterns):
        required.append("drug")

    return required


def _has_general_medical_signal(user_text: str) -> bool:
    text = user_text.lower()
    symptom_patterns = [
        r"cough",
        r"bloat",
        r"bloated",
        r"bloating",
        r"pain",
        r"fever",
        r"dizziness",
        r"headache",
        r"nausea",
        r"vomit",
    ]
    return any(re.search(pattern, text) for pattern in symptom_patterns)


def _is_in_scope_medical_query(user_text: str) -> bool:
    text = user_text.lower()
    in_scope_patterns = [
        r"sleep",
        r"insomnia",
        r"heart",
        r"cardio",
        r"bp",
        r"blood pressure",
        r"chest",
        r"palpitation",
        r"drug",
        r"medication",
        r"medicine",
        r"tablet",
        r"dose",
        r"side effect",
        r"cough",
        r"bloat",
        r"bloated",
        r"bloating",
        r"pain",
        r"fever",
        r"dizziness",
        r"headache",
        r"nausea",
        r"vomit",
        r"sick",
        r"unwell",
    ]
    return any(re.search(pattern, text) for pattern in in_scope_patterns)


def _is_clearly_non_medical_query(user_text: str) -> bool:
    text = user_text.lower()
    non_medical_patterns = [
        r"movie",
        r"cinema",
        r"song",
        r"music",
        r"cricket",
        r"football",
        r"score",
        r"stock",
        r"bitcoin",
        r"recipe",
        r"travel",
        r"visa",
        r"python code",
        r"programming",
        r"debug",
        r"joke",
        r"weather",
        r"news",
    ]
    has_non_medical = any(re.search(pattern, text) for pattern in non_medical_patterns)
    return has_non_medical and (not _is_in_scope_medical_query(user_text))


def _repeated_specialist_tail(route_history: list[str], repeat_count: int = 2) -> str | None:
    if len(route_history) < repeat_count:
        return None
    tail = route_history[-repeat_count:]
    first = tail[0]
    if first not in SPECIALIST_AGENTS:
        return None
    if all(agent == first for agent in tail):
        return first
    return None


def _policy_route(user_text: str, route_history: list[str]) -> tuple[str, str] | None:
    """Return a policy-enforced route when required domain coverage is incomplete."""
    repeated_specialist = _repeated_specialist_tail(route_history, repeat_count=2)
    if repeated_specialist and "writing" not in route_history:
        return "writing", f"Policy forced writing after repeated {repeated_specialist} routing."

    required_agents = _required_agents_from_query(user_text)
    if not required_agents:
        if _has_general_medical_signal(user_text):
            called_specialists = {agent for agent in route_history if agent in SPECIALIST_AGENTS}
            if called_specialists and "writing" not in route_history:
                return "writing", "Policy enforced writing synthesis for general symptom report."
        return None

    called_specialists = {agent for agent in route_history if agent in SPECIALIST_AGENTS}
    pending = [agent for agent in required_agents if agent not in called_specialists]
    if pending:
        next_agent = pending[0]
        return next_agent, f"Policy enforced pending specialist coverage: {pending}."

    if "writing" not in route_history:
        return "writing", "Policy enforced writing synthesis after specialist coverage."

    if route_history and route_history[-1] == "writing":
        return "finish", "Policy finished after writing synthesis."

    return None


def supervisor_node(state: MedicalAgentState, llm: Any) -> dict[str, Any]:
    """Choose the next node based on chat context and route history."""
    loop_count = state.get("loop_count", 0)
    route_history = state.get("route_history", [])
    messages = state.get("messages", [])

    if loop_count >= MAX_SUPERVISOR_LOOPS:
        fallback = "Loop guard reached. Finishing with current information."
        return {
            "next_agent": "finish",
            "loop_count": loop_count + 1,
            "route_history": route_history + ["finish"],
            "execution_trace": state.get("execution_trace", [])
            + [_trace_event("supervisor", _extract_latest_user_text(messages), "finish", fallback)],
        }

    latest_user_text = _extract_latest_user_text(messages)

    if latest_user_text.strip() and _is_clearly_non_medical_query(latest_user_text):
        if "writing" not in route_history:
            return {
                "next_agent": "writing",
                "loop_count": loop_count + 1,
                "route_history": route_history + ["writing"],
                "execution_trace": state.get("execution_trace", [])
                + [
                    _trace_event(
                        "supervisor",
                        latest_user_text,
                        "writing",
                        "Out-of-scope query detected; route to scoped guidance response.",
                    )
                ],
            }
        return {
            "next_agent": "finish",
            "loop_count": loop_count + 1,
            "route_history": route_history + ["finish"],
            "execution_trace": state.get("execution_trace", [])
            + [
                _trace_event(
                    "supervisor",
                    latest_user_text,
                    "finish",
                    "Out-of-scope guidance already provided; finishing turn.",
                )
            ],
        }

    # Keep greeting-only turns concise and avoid unnecessary routing loops.
    if _is_greeting_only(latest_user_text):
        if "writing" not in route_history:
            return {
                "next_agent": "writing",
                "loop_count": loop_count + 1,
                "route_history": route_history + ["writing"],
                "execution_trace": state.get("execution_trace", [])
                + [_trace_event("supervisor", latest_user_text, "writing", "Greeting detected; route to concise writer reply.")],
            }
        return {
            "next_agent": "finish",
            "loop_count": loop_count + 1,
            "route_history": route_history + ["finish"],
            "execution_trace": state.get("execution_trace", [])
            + [_trace_event("supervisor", latest_user_text, "finish", "Greeting handled; finishing turn.")],
        }

    policy_choice = _policy_route(latest_user_text, route_history)
    if policy_choice is not None:
        forced_agent, reason = policy_choice
        return {
            "next_agent": forced_agent,
            "loop_count": loop_count + 1,
            "route_history": route_history + [forced_agent],
            "execution_trace": state.get("execution_trace", [])
            + [_trace_event("supervisor", latest_user_text, forced_agent, reason)],
        }

    structured_llm = llm.with_structured_output(RouteDecision)
    decision: RouteDecision = structured_llm.invoke(
        [
            SystemMessage(content=SUPERVISOR_PROMPT),
            HumanMessage(
                content=(
                    f"Route history: {route_history}\n"
                    f"Loop count: {loop_count}\n"
                    "Pick the next best agent now."
                )
            ),
            *messages,
        ]
    )

    return {
        "next_agent": decision.next_agent,
        "loop_count": loop_count + 1,
        "route_history": route_history + [decision.next_agent],
        "execution_trace": state.get("execution_trace", [])
        + [
            _trace_event(
                "supervisor",
                latest_user_text,
                decision.next_agent,
                decision.reason,
            )
        ],
    }


def _specialist_node(
    state: MedicalAgentState,
    llm: Any,
    *,
    agent_name: AgentName,
    system_prompt: str,
) -> dict[str, Any]:
    messages = state.get("messages", [])
    response = llm.invoke([SystemMessage(content=system_prompt), *messages])
    output_text = _normalize_model_output(getattr(response, "content", response))
    if not output_text or output_text in {"[]", "{}"}:
        output_text = "No content returned by model."
    output_text = _apply_agent_output_limit(agent_name, output_text)

    return {
        "messages": [AIMessage(content=f"[{agent_name}] {output_text}")],
        "execution_trace": state.get("execution_trace", [])
        + [_trace_event(agent_name, _extract_latest_user_text(messages), output_text)],
    }


def cardiovascular_node(state: MedicalAgentState, llm: Any) -> dict[str, Any]:
    messages = state.get("messages", [])
    user_text = _extract_latest_user_text(messages)
    response, tool_traces = _invoke_with_bound_tools(
        llm=llm,
        base_messages=[
            SystemMessage(content=CARDIOVASCULAR_PROMPT),
            HumanMessage(content="Use tool cardio_mock_lookup first, then provide concise cardiovascular reasoning."),
            *messages,
        ],
        tools=[cardio_mock_lookup],
    )

    # Fallback for models that do not support bind_tools.
    if not tool_traces:
        cardio_context = get_mock_cardiovascular_context(user_text)
        tool_note = (
            f"mock_data_patient_id={cardio_context['patient_id']};"
            f"missing_fields={cardio_context['missing_fields']}"
        )
        tool_traces = [
            _trace_event(
                "cardio_mock_tool",
                user_text,
                json.dumps(cardio_context["record"]),
                tool_note,
            )
        ]
        response = llm.invoke(
            [
                SystemMessage(content=CARDIOVASCULAR_PROMPT),
                HumanMessage(
                    content=(
                        "Mock cardiovascular tool output (JSON):\n"
                        f"{json.dumps(cardio_context, indent=2)}"
                    )
                ),
                *messages,
            ]
        )

    output_text = _normalize_model_output(getattr(response, "content", response))
    if not output_text or output_text in {"[]", "{}"}:
        output_text = "No cardiovascular content returned by model."
    output_text = _apply_agent_output_limit("cardiovascular", output_text)

    return {
        "messages": [AIMessage(content=f"[cardiovascular] {output_text}")],
        "execution_trace": state.get("execution_trace", []) + tool_traces + [_trace_event("cardiovascular", user_text, output_text)],
    }


def sleep_node(state: MedicalAgentState, llm: Any) -> dict[str, Any]:
    return _specialist_node(
        state,
        llm,
        agent_name="sleep",
        system_prompt=SLEEP_PROMPT,
    )


def drug_node(state: MedicalAgentState, llm: Any) -> dict[str, Any]:
    messages = state.get("messages", [])
    user_text = _extract_latest_user_text(messages)

    response, tool_traces = _invoke_with_bound_tools(
        llm=llm,
        base_messages=[
            SystemMessage(content=DRUG_PROMPT),
            HumanMessage(
                content=(
                    "First call extract_symptoms_from_conversation with full conversation text, "
                    "then call wikipedia_drug_lookup using the most relevant drug term, and then respond."
                )
            ),
            HumanMessage(content=f"Conversation text:\n{user_text}"),
            *messages,
        ],
        tools=[extract_symptoms_from_conversation, wikipedia_drug_lookup],
    )

    # Fallback for models without bind_tools support.
    if not tool_traces:
        symptom_context = extract_symptoms_and_problems_from_conversation(messages)
        wiki_context = search_wikipedia_drug_info(symptom_context["primary_lookup_term"])
        tool_traces = [
            _trace_event(
                "drug_symptom_extractor",
                user_text,
                json.dumps(symptom_context),
                "Extracted user symptoms/problems for drug context.",
            ),
            _trace_event(
                "drug_wikipedia_search",
                symptom_context["primary_lookup_term"],
                json.dumps(wiki_context),
                f"source={wiki_context.get('source', 'unknown')}",
            ),
        ]
        response = llm.invoke(
            [
                SystemMessage(content=DRUG_PROMPT),
                HumanMessage(
                    content=(
                        "Extracted symptom/problem context (JSON):\n"
                        f"{json.dumps(symptom_context, indent=2)}\n\n"
                        "Wikipedia drug lookup context (JSON):\n"
                        f"{json.dumps(wiki_context, indent=2)}"
                    )
                ),
                *messages,
            ]
        )

    output_text = _normalize_model_output(getattr(response, "content", response))
    if not output_text or output_text in {"[]", "{}"}:
        output_text = "No drug-safety content returned by model."
    output_text = _apply_agent_output_limit("drug", output_text)

    return {
        "messages": [AIMessage(content=f"[drug] {output_text}")],
        "execution_trace": state.get("execution_trace", []) + tool_traces + [_trace_event("drug", user_text, output_text)],
    }


def writing_node(state: MedicalAgentState, llm: Any) -> dict[str, Any]:
    latest_user_text = _extract_latest_user_text(state.get("messages", []))
    if latest_user_text.strip() and _is_clearly_non_medical_query(latest_user_text):
        scoped = (
            "Please ask questions about your sleep, heart, blood pressure, medications, or related symptoms. "
            "I am designed only for those medical issues."
        )
        return {
            "messages": [AIMessage(content=f"[writing] {scoped}")],
            "execution_trace": state.get("execution_trace", [])
            + [_trace_event("writing", latest_user_text, scoped, "Out-of-scope guardrail response.")],
            "final_response": scoped,
        }

    if _is_greeting_only(latest_user_text):
        concise = (
            "Hello. Share your symptoms, duration, medications, and any vitals you have, "
            "and I will provide a structured medical summary with safety guidance."
        )
        return {
            "messages": [AIMessage(content=f"[writing] {concise}")],
            "execution_trace": state.get("execution_trace", [])
            + [_trace_event("writing", latest_user_text, concise, "Concise greeting response.")],
            "final_response": concise,
        }

    result = _specialist_node(
        state,
        llm,
        agent_name="writing",
        system_prompt=WRITING_PROMPT,
    )

    messages = result["messages"]
    final_text = ""
    if messages:
        candidate = messages[0]
        if isinstance(candidate, AIMessage) and isinstance(candidate.content, str):
            final_text = candidate.content
            if final_text.startswith("[writing] "):
                final_text = final_text[len("[writing] ") :]

    marker_text = final_text.strip().lower()
    if (not final_text) or (marker_text in {"[]", "{}"}) or ("no content returned by model" in marker_text):
        final_text = _writing_fallback_from_messages(state.get("messages", []))
        result["messages"] = [AIMessage(content=f"[writing] {final_text}")]

    final_text = _apply_agent_output_limit("writing", final_text)
    result["messages"] = [AIMessage(content=f"[writing] {final_text}")]

    result["final_response"] = final_text
    return result
