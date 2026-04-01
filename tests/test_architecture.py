"""Phase 1 architecture tests for LangGraph routing structure."""

from __future__ import annotations

import os
import tempfile
import unittest

from langchain_core.messages import HumanMessage

from agents.context_memory import (
    build_context_from_db_fetch,
    build_messages_for_turn,
    update_conversation_memory,
)
from agents import nodes
from agents.graph import _route_from_supervisor
from agents.graph import build_graph
from database.database import SQLiteMemoryStore
from tools.cardio_mock_data import get_mock_cardiovascular_context


class FakeStructuredRouter:
    def __init__(self, parent):
        self.parent = parent

    def invoke(self, _messages):
        if self.parent.route_index >= len(self.parent.route_outputs):
            pick = self.parent.route_outputs[-1]
        else:
            pick = self.parent.route_outputs[self.parent.route_index]
        self.parent.route_index += 1

        class Decision:
            def __init__(self, next_agent, reason):
                self.next_agent = next_agent
                self.reason = reason

        return Decision(pick, f"route to {pick}")


class FakeLLM:
    def __init__(self, route_outputs):
        self.route_outputs = route_outputs
        self.route_index = 0
        self.agent_invocations = 0

    def with_structured_output(self, _schema):
        return FakeStructuredRouter(self)

    def invoke(self, _messages):
        self.agent_invocations += 1

        if _messages and hasattr(_messages[0], "content"):
            first_content = str(_messages[0].content)
            if "medical writing agent" in first_content:
                return type(
                    "Message",
                    (),
                    {
                        "content": (
                            "Summary: synthesized report\n"
                            "Key concerns: clinical uncertainty\n"
                            "Suggested next steps: follow-up\n"
                            "Safety note: seek urgent care if worsening"
                        )
                    },
                )()

        class Message:
            def __init__(self, content):
                self.content = content

        return Message(f"mock specialist response {self.agent_invocations}")


class EmptyWritingLLM(FakeLLM):
    def invoke(self, _messages):
        if _messages and hasattr(_messages[0], "content"):
            first_content = str(_messages[0].content)
            if "medical writing agent" in first_content:
                class Message:
                    def __init__(self):
                        self.content = []

                return Message()

        return super().invoke(_messages)


class ArchitectureTests(unittest.TestCase):
    def test_graph_compiles(self):
        llm = FakeLLM(route_outputs=["finish"])
        graph = build_graph(llm=llm)
        self.assertIsNotNone(graph)

    def test_supervisor_can_route_all_agents(self):
        llm = FakeLLM(
            route_outputs=[
                "cardiovascular",
                "sleep",
                "drug",
                "writing",
                "finish",
            ]
        )
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="complex multi-domain query")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertIn("cardiovascular", route_history)
        self.assertIn("sleep", route_history)
        self.assertIn("drug", route_history)
        self.assertIn("writing", route_history)
        self.assertEqual(route_history[-1], "finish")
        self.assertTrue(result.get("final_response"))

    def test_cardio_tool_returns_structured_record(self):
        context = get_mock_cardiovascular_context("chest pain with hypertension")
        self.assertIn("patient_id", context)
        self.assertIn("record", context)
        self.assertIn("vitals", context["record"])
        self.assertIn("labs", context["record"])
        self.assertEqual(context["missing_fields"], [])

    def test_cardio_route_adds_tool_trace(self):
        llm = FakeLLM(route_outputs=["cardiovascular", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="chest tightness and high blood pressure")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        trace = result.get("execution_trace", [])
        agents_called = [event.get("agent") for event in trace]
        self.assertIn("cardio_mock_tool", agents_called)
        self.assertIn("cardiovascular", agents_called)

    def test_drug_route_adds_extractor_and_wikipedia_trace(self):
        llm = FakeLLM(route_outputs=["drug", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "I have headache and dizziness after taking aspirin medication"
                        )
                    )
                ],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        trace = result.get("execution_trace", [])
        agents_called = [event.get("agent") for event in trace]
        self.assertIn("drug_symptom_extractor", agents_called)
        self.assertIn("drug_wikipedia_search", agents_called)
        self.assertIn("drug", agents_called)

    def test_supervisor_enforces_cardio_drug_hybrid_flow(self):
        llm = FakeLLM(route_outputs=["finish", "finish", "finish", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "I have high blood pressure and chest discomfort after taking a new medication"
                        )
                    )
                ],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertIn("cardiovascular", route_history)
        self.assertIn("drug", route_history)
        self.assertIn("writing", route_history)
        self.assertEqual(route_history[-1], "finish")

    def test_supervisor_enforces_sleep_cardio_shared_context_flow(self):
        llm = FakeLLM(route_outputs=["finish", "finish", "finish", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "I haven't slept in 2 days and I feel uneasy, maybe my BP is high or low"
                        )
                    )
                ],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertIn("sleep", route_history)
        self.assertIn("cardiovascular", route_history)
        self.assertIn("writing", route_history)
        self.assertEqual(route_history[-1], "finish")

    def test_full_multi_intent_can_use_all_nodes(self):
        llm = FakeLLM(route_outputs=["finish", "finish", "finish", "finish", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "I have not slept for 2 days, I feel uneasy, my BP may be high, "
                            "and I took extra medication"
                        )
                    )
                ],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertIn("sleep", route_history)
        self.assertIn("cardiovascular", route_history)
        self.assertIn("drug", route_history)
        self.assertIn("writing", route_history)
        self.assertEqual(route_history[-1], "finish")

    def test_ambiguous_query_can_follow_llm_path(self):
        llm = FakeLLM(route_outputs=["sleep", "writing", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="I feel off and not sure what is happening")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertEqual(route_history, ["sleep", "writing", "finish"])

    def test_invalid_route_falls_back_to_finish(self):
        next_step = _route_from_supervisor({"next_agent": "unknown-route"})
        self.assertEqual(next_step, "finish")

    def test_loop_guard_prevents_infinite_cycle(self):
        llm = FakeLLM(route_outputs=["sleep"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="hello there")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertEqual(route_history[-1], "finish")
        self.assertLessEqual(len(route_history), nodes.MAX_SUPERVISOR_LOOPS + 1)

    def test_writing_output_contains_synthesis_sections(self):
        llm = FakeLLM(route_outputs=["writing", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="Please summarize safely")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        final_response = result.get("final_response", "")
        self.assertIn("Summary", final_response)
        self.assertIn("Key concerns", final_response)
        self.assertIn("Suggested next steps", final_response)
        self.assertIn("Safety note", final_response)

    def test_writing_fallback_avoids_empty_response(self):
        llm = EmptyWritingLLM(route_outputs=["sleep", "writing", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="sleep issue and headache")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        final_response = result.get("final_response", "")
        self.assertTrue(final_response)
        self.assertNotEqual(final_response.strip(), "[]")

    def test_repeated_specialist_route_forces_writing(self):
        llm = FakeLLM(route_outputs=["cardiovascular", "cardiovascular", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="I have cough and bloating symptoms")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertIn("writing", route_history)
        self.assertEqual(route_history[-1], "finish")
        self.assertTrue(result.get("final_response"))

    def test_out_of_scope_prompt_returns_guardrail_response(self):
        llm = FakeLLM(route_outputs=["cardiovascular", "finish"])
        graph = build_graph(llm=llm)

        result = graph.invoke(
            {
                "messages": [HumanMessage(content="Tell me a cricket score and movie recommendations")],
                "next_agent": None,
                "route_history": [],
                "loop_count": 0,
                "final_response": None,
                "execution_trace": [],
                "token_usage": {},
            }
        )

        route_history = result.get("route_history", [])
        self.assertEqual(route_history, ["writing", "finish"])
        final_response = result.get("final_response", "")
        self.assertIn("Please ask questions about your sleep, heart", final_response)

    def test_summary_triggers_on_20th_turn_and_trims_history(self):
        memory = {
            "history_messages": [],
            "conversation_summary": None,
            "turn_count": 19,
        }

        updated = update_conversation_memory(
            history_messages=memory["history_messages"],
            conversation_summary=memory["conversation_summary"],
            turn_count=memory["turn_count"],
            query="new turn question",
            final_response="new turn answer",
            summary_every=20,
            recent_window=2,
        )

        self.assertTrue(updated["summary_updated"])
        self.assertEqual(updated["turn_count"], 20)
        self.assertTrue(updated["conversation_summary"])
        self.assertEqual(len(updated["history_messages"]), 2)

    def test_message_builder_injects_summary_and_recent_window(self):
        history = [
            HumanMessage(content="u1"),
            HumanMessage(content="u2"),
            HumanMessage(content="u3"),
        ]
        payload = build_messages_for_turn(
            query="latest",
            history_messages=history,
            conversation_summary="summarized prior context",
            recent_window=2,
        )

        self.assertEqual(len(payload), 4)
        self.assertIn("Conversation summary", str(payload[0].content))
        self.assertEqual(str(payload[-1].content), "latest")

    def test_db_fetch_goes_through_summary_layer(self):
        fetched = [
            {"role": "user", "content": "I am not sleeping"},
            {"role": "assistant", "content": "Track sleep schedule"},
            {"role": "user", "content": "My BP seems high today"},
        ]
        context = build_context_from_db_fetch(
            fetched_records=fetched,
            existing_summary="older summary",
            recent_window=2,
        )

        self.assertTrue(context["conversation_summary"])
        self.assertIn("Previous summary", context["conversation_summary"])
        self.assertEqual(len(context["history_messages"]), 2)

    def test_sqlite_store_persists_and_loads_session_data(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = os.path.join(temp_dir, "test_medical_agent.db")
            store = SQLiteMemoryStore(db_path=db_path)

            store.save_turn(
                session_id="session_1",
                query="first query",
                response="first answer",
                turn_count=1,
                conversation_summary="summary text",
                execution_trace=[{"agent": "supervisor", "output": "finish"}],
            )

            loaded = store.load_session_context("session_1")
            self.assertEqual(loaded["turn_count"], 1)
            self.assertEqual(loaded["conversation_summary"], "summary text")
            self.assertEqual(len(loaded["messages"]), 2)
            self.assertEqual(loaded["messages"][0]["role"], "user")
            self.assertEqual(loaded["messages"][1]["role"], "assistant")


if __name__ == "__main__":
    unittest.main()
