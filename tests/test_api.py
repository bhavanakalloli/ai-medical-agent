"""API tests for Phase 7 FastAPI execution and telemetry behavior."""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from database.database import SQLiteMemoryStore
from main import app


class APITests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = os.path.join(self.temp_dir.name, "api_test.db")
        self.store = SQLiteMemoryStore(self.db_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_health_endpoint(self):
        with patch("main.store", self.store):
            client = TestClient(app)
            resp = client.get("/v1/health")
            self.assertEqual(resp.status_code, 200)
            self.assertEqual(resp.json()["status"], "ok")

    def test_execute_uses_retrieval_when_query_matches(self):
        self.store.save_turn(
            session_id="s1",
            query="i have chest pain and high bp",
            response="previous answer",
            turn_count=1,
            conversation_summary="summary",
            execution_trace=[{"agent": "supervisor", "output": "finish"}],
        )

        with patch("main.store", self.store):
            client = TestClient(app)
            resp = client.post(
                "/v1/execute",
                json={
                    "session_id": "s1",
                    "query": "i have chest pain and high bp",
                },
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertTrue(body["retrieval"]["hit"])
        self.assertIn("estimated_cost_usd", body["token_usage"])
        self.assertTrue(body["token_usage"]["is_estimated"])
        self.assertIn("by_agent", body["token_usage"])
        self.assertEqual(body["route_history"][0], "retrieval_hit")

    def test_execute_no_retrieval_uses_mocked_graph_path(self):
        mocked_graph_result = {
            "final_response": "mocked graph answer",
            "route_history": ["sleep", "writing", "finish"],
            "execution_trace": [
                {"agent": "supervisor", "input": "x", "output": "sleep", "note": "mock"},
                {"agent": "sleep", "input": "x", "output": "sleep output", "note": "mock"},
            ],
            "memory": {
                "history_messages": [],
                "conversation_summary": None,
                "turn_count": 1,
                "summary_updated": False,
            },
        }

        with patch("main.store", self.store), patch("main.run_once", return_value=mocked_graph_result):
            client = TestClient(app)
            resp = client.post(
                "/v1/execute",
                json={
                    "session_id": "s2",
                    "query": "new query with no previous match",
                },
            )

        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertFalse(body["retrieval"]["hit"])
        self.assertEqual(body["final_report"], "mocked graph answer")
        self.assertEqual(body["route_history"], ["sleep", "writing", "finish"])
        self.assertIn("estimated_cost_usd", body["token_usage"])
        self.assertIn("by_agent", body["token_usage"])


if __name__ == "__main__":
    unittest.main()
