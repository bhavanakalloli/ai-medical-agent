"""SQLite persistence layer for conversation sessions and messages."""

from __future__ import annotations

from difflib import SequenceMatcher
import sqlite3
from pathlib import Path
from typing import Any


class SQLiteMemoryStore:
    """Stores and retrieves multi-turn conversation data for sessions."""

    def __init__(self, db_path: str = "database/medical_agent.db") -> None:
        self.db_path = db_path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    conversation_summary TEXT,
                    turn_count INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY(session_id) REFERENCES sessions(session_id)
                );

                CREATE INDEX IF NOT EXISTS idx_messages_session_id_id
                ON messages(session_id, id);
                """
            )

    def ensure_session(self, session_id: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO sessions (session_id)
                VALUES (?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id,),
            )

    def load_session_context(self, session_id: str) -> dict[str, Any]:
        self.ensure_session(session_id)
        with self._connect() as conn:
            session_row = conn.execute(
                "SELECT conversation_summary, turn_count FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            msg_rows = conn.execute(
                "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id ASC",
                (session_id,),
            ).fetchall()

        return {
            "conversation_summary": session_row["conversation_summary"] if session_row else None,
            "turn_count": int(session_row["turn_count"]) if session_row else 0,
            "messages": [{"role": row["role"], "content": row["content"]} for row in msg_rows],
        }

    def save_turn(
        self,
        *,
        session_id: str,
        query: str,
        response: str,
        turn_count: int,
        conversation_summary: str | None,
        execution_trace: list[dict[str, Any]] | None = None,
    ) -> None:
        self.ensure_session(session_id)
        with self._connect() as conn:
            conn.executemany(
                "INSERT INTO messages (session_id, role, content) VALUES (?, ?, ?)",
                [
                    (session_id, "user", query),
                    (session_id, "assistant", response),
                ],
            )
            conn.execute(
                """
                UPDATE sessions
                SET conversation_summary = ?,
                    turn_count = ?,
                    updated_at = CURRENT_TIMESTAMP
                WHERE session_id = ?
                """,
                (conversation_summary, turn_count, session_id),
            )

    def load_recent_messages(self, session_id: str, limit: int = 8) -> list[dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT role, content
                FROM messages
                WHERE session_id = ?
                ORDER BY id DESC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
        records = [{"role": row["role"], "content": row["content"]} for row in rows]
        records.reverse()
        return records

    def list_sessions(self, limit: int = 200) -> list[dict[str, Any]]:
        """List available sessions ordered by most recently updated."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT session_id, turn_count, updated_at
                FROM sessions
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        return [
            {
                "session_id": row["session_id"],
                "turn_count": int(row["turn_count"]),
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def find_similar_query_response(
        self,
        *,
        session_id: str,
        query: str,
        min_similarity: float = 0.9,
    ) -> dict[str, Any] | None:
        """Find a similar prior user query in the same session and return its paired response."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT
                    u.id AS user_id,
                    u.content AS user_query,
                    (
                        SELECT a.content
                        FROM messages a
                        WHERE a.session_id = u.session_id
                          AND a.id > u.id
                          AND a.role = 'assistant'
                        ORDER BY a.id ASC
                        LIMIT 1
                    ) AS assistant_response
                FROM messages u
                WHERE u.session_id = ?
                  AND u.role = 'user'
                ORDER BY u.id DESC
                LIMIT 100
                """,
                (session_id,),
            ).fetchall()

        best_match: dict[str, Any] | None = None
        for row in rows:
            old_query = row["user_query"] or ""
            similarity = SequenceMatcher(a=query.lower(), b=old_query.lower()).ratio()
            if similarity >= min_similarity and row["assistant_response"]:
                candidate = {
                    "matched_query": old_query,
                    "matched_response": row["assistant_response"],
                    "similarity": similarity,
                }
                if best_match is None or candidate["similarity"] > best_match["similarity"]:
                    best_match = candidate

        return best_match
