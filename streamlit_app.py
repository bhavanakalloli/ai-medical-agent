"""Streamlit frontend for the Medical AI Multi-Agent System."""

from __future__ import annotations

import uuid
from typing import Any

import httpx
import streamlit as st


def build_url(base_url: str, path: str) -> str:
    """Safely concatenate API base URL and endpoint path."""
    return f"{base_url.rstrip('/')}/{path.lstrip('/')}"


def call_execute(base_url: str, query: str, session_id: str) -> dict[str, Any]:
    """Invoke POST /v1/execute."""
    payload = {"query": query, "session_id": session_id}
    with httpx.Client(timeout=90.0) as client:
        response = client.post(build_url(base_url, "/v1/execute"), json=payload)
        response.raise_for_status()
        return response.json()


def call_session(base_url: str, session_id: str) -> dict[str, Any]:
    """Invoke GET /v1/session/{session_id}."""
    with httpx.Client(timeout=30.0) as client:
        response = client.get(build_url(base_url, f"/v1/session/{session_id}"))
        response.raise_for_status()
        return response.json()


def call_health(base_url: str) -> dict[str, Any]:
    """Invoke GET /v1/health."""
    with httpx.Client(timeout=15.0) as client:
        response = client.get(build_url(base_url, "/v1/health"))
        response.raise_for_status()
        return response.json()


def call_sessions(base_url: str, limit: int = 200) -> dict[str, Any]:
    """Invoke GET /v1/sessions."""
    with httpx.Client(timeout=20.0) as client:
        response = client.get(build_url(base_url, f"/v1/sessions?limit={limit}"))
        response.raise_for_status()
        return response.json()


def inject_styles() -> None:
    """Inject custom CSS for a bold, readable look and responsive layout."""
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap');

        :root {
            --ink: #102a43;
            --teal: #0ea5a4;
            --amber: #f59e0b;
            --mint: #d1fae5;
            --sky: #e0f2fe;
            --rose: #ffe4e6;
            --card: rgba(255, 255, 255, 0.88);
        }

        .stApp {
            background:
              radial-gradient(circle at 8% 12%, var(--mint) 0%, transparent 32%),
              radial-gradient(circle at 92% 20%, var(--rose) 0%, transparent 38%),
              radial-gradient(circle at 58% 84%, var(--sky) 0%, transparent 40%),
              linear-gradient(130deg, #f0fdfa, #fff7ed 50%, #f8fafc 100%);
            color: var(--ink);
            font-family: 'Space Grotesk', sans-serif;
        }

        h1, h2, h3 {
            font-family: 'Source Serif 4', serif;
            letter-spacing: -0.01em;
            color: #0b2545;
        }

        .block-container {
            padding-top: 1.5rem;
            padding-bottom: 2.5rem;
            max-width: 1200px;
        }

        div[data-testid='stSidebar'] {
            background: linear-gradient(180deg, #0f172a 0%, #0b2545 100%);
            color: #e2e8f0;
        }

        div[data-testid='stSidebar'] * {
            color: #e2e8f0 !important;
        }

        .hero {
            background: var(--card);
            border: 1px solid rgba(14, 165, 164, 0.25);
            border-radius: 18px;
            padding: 1.1rem 1.25rem;
            box-shadow: 0 14px 40px rgba(15, 23, 42, 0.12);
            animation: reveal 420ms ease-out;
        }

        @keyframes reveal {
            from {
                opacity: 0;
                transform: translateY(8px);
            }
            to {
                opacity: 1;
                transform: translateY(0);
            }
        }

        .metric-card {
            background: var(--card);
            border: 1px solid rgba(16, 42, 67, 0.08);
            border-radius: 14px;
            padding: 0.75rem 0.9rem;
        }

        /* Keep chat response text dark for readability on light chat bubbles. */
        div[data-testid='stChatMessageContent'],
        div[data-testid='stChatMessageContent'] p,
        div[data-testid='stChatMessageContent'] span,
        div[data-testid='stChatMessageContent'] li {
            color: #111111 !important;
        }

        @media (max-width: 768px) {
            .block-container {
                padding-left: 0.9rem;
                padding-right: 0.9rem;
            }
            .hero {
                padding: 0.95rem;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    """Initialize stable defaults once per browser session."""
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "latest_result" not in st.session_state:
        st.session_state.latest_result = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "known_sessions" not in st.session_state:
        st.session_state.known_sessions = []
    if "loaded_session_id" not in st.session_state:
        st.session_state.loaded_session_id = None


def load_session_into_chat(base_url: str, session_id: str) -> None:
    """Load a stored session's messages into chat history UI state."""
    session_data = call_session(base_url, session_id)
    messages = session_data.get("messages", [])

    chat_rows: list[dict[str, str]] = []
    for msg in messages:
        role = str(msg.get("role", "assistant"))
        content = str(msg.get("content", ""))
        if role not in {"user", "assistant"}:
            role = "assistant"
        chat_rows.append({"role": role, "content": content})

    st.session_state.chat_history = chat_rows
    st.session_state.loaded_session_id = session_id
    st.session_state.latest_result = None


def render_sidebar() -> tuple[str, str]:
    """Render configuration controls and return base URL and session id."""
    st.sidebar.title("Medical AI Console")
    base_url = st.sidebar.text_input("API base URL", value="http://localhost:8000")

    session_id = st.sidebar.text_input("Session ID", value=st.session_state.session_id)
    st.session_state.session_id = session_id.strip() or st.session_state.session_id

    if st.sidebar.button("Generate New Session", use_container_width=True):
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()

    if st.sidebar.button("Refresh Sessions", use_container_width=True):
        try:
            payload = call_sessions(base_url)
            st.session_state.known_sessions = payload.get("sessions", [])
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or exc.response.reason_phrase
            st.sidebar.error(f"Failed to load sessions ({exc.response.status_code}): {detail}")
        except httpx.RequestError as exc:
            st.sidebar.error(f"Could not reach API for sessions: {exc}")

    if not st.session_state.known_sessions:
        try:
            payload = call_sessions(base_url)
            st.session_state.known_sessions = payload.get("sessions", [])
        except Exception:
            pass

    sessions = st.session_state.known_sessions
    if sessions:
        labels = [f"{item['session_id']}  | turns: {item['turn_count']}" for item in sessions]
        selected = st.sidebar.selectbox("Stored Sessions", options=labels, index=0, key="stored_session_pick")
        selected_id = selected.split("  | ")[0]
        if st.sidebar.button("Open Selected Session", use_container_width=True):
            try:
                st.session_state.session_id = selected_id
                load_session_into_chat(base_url, selected_id)
                st.rerun()
            except httpx.HTTPStatusError as exc:
                detail = exc.response.text.strip() or exc.response.reason_phrase
                st.sidebar.error(f"Failed to open session ({exc.response.status_code}): {detail}")
            except httpx.RequestError as exc:
                st.sidebar.error(f"Could not reach API while opening session: {exc}")
    else:
        st.sidebar.caption("No sessions loaded yet. Click Refresh Sessions.")

    if st.sidebar.button("Health Check", use_container_width=True):
        try:
            health = call_health(base_url)
            st.sidebar.success("API is healthy")
            st.sidebar.json(health)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or exc.response.reason_phrase
            st.sidebar.error(f"Health check failed ({exc.response.status_code}): {detail}")
        except httpx.RequestError as exc:
            st.sidebar.error(f"Could not reach API: {exc}")

    st.sidebar.caption("Tip: start backend with uvicorn main:app --reload")
    return base_url, st.session_state.session_id


def render_chatbot_workspace(base_url: str, session_id: str) -> None:
    """Render chatbot interface with right-side execution insights."""
    st.markdown(
        """
        <div class='hero'>
          <h2 style='margin-bottom:0.25rem;'>Clinical Copilot Chat</h2>
          <p style='margin:0;color:#334155;'>Ask clinical questions in chat. Escalation path and token telemetry update live on the right.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    left_col, right_col = st.columns([1.35, 1], gap="large")

    with left_col:
        st.subheader("Chat")

        if not st.session_state.chat_history:
            st.info("Start the conversation by entering a patient query below.")

        for msg in st.session_state.chat_history:
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            with st.chat_message(role):
                st.markdown(content)

        query = st.chat_input("Describe symptoms, history, concerns, or follow-up questions...")
        if query:
            if not session_id.strip():
                st.warning("Session ID cannot be empty.")
                return

            st.session_state.chat_history.append({"role": "user", "content": query.strip()})

            with st.status("Executing agents...", expanded=True) as status:
                st.write("Routing through Triage, GP, Specialist, Drug, and Writer nodes.")
                try:
                    result = call_execute(base_url, query.strip(), session_id.strip())
                    st.session_state.latest_result = result
                    assistant_text = result.get("final_report", "No report was returned.")
                    st.session_state.chat_history.append({"role": "assistant", "content": assistant_text})
                    status.update(label="Execution complete", state="complete", expanded=False)
                    st.rerun()
                except httpx.HTTPStatusError as exc:
                    detail = exc.response.text.strip() or exc.response.reason_phrase
                    status.update(label="Execution failed", state="error", expanded=True)
                    st.error(f"API error ({exc.response.status_code}): {detail}")
                except httpx.RequestError as exc:
                    status.update(label="Execution failed", state="error", expanded=True)
                    st.error(f"Connection failed: {exc}")

    with right_col:
        st.subheader("Execution Insights")
        result = st.session_state.latest_result

        if not result:
            st.caption("No run yet. Send a chat message to populate insights.")
            return

        metric_cols = st.columns(3)
        metric_cols[0].metric("Session", result.get("session_id", "-"))
        metric_cols[1].metric("Time (ms)", result.get("time_taken_ms", "-"))
        metric_cols[2].metric("Agent Hops", len(result.get("execution_trace", [])))

        st.subheader("Escalation Path")
        path = result.get("route_history", []) or result.get("escalation_path", [])
        if path:
            st.write(" -> ".join(path))
        else:
            st.caption("No escalations recorded.")

        st.subheader("Token Usage")
        st.json(result.get("token_usage", {}))

        st.subheader("Retrieval")
        st.json(result.get("retrieval", {"hit": False}))

        st.subheader("Execution Trace")
        trace = result.get("execution_trace", [])
        if trace:
            st.json(trace)
        else:
            st.caption("No trace events available.")

        with st.expander("Conversation Context", expanded=False):
            context = result.get("conversation_context", {})
            if context:
                st.json(context)
            else:
                st.caption("No prior context for this session.")


def render_session_tab(base_url: str) -> None:
    """Render session lookup by ID."""
    st.markdown(
        """
        <div class='hero'>
          <h2 style='margin-bottom:0.25rem;'>Session Retrieval</h2>
          <p style='margin:0;color:#334155;'>Load a previous assessment from SQLite using its session identifier.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    lookup_id = st.text_input("Session ID to fetch", placeholder="Paste a session UUID")
    if st.button("Fetch Session", use_container_width=True):
        if not lookup_id.strip():
            st.warning("Please provide a session ID.")
            return
        try:
            session_data = call_session(base_url, lookup_id.strip())
            st.success("Session retrieved")
            st.json(session_data)
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text.strip() or exc.response.reason_phrase
            st.error(f"Lookup failed ({exc.response.status_code}): {detail}")
        except httpx.RequestError as exc:
            st.error(f"Connection failed: {exc}")


def main() -> None:
    """Streamlit app entrypoint."""
    st.set_page_config(
        page_title="Medical AI Multi-Agent Frontend",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_styles()
    init_state()

    base_url, session_id = render_sidebar()

    st.title("Medical AI Multi-Agent Frontend")
    st.caption("Chat-first operator interface with real-time execution insights")

    chat_tab, session_tab = st.tabs(["Chat Workspace", "Retrieve Session"])
    with chat_tab:
        render_chatbot_workspace(base_url, session_id)
    with session_tab:
        render_session_tab(base_url)


if __name__ == "__main__":
    main()
