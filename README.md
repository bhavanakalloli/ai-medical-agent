# AI Medical Supervisor — Multi-Agent System

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangGraph](https://img.shields.io/badge/Multi--Agent-LangGraph-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

An AI-powered multi-agent healthcare assistant built using supervisor-driven orchestration for intelligent medical triage, specialist routing, and structured response generation.

---

## My Contributions

This project was built on top of a base scaffolding. Everything listed below was designed, implemented, and documented by me:

- **Supervisor-based routing architecture** — designed the LangGraph graph with a central supervisor agent dynamically routing queries to 4 specialist agents based on user intent
- **4 specialist agents** — implemented Sleep Specialist, Cardiovascular Specialist, Drug Interaction, and Response Writing agents, each with dedicated tool reasoning and fallback logic
- **SQLite session memory** — built persistent session-based memory using SQLite so conversations retain context across turns
- **Token & cost telemetry** — instrumented all agent calls in `utils/telemetry.py` to track token usage and API cost per session
- **Test suite** — wrote unit and integration tests in `tests/` covering 5 areas: graph routing logic, tool trace validation, fallback behavior, API telemetry, and retrieval behavior
- **EVAL.md** — created an evaluation document describing how agent output quality is measured, what failure modes were identified, and how recovery paths were implemented
- **FastAPI backend** — built the API layer with 4 endpoints (`/v1/health`, `/v1/execute`, `/v1/session/{id}`, `/v1/sessions`)
- **Streamlit frontend** — developed the interactive chat UI with execution trace display, session history, and response insights

---

## Architecture

```
User Query
    └── Supervisor Agent (LangGraph)
            ├── Sleep Specialist Agent
            ├── Cardiovascular Specialist Agent
            ├── Drug Interaction Agent
            └── Response Writing Agent
                        └── Final Response
```

Specialist agents are selected dynamically based on user intent. Each agent has access to dedicated tools and falls back gracefully when queries fall outside its scope.

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Agent orchestration | LangGraph |
| Backend API | FastAPI |
| LLM | Google Gemini API |
| Memory | SQLite |
| Frontend | Streamlit |
| Telemetry | Custom (`utils/telemetry.py`) |
| Testing | Python unittest |

---

## Project Structure

```
.
├── main.py                  # FastAPI entry point
├── streamlit_app.py         # Frontend UI
├── state.py                 # Shared agent state
├── agents/
│   ├── graph.py             # LangGraph supervisor graph
│   └── nodes.py             # Specialist agent nodes
├── tools/                   # Agent tool definitions
├── database/
│   └── database.py          # SQLite session memory
├── utils/
│   └── telemetry.py         # Token & cost tracking
├── tests/                   # Test suite
├── EVAL.md                  # Evaluation framework & failure analysis
└── requirements.txt
```

---

## Setup

### Prerequisites
- Python 3.12+
- Google Gemini API key

```bash
export GOOGLE_API_KEY="your_key_here"
pip install -r requirements.txt
```

### Run Backend
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### Run Frontend
```bash
streamlit run streamlit_app.py
```

### Run Tests
```bash
python -m unittest -v tests.test_architecture tests.test_api
```

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/v1/health` | Health check |
| POST | `/v1/execute` | Submit a query |
| GET | `/v1/session/{session_id}` | Fetch session history |
| GET | `/v1/sessions` | List all sessions |

---

## Evaluation

See [EVAL.md](./EVAL.md) for a detailed breakdown of:
- Agent output quality measurement
- Identified failure modes (hallucination, tool misuse, routing errors)
- Recovery paths and fallback logic
- Test coverage rationale

---

## Disclaimer

This project is for educational and research purposes only. Outputs are non-diagnostic and should not replace professional medical advice.
