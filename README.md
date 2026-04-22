# AI Medical Supervisor Multi-Agent System

An AI-powered multi-agent healthcare assistant built using supervisor-driven orchestration for intelligent medical triage, specialist routing, and structured response generation.

## Overview
This project uses a supervisor agent to dynamically route medical queries to specialized agents such as:

- Sleep Specialist Agent  
- Cardiovascular Specialist Agent  
- Drug Interaction Agent  
- Response Writing Agent  

The system combines multi-agent reasoning, tool integration, persistent memory, and an interactive frontend for healthcare-oriented assistance.

---

## Key Features

- Supervisor-based agent routing using LangGraph  
- Tool-augmented specialist reasoning with fallback support  
- Session-based conversation memory using SQLite  
- Token and cost telemetry tracking  
- Interactive Streamlit chat interface  
- FastAPI backend for scalable API access  
- Safety-oriented structured medical guidance

---

## Architecture
Multi-agent workflow includes:

User Query → Supervisor Agent → Specialist Agent Routing → Tool Reasoning → Writing Agent → Final Response

Specialist agents are selected dynamically based on user intent.

---

## Tech Stack

- Python  
- FastAPI  
- LangGraph  
- Streamlit  
- SQLite  
- Google Gemini API  
- AI Agent Tooling

---

## Project Structure

```bash
.
├── main.py                    # FastAPI orchestration
├── agents/
│   ├── graph.py               # LangGraph routing
│   └── nodes.py               # Supervisor + specialist logic
├── tools/                     # Tool integrations
├── database/database.py       # Persistence layer
├── utils/telemetry.py         # Cost/token estimates
├── streamlit_app.py           # Frontend UI
├── tests/                     # Test suites
└── EVAL.md                    # Evaluation analysis
```

---

## Setup

### Prerequisites
- Python 3.12+
- Google Gemini API Key

Set environment variable:

```bash
export GOOGLE_API_KEY="your_key_here"
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run Backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Available endpoints:

- GET `/v1/health`
- POST `/v1/execute`
- GET `/v1/session/{session_id}`
- GET `/v1/sessions`

---

## Run Frontend

```bash
streamlit run streamlit_app.py
```

Launches interactive medical assistant interface with:
- Session history
- Execution traces
- Response insights

---

## Testing

Run all tests:

```bash
python -m unittest -v tests.test_architecture tests.test_api
```

Covers:
- Graph routing logic
- Tool trace validation
- Fallback behavior
- API telemetry
- Retrieval behavior

---

## My Contributions / Enhancements
- Environment setup and deployment customization  
- Repository curation and documentation improvements  
- Planned future enhancements for multi-agent healthcare workflows

---

## Future Improvements
- Voice-enabled medical assistant  
- Expanded specialist agents  
- EHR integration  
- Multi-agent collaboration workflows

---

## Disclaimer
This system is for educational and research purposes only.  
Outputs are non-diagnostic and should not replace professional medical advice.

---

## Credits
Original project inspired by Akash Gupta.  
Customized, documented, and maintained by Bhavana.
