# AI Medical Supervisor Multi-Agent System

![Python](https://img.shields.io/badge/Python-3.12+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-green)
![LangGraph](https://img.shields.io/badge/Multi--Agent-LangGraph-orange)
![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-red)

An AI-powered multi-agent healthcare assistant built using supervisor-driven orchestration for intelligent medical triage, specialist routing, and structured response generation.

---

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

User Query → Supervisor Agent → Specialist Routing → Tool Reasoning → Writing Agent → Final Response

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
├── main.py
├── agents/
│   ├── graph.py
│   └── nodes.py
├── tools/
├── database/database.py
├── utils/telemetry.py
├── streamlit_app.py
├── tests/
└── EVAL.md
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

Endpoints:

- GET `/v1/health`
- POST `/v1/execute`
- GET `/v1/session/{session_id}`
- GET `/v1/sessions`

---

## Run Frontend

```bash
streamlit run streamlit_app.py
```

Features include:
- Interactive medical assistant chat  
- Session history  
- Execution traces  
- Response insights

---

## Testing

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

## Demo Preview

_Add screenshots or demo GIFs here._

---

## Customization & Enhancements

- Environment setup and deployment customization  
- Documentation and repository improvements  
- Planned enhancements for advanced multi-agent healthcare workflows

---

## Future Improvements

- Voice-enabled medical assistant  
- Additional specialist agents  
- Electronic health record integration  
- Advanced multi-agent collaboration

---

## Disclaimer

This project is for educational and research purposes only.  
Outputs are non-diagnostic and should not replace professional medical advice.

---

## Credits

Original project created by Akash Gupta.  
Adapted, documented, and enhanced by Bhavana Kalloli.
