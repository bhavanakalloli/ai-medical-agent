"""Drug-oriented tools: conversation symptom extraction and Wikipedia lookup."""

from __future__ import annotations

import re
from typing import Any

import httpx
from langchain_core.tools import tool

MOCK_WIKIPEDIA_DRUG_SNIPPETS = {
    "aspirin": "Aspirin is a nonsteroidal anti-inflammatory drug used for pain, fever, and antiplatelet effects.",
    "ibuprofen": "Ibuprofen is a nonsteroidal anti-inflammatory drug used to treat pain, fever, and inflammation.",
    "paracetamol": "Paracetamol (acetaminophen) is used for pain and fever relief and is not an anti-inflammatory agent.",
    "metformin": "Metformin is a first-line medication for type 2 diabetes and may cause gastrointestinal side effects.",
    "amlodipine": "Amlodipine is a calcium channel blocker used to treat hypertension and angina.",
    "medication": "Medication safety requires checking indications, interactions, dose, renal/hepatic status, and side effects.",
}

SYMPTOM_KEYWORDS = [
    "pain",
    "chest",
    "dizziness",
    "uneasy",
    "nausea",
    "insomnia",
    "sleep",
    "palpitation",
    "bp",
    "blood pressure",
    "headache",
]

DRUG_KEYWORDS = [
    "aspirin",
    "ibuprofen",
    "paracetamol",
    "acetaminophen",
    "metformin",
    "amlodipine",
    "statin",
    "drug",
    "medication",
    "tablet",
    "dose",
    "side effect",
]


def extract_symptoms_and_problems_from_conversation(messages: list[Any]) -> dict[str, Any]:
    """Extract probable symptoms and drug-related terms from conversation messages."""
    text_chunks: list[str] = []
    for msg in messages:
        content = getattr(msg, "content", "")
        if isinstance(content, str):
            text_chunks.append(content.lower())
        else:
            text_chunks.append(str(content).lower())

    merged = " ".join(text_chunks)

    symptoms = sorted({kw for kw in SYMPTOM_KEYWORDS if kw in merged})
    drug_terms = sorted({kw for kw in DRUG_KEYWORDS if kw in merged})

    explicit_names = re.findall(r"\b[a-zA-Z]{4,20}\b", merged)
    candidates = [name for name in explicit_names if name in MOCK_WIKIPEDIA_DRUG_SNIPPETS]
    if not drug_terms and candidates:
        drug_terms = sorted(set(candidates))

    return {
        "symptoms": symptoms,
        "drug_terms": drug_terms,
        "primary_lookup_term": drug_terms[0] if drug_terms else "medication",
    }


def search_wikipedia_drug_info(term: str) -> dict[str, str]:
    """Fetch concise drug information from Wikipedia with deterministic fallback."""
    normalized = term.strip().lower() or "medication"

    if normalized in MOCK_WIKIPEDIA_DRUG_SNIPPETS:
        return {
            "term": normalized,
            "summary": MOCK_WIKIPEDIA_DRUG_SNIPPETS[normalized],
            "source": "wikipedia_mock",
        }

    try:
        url = f"https://en.wikipedia.org/api/rest_v1/page/summary/{normalized}"
        response = httpx.get(url, timeout=3.0)
        response.raise_for_status()
        payload = response.json()
        extract = str(payload.get("extract", "")).strip()
        if extract:
            return {
                "term": normalized,
                "summary": extract[:600],
                "source": "wikipedia_live",
            }
    except Exception:
        pass

    return {
        "term": normalized,
        "summary": (
            "No direct Wikipedia summary available. Use standard medication safety checks: "
            "indication, dose, interactions, contraindications, and red-flag symptoms."
        ),
        "source": "fallback",
    }


@tool("extract_symptoms_from_conversation")
def extract_symptoms_from_conversation(conversation_text: str) -> dict[str, Any]:
    """Extract probable symptoms/problems and drug terms from conversation text."""
    return extract_symptoms_and_problems_from_conversation([conversation_text])


@tool("wikipedia_drug_lookup")
def wikipedia_drug_lookup(term: str) -> dict[str, str]:
    """Look up concise drug information from Wikipedia or fallback guidance."""
    return search_wikipedia_drug_info(term)
