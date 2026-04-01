"""Mock cardiovascular dataset and retrieval helper for Phase 2."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.tools import tool

# Structured mock patient records used only in Phase 2.
MOCK_CARDIO_RECORDS: dict[str, dict[str, Any]] = {
    "patient_alpha": {
        "patient_id": "patient_alpha",
        "age": 58,
        "sex": "male",
        "vitals": {
            "bp_systolic": 154,
            "bp_diastolic": 94,
            "heart_rate": 96,
        },
        "labs": {
            "ldl": 168,
            "hdl": 38,
            "triglycerides": 210,
            "hba1c": 6.4,
        },
        "history": ["hypertension", "former smoker"],
        "symptoms": ["chest tightness on exertion", "mild dyspnea"],
    },
    "patient_beta": {
        "patient_id": "patient_beta",
        "age": 44,
        "sex": "female",
        "vitals": {
            "bp_systolic": 126,
            "bp_diastolic": 82,
            "heart_rate": 74,
        },
        "labs": {
            "ldl": 98,
            "hdl": 56,
            "triglycerides": 132,
            "hba1c": 5.5,
        },
        "history": ["family_history_cad"],
        "symptoms": ["intermittent palpitations"],
    },
}

REQUIRED_RECORD_FIELDS = ["patient_id", "age", "sex", "vitals", "labs", "history", "symptoms"]


def _pick_patient_id_from_query(query: str) -> str:
    normalized = query.lower()
    if re.search(r"alpha|high risk|chest|hypertension", normalized):
        return "patient_alpha"
    if re.search(r"beta|palpitation|family history", normalized):
        return "patient_beta"
    return "patient_alpha"


def get_mock_cardiovascular_context(query: str) -> dict[str, Any]:
    """Fetch a structured mock cardiovascular record and validation metadata."""
    patient_id = _pick_patient_id_from_query(query)
    record = dict(MOCK_CARDIO_RECORDS[patient_id])

    missing_fields = [field for field in REQUIRED_RECORD_FIELDS if field not in record]
    return {
        "patient_id": patient_id,
        "record": record,
        "missing_fields": missing_fields,
    }


@tool("cardio_mock_lookup")
def cardio_mock_lookup(query: str) -> dict[str, Any]:
    """Fetch structured mock cardiovascular patient context from a user query."""
    return get_mock_cardiovascular_context(query)
