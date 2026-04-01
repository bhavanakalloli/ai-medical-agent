"""Gemini model factory for LangGraph nodes."""

from __future__ import annotations

import os

from langchain_google_genai import ChatGoogleGenerativeAI
from utils.config import GOOGLE_API_KEY

DEFAULT_MODEL = "gemini-3.1-flash-lite-preview"

def build_gemini_model(model_name: str = DEFAULT_MODEL, temperature: float = 0.1) -> ChatGoogleGenerativeAI:
    """Create a Gemini chat model configured through GOOGLE_API_KEY."""
    if not os.getenv("GOOGLE_API_KEY"):
        raise RuntimeError("Missing GOOGLE_API_KEY environment variable.")

    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        convert_system_message_to_human=True,
    )
