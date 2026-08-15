#############################################################################
# File: llm_client.py
#
# Description:
#   Provides the dataset generator's minimal OpenAI JSON-generation client.
#
#   - Validates provider selection and OPENAI_API_KEY before live generation.
#   - Uses Chat Completions JSON-object mode with configurable model and temperature.
#   - Normalizes model text through the shared JSON extraction helper.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

import os
from typing import Any

from .io_utils import extract_json_object


class LLMClient:
    def __init__(self, provider: str, model: str, temperature: float = 0.2):
        self.provider = provider
        self.model = model
        self.temperature = temperature
        if provider != "openai":
            raise ValueError(
                "This scaffold currently implements provider='openai'. Add other providers here if needed."
            )
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError(
                "OPENAI_API_KEY is not set. Use --dry-run for a no-API smoke test."
            )
        try:
            from openai import OpenAI
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Install dependencies first: pip install -r requirements.txt"
            ) from exc
        self.client = OpenAI()

    def generate_json(
        self, messages: list[dict[str, str]], schema: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        # json_object is robust and works across many OpenAI chat models.
        # If you want strict Structured Outputs, replace this with client.responses.parse.
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            response_format={"type": "json_object"},
        )
        content = response.choices[0].message.content or "{}"
        return extract_json_object(content)
