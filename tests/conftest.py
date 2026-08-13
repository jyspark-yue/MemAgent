#############################################################################
# File: conftest.py
#
# Description:
#   Provides deterministic fixtures and fake services shared by the test
#   suite. These helpers keep unit tests local, repeatable, and free of
#   API or Qdrant requirements.
#
#   - Adds the project root to the import path when the package is not
#     installed.
#   - Defines a word-based counter for exact chunking tests.
#   - Defines stable hash-based vectors and scripted model responses.
#   - Records fake embedding and completion calls for later checks.
#   - Replaces the Qdrant network client during local memory tests.
#   - Provides source-entry and LongMemEval fixtures in real dataset
#     shapes.
#############################################################################


from __future__ import annotations

import hashlib
import importlib
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pytest

# Add the project folder when tests run without installing the package.
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class WordCounter:
    # Deterministic counter that makes boundary tests cheap and exact.

    def count(self, text: str) -> int:
        # Treat each whitespace-separated word as one test token.
        return len(text.split())

    def count_many(self, texts: Sequence[str]) -> list[int]:
        # Keep each count paired with its input string.
        return [self.count(text) for text in texts]

    def split(self, text: str, max_tokens: int) -> list[str]:
        # Losslessly split text using deterministic whitespace tokens.
        if max_tokens <= 0:
            raise ValueError(f"max_tokens must be positive, got {max_tokens}")

        words = text.split()  # Exact units used by count().
        return [
            " ".join(words[start : start + max_tokens])
            for start in range(0, len(words), max_tokens)
        ]


class FakeRuntime:
    # Zero-credit runtime with deterministic embeddings and scripted JSON LLM output.

    def __init__(self, responses: Sequence[Any] | None = None, dimension: int = 8):
        self.responses = list(responses or [])  # Scripted model responses in order.
        self.dimension = dimension  # Width of each fake vector.
        self.llm_counter = WordCounter()  # Cheap prompt and chunk counter.
        self.embed_calls: list[tuple[str, list[str]]] = []  # Calls made by tests.
        self.complete_calls: list[tuple[str, str]] = []  # Model prompts made by tests.

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()  # Stable source bytes.
        return [((digest[i] / 255.0) * 2 - 1) for i in range(self.dimension)]

    async def embed_many(self, texts: Sequence[str], *, phase: str):
        values = list(texts)  # Freeze the input before recording it.
        self.embed_calls.append((phase, values))
        return [self._vector(text) for text in values]

    async def embed_one(self, text: str, *, phase: str):
        return (await self.embed_many([text], phase=phase))[0]

    async def complete(self, prompt: str, *, phase: str = "memory", **_: Any):
        self.complete_calls.append((phase, prompt))
        response = self.responses.pop(0) if self.responses else {}  # Next script item.
        if isinstance(response, str):
            return response
        return json.dumps(response)

    async def chat(self, *args: Any, **kwargs: Any):
        return await self.complete(str(args), **kwargs)


class FakeQdrantClient:
    # Unit tests replace the network client because they test memory logic only.

    async def close(self):
        return None


@pytest.fixture
def counter():
    # Supply the exact word counter used by chunking tests.
    return WordCounter()


@pytest.fixture
def fake_runtime(monkeypatch):
    # Supply a runtime that never makes an API request.
    base_memory = importlib.import_module("asdrp.memory.BaseMemBlock")
    monkeypatch.setattr(
        base_memory,
        "AsyncQdrantClient",
        lambda **_kwargs: FakeQdrantClient(),
    )
    return FakeRuntime()


@pytest.fixture
def memory_entry_factory():
    # Make short source entries with stable IDs and metadata.
    from asdrp.eval_schemas import MemoryEntry

    def make(i: int, text: str | None = None, kind: str = "session"):
        return MemoryEntry(
            entry_id=f"e{i}",
            text=text or f"entry {i} alpha beta gamma",
            ordinal=i,
            metadata={"entry_kind": kind, "source_id": f"s{i}"},
        )

    return make


@pytest.fixture
def longmemeval_row():
    # A small update example that follows the real LongMemEval field shape.
    return {
        "question_id": "lme-q1",
        "question": "What changed?",
        "answer": ["new value"],
        "question_type": "knowledge-update",
        "question_date": "2025/01/03 (Fri) 12:00",
        "haystack_session_ids": ["s-unnecessary", "s-new"],
        "haystack_dates": ["2025/01/01 (Wed) 10:00", "2025/01/02 (Thu) 10:00"],
        "haystack_sessions": [
            [
                {"role": "user", "content": "My code is OLD."},
                {"role": "assistant", "content": "Noted."},
            ],
            [
                {"role": "user", "content": "My code is NEW.<|endoftext|>"},
                {"role": "assistant", "content": "Updated."},
            ],
        ],
        "answer_session_ids": ["s-new"],
    }
