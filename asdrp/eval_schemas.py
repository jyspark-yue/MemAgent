#############################################################################
# File: eval_schemas.py
#
# Description:
#   Defines the small records shared by dataset adapters, memory blocks,
#   agents, and the evaluator. These types keep storage, retrieval,
#   questions, and pricing consistent across architectures.
#
#   - Defines source entries, retrieved memories, questions, retrieval
#     plans, and evaluation tasks.
#   - Keeps records immutable where later mutation would be unsafe.
#   - Stores standard and long-context LLM rates for input, cache reads,
#     cache writes, and output.
#   - Selects the correct rate set from the request's total input
#     tokens.
#   - Stores the embedding input rate used by the usage ledger.
#############################################################################

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


@dataclass(slots=True, frozen=True)
class MemoryEntry:
    # A dataset adapter creates one entry for each complete unit of source text.
    # This keeps dataset splitting out of the memory architectures.

    entry_id: str  # Stable ID for this source unit.
    text: str  # Full text that the memory block can store or transform.
    ordinal: int  # Source order used for time and neighbor lookup.
    metadata: dict[str, Any] = field(default_factory=dict)  # Dataset details.


@dataclass(slots=True, frozen=True)
class RetrievedMemory:
    # A memory block returns these records for one question.

    memory_id: str  # ID assigned by the memory store.
    text: str  # Evidence passed to the agent.
    score: float  # Retrieval score, where a larger value ranks first.
    ordinal: int  # Original source order.
    metadata: dict[str, Any] = field(default_factory=dict)  # Stored structure.


@dataclass(slots=True, frozen=True)
class EvaluationQuestion:
    # Each question runs on its own, even when a task holds several questions.

    question_id: str  # Stable ID used for resume support and usage totals.
    text: str  # Question sent to the agent.
    answers: list[str]  # Reference answers kept in the output file.
    metadata: dict[str, Any] = field(default_factory=dict)  # Benchmark labels.


@dataclass(slots=True, frozen=True)
class RetrievalPlan:
    # The dataset adapter sets a safe default plan for each task.

    mode: Literal["top_k", "all", "global"] = "top_k"  # Retrieval scope.
    top_k: int = 8  # Final number of high-ranked records.
    candidate_multiplier: int = 4  # Extra records checked before filtering.
    neighbor_window: int = 0  # Entries kept on each side of a match.
    graph_hops: int = 2  # Maximum number of graph links to follow.
    max_context_tokens: int = 96_000  # Evidence budget for the final prompt.
    prefer_latest: bool = False  # Whether newer facts override older facts.


@dataclass(slots=True, frozen=True)
class EvaluationTask:
    # A task owns one isolated context and all questions that share it.

    task_id: str  # Stable context ID.
    dataset: str  # Normalized benchmark name.
    source: str  # Source family inside the benchmark.
    entries: list[MemoryEntry]  # Context loaded into memory once.
    questions: list[EvaluationQuestion]  # Questions asked after memory loads.
    retrieval: RetrievalPlan  # Dataset-specific retrieval defaults.
    metadata: dict[str, Any] = field(default_factory=dict)  # Task details.


@dataclass(frozen=True, slots=True)
class Pricing:
    # Prices use US dollars per one million tokens.

    llm_input_per_million: float  # Standard uncached input rate.
    llm_cached_input_per_million: float  # Standard cached input rate.
    llm_output_per_million: float  # Standard output rate.
    embedding_input_per_million: float  # Embedding input rate.

    llm_cache_write_per_million: float | None = None  # Standard cache-write rate.
    llm_long_input_per_million: float | None = None  # Long uncached input rate.
    llm_long_cached_input_per_million: float | None = None  # Long cached input rate.
    llm_long_cache_write_per_million: float | None = None  # Long cache-write rate.
    llm_long_output_per_million: float | None = None  # Long output rate.
    llm_long_context_threshold: int = 272_000  # Requests above this use long rates.

    def uses_long_context(self, input_tokens: int) -> bool:
        # OpenAI applies long-context pricing to the full request above the cutoff.
        return input_tokens > self.llm_long_context_threshold

    def llm_rates(self, input_tokens: int) -> tuple[float, float, float, float]:
        # Use long prices only when the complete long-rate set is configured.

        if (
            self.uses_long_context(input_tokens)
            and self.llm_long_input_per_million is not None
            and self.llm_long_cached_input_per_million is not None
            and self.llm_long_output_per_million is not None
        ):
            return (
                self.llm_long_input_per_million,
                self.llm_long_cached_input_per_million,
                (
                    self.llm_long_cache_write_per_million
                    if self.llm_long_cache_write_per_million is not None
                    else self.llm_long_input_per_million
                ),
                self.llm_long_output_per_million,
            )

        # Unknown or partly configured models keep standard pricing as a safe fallback.
        return (
            self.llm_input_per_million,
            self.llm_cached_input_per_million,
            (
                self.llm_cache_write_per_million
                if self.llm_cache_write_per_million is not None
                else self.llm_input_per_million
            ),
            self.llm_output_per_million,
        )
