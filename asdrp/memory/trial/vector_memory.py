#############################################################################
# File: vector_memory.py
#
# Description:
#   Implements the raw vector memory baseline. It stores each adapter-produced
#   source entry without summarizing or restructuring its text.
#
#   - Preserves source text, ID, order, type, timestamp, and dataset
#     metadata.
#   - Stores records through the shared batched embedding and Qdrant
#     path.
#   - Combines dense Qdrant search with a lightweight local BM25 index.
#   - Fuses dense and lexical ranks before architecture-specific reranking.
#   - Adds a small recency tie-break for current-state questions.
#   - Expands and deduplicates neighboring source entries when the plan
#     requests context.
#
# Authors:
#   @author     Eric Vincent Fernandes
#
# Date:
#   Modified:   August 8, 2026 (Eric Vincent Fernandes)
#############################################################################

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseQdrantMemoryBlock
from asdrp.memory.hybrid_retrieval import HybridTextRetriever


class VectorMemoryBlock(BaseQdrantMemoryBlock):
    # Store the runner's source entries without changing their text.
    #
    # This is intentionally the least transformative architecture. Sessions stay
    # sessions, documents stay documents, facts stay facts, and chapter-aware
    # book segments stay chapter-aware segments. Storing summaries here would
    # erase the clean vector-memory baseline and confound comparisons against the
    # condensed and RAPTOR architectures.

    memory_name = "vector"  # Prefix for this task's Qdrant collection.

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._hybrid_retriever = HybridTextRetriever()

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Embed and store every source entry exactly once.

        self._hybrid_retriever.clear()  # Supports a clean rebuild if reused.

        # Keep the source text, order, type, ID, time, and all extra metadata.
        records = [
            {
                "id": entry.entry_id,
                "text": entry.text,
                "ordinal": entry.ordinal,
                "entry_kind": entry.metadata.get("entry_kind", "context"),
                "source_id": entry.metadata.get("source_id", entry.entry_id),
                "timestamp": entry.metadata.get("timestamp"),
                **entry.metadata,
            }
            for entry in entries
            if entry.text.strip()
        ]
        await self._store_text_records(records)
        self._hybrid_retriever.build(
            records,
            memory_id_for_record=self._qdrant_point_id,
        )

    async def close(self) -> None:
        # Release Qdrant state and the task-local lexical index.

        try:
            await super().close()
        finally:
            self._hybrid_retriever.clear()

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Use hybrid retrieval, optional recency bias, and adjacent expansion.

        if plan.mode in {"all", "global"}:
            return await self._retrieve_all()

        if plan.top_k <= 0:
            return []

        multiplier = max(1, int(plan.candidate_multiplier))
        candidate_limit = max(16, plan.top_k, plan.top_k * multiplier)
        lexical_limit = max(candidate_limit, plan.top_k * 3)
        candidates = await self._hybrid_retriever.search(
            query,
            self._query_similar(query, limit=candidate_limit),
            lexical_limit=lexical_limit,
        )
        if not candidates:
            return []

        max_ordinal = max(item.ordinal for item in candidates) or 1  # Recency scale.
        if plan.prefer_latest:
            # Conflict-resolution tasks define later statements as authoritative.
            # A small recency term breaks ties without overwhelming semantics.
            candidates.sort(
                key=lambda item: item.score + 0.08 * (item.ordinal / max_ordinal),
                reverse=True,
            )
        else:
            candidates.sort(key=lambda item: item.score, reverse=True)

        seeds = candidates[: plan.top_k]  # Best matches before neighbor expansion.
        if plan.neighbor_window <= 0:
            return seeds

        # Collect the source positions around every strong match.
        neighbor_ordinals = {
            seed.ordinal + delta
            for seed in seeds
            for delta in range(-plan.neighbor_window, plan.neighbor_window + 1)
        }
        neighbors = await self._retrieve_ordinals(neighbor_ordinals)  # Full records.
        score_by_id = {item.memory_id: item.score for item in seeds}  # Real scores.
        merged: dict[str, RetrievedMemory] = {}  # Removes overlapping neighbors.
        for item in neighbors:
            merged[item.memory_id] = RetrievedMemory(  # One copy of each neighbor.
                memory_id=item.memory_id,
                text=item.text,
                score=score_by_id.get(item.memory_id, 0.0),
                ordinal=item.ordinal,
                metadata=item.metadata,
            )
        # Chronological ordering makes adjacent book/session evidence readable.
        return sorted(merged.values(), key=lambda item: (item.ordinal, -item.score))
