#############################################################################
# File: hybrid_retrieval.py
#
# Description:
#   Provides a lightweight local lexical index for text-based memory blocks.
#   It combines standard BM25 with existing dense Qdrant results without
#   adding model, embedding, or network requests.
#
#   - Indexes only the records already retained by each memory architecture.
#   - Uses compact in-memory postings and standard BM25 scoring.
#   - Runs lexical search beside the existing dense query in a worker thread.
#   - Fuses dense and lexical ranks without comparing incompatible raw scores.
#   - Preserves the original stored text, metadata, IDs, and source order.
#############################################################################

from __future__ import annotations

import asyncio
import heapq
import math
import re
from collections import Counter, defaultdict
from collections.abc import Awaitable, Callable, Sequence
from typing import Any

from asdrp.eval_schemas import RetrievedMemory


# Keep exact identifiers and apostrophes while avoiding expensive tokenization.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", re.IGNORECASE)

# Common question words add little value to sparse retrieval.
_STOPWORDS = frozenset(
    """
        a an and are as at be been but by can did do does for from had has have he her
        hers him his how i if in is it its me my of on or our ours she that the their
        theirs them they this to was we were what where which who why will with would
        you your yours
    """.split()
)


class HybridTextRetriever:
    # Add a small BM25 index beside an architecture's existing dense retriever.

    def __init__(self) -> None:
        self._memories_by_id: dict[str, RetrievedMemory] = {}
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_lengths: dict[str, int] = {}
        self._doc_count = 0
        self._avg_doc_length = 1.0

    def clear(self) -> None:
        # Drop all task-local lexical state.

        self._memories_by_id.clear()
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_count = 0
        self._avg_doc_length = 1.0

    def build(
        self,
        records: Sequence[dict[str, Any]],
        *,
        memory_id_for_record: Callable[[str], str],
    ) -> None:
        # Index exactly the records that the architecture stores in Qdrant.

        self.clear()
        total_length = 0

        for record in records:
            text = str(record.get("text") or "")
            if not text.strip():
                continue

            record_id = str(record["id"])
            memory_id = memory_id_for_record(record_id)
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"id", "text", "ordinal"}
            }
            metadata.setdefault("record_id", record_id)
            memory = RetrievedMemory(
                memory_id=memory_id,
                text=text,
                score=0.0,
                ordinal=int(record.get("ordinal", 0)),
                metadata=metadata,
            )
            self._memories_by_id[memory_id] = memory

            terms = self._tokenize(text, remove_stopwords=True)
            if not terms:
                terms = self._tokenize(text, remove_stopwords=False)
            if not terms:
                continue

            counts = Counter(terms)
            length = len(terms)
            self._doc_lengths[memory_id] = length
            self._doc_count += 1
            total_length += length

            for term, frequency in counts.items():
                self._postings[term][memory_id] = frequency

        if self._doc_count:
            self._avg_doc_length = total_length / self._doc_count

    async def search(
        self,
        query: str,
        dense_search: Awaitable[list[RetrievedMemory]],
        *,
        lexical_limit: int,
    ) -> list[RetrievedMemory]:
        # Run Qdrant and local BM25 together, then fuse their rankings.

        dense, lexical = await asyncio.gather(
            dense_search,
            asyncio.to_thread(self._lexical_search, query, limit=lexical_limit),
        )
        if not dense and not lexical:
            return []
        return self._fuse(dense=dense, lexical=lexical)

    def _lexical_search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[tuple[RetrievedMemory, float]]:
        # Run standard BM25 over the locally retained text records.

        if limit <= 0 or not self._doc_count:
            return []

        query_terms = self._tokenize(query, remove_stopwords=True)
        if not query_terms:
            query_terms = self._tokenize(query, remove_stopwords=False)
        if not query_terms:
            return []

        k1 = 1.2  # Standard BM25 term-frequency saturation.
        b = 0.75  # Standard BM25 document-length normalization.
        scores: dict[str, float] = defaultdict(float)

        for term in set(query_terms):
            postings = self._postings.get(term)
            if not postings:
                continue

            document_frequency = len(postings)
            idf = math.log(
                1.0
                + (self._doc_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            for memory_id, frequency in postings.items():
                document_length = self._doc_lengths.get(memory_id, 1)
                denominator = frequency + k1 * (
                    1.0 - b + b * document_length / self._avg_doc_length
                )
                scores[memory_id] += idf * (frequency * (k1 + 1.0) / denominator)

        if not scores:
            return []

        ranked_ids = heapq.nlargest(
            limit,
            scores,
            key=lambda memory_id: (
                scores[memory_id],
                self._memories_by_id[memory_id].ordinal,
            ),
        )
        return [
            (self._memories_by_id[memory_id], scores[memory_id])
            for memory_id in ranked_ids
        ]

    def _fuse(
        self,
        *,
        dense: list[RetrievedMemory],
        lexical: list[tuple[RetrievedMemory, float]],
    ) -> list[RetrievedMemory]:
        # Fuse ranks because cosine and BM25 scores use different scales.

        fused_scores: dict[str, float] = defaultdict(float)
        chosen: dict[str, RetrievedMemory] = {}

        dense_count = max(1, len(dense))
        for rank, memory in enumerate(dense, start=1):
            local = self._memories_by_id.get(memory.memory_id, memory)
            chosen[memory.memory_id] = local
            dense_rank_score = 1.0 - (rank - 1) / dense_count
            fused_scores[memory.memory_id] += 0.60 * dense_rank_score

        lexical_count = max(1, len(lexical))
        max_lexical_score = max((score for _, score in lexical), default=1.0) or 1.0
        for rank, (memory, lexical_score) in enumerate(lexical, start=1):
            chosen[memory.memory_id] = memory
            lexical_rank_score = 1.0 - (rank - 1) / lexical_count
            fused_scores[memory.memory_id] += 0.40 * lexical_rank_score
            fused_scores[memory.memory_id] += 0.12 * (lexical_score / max_lexical_score)

        ranked = sorted(
            chosen.values(),
            key=lambda memory: (
                fused_scores[memory.memory_id],
                memory.ordinal,
            ),
            reverse=True,
        )
        return [
            RetrievedMemory(
                memory_id=memory.memory_id,
                text=memory.text,
                score=float(fused_scores[memory.memory_id]),
                ordinal=memory.ordinal,
                metadata=memory.metadata,
            )
            for memory in ranked
        ]

    @staticmethod
    def _tokenize(text: str, *, remove_stopwords: bool) -> list[str]:
        # Use the same cheap lexical token shape as episodic retrieval.

        terms = _TOKEN_RE.findall(text.casefold())
        if not remove_stopwords:
            return terms
        return [term for term in terms if term not in _STOPWORDS]
