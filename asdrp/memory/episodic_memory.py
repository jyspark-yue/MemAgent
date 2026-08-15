#############################################################################
# File: episodic_memory.py
#
# Description:
#   Builds fast episodic memory from chronological source entries while keeping
#   exact source evidence available for retrieval. Local source-derived episodes
#   are the default fast path, with the original structured LLM extraction kept
#   as an optional higher-cost mode.
#
#   - Uses exact source-derived episodes by default, eliminating memory-build LLM
#     calls while retaining dense and lexical retrieval over the original facts.
#   - Splits only unusually large source entries so embedding records stay bounded
#     without paying for full tokenizer passes during normal local ingestion.
#   - Keeps structured LLM extraction as an opt-in compatibility mode with bounded
#     concurrent packs, provenance validation, and exact-source fallbacks.
#   - Removes duplicate extracted events only in LLM mode while preserving repeated
#     local source text losslessly.
#   - Stores compact local payloads by default and keeps full structured event
#     metadata only when structured LLM extraction is enabled.
#   - Keeps exact source entries locally for global retrieval without adding extra
#     Qdrant records; local episodes carry the sparse index in the default mode.
#   - Combines dense Qdrant search with local BM25-style lexical retrieval for
#     names, numbers, product IDs, dates, and other exact benchmark details.
#   - Uses rank fusion instead of mixing incomparable dense and lexical scores.
#   - Applies recency only to current/latest queries rather than every temporal
#     question.
#   - Expands chronological neighbors from local memory without another Qdrant
#     request and tightly caps low-value expansion before it can flood context.
#   - Returns exact source entries directly for global retrieval so recursive
#     reduction never depends on lossy extraction alone.
#   - Prints concise build-stage progress so long embedding/storage calls are easy
#     to distinguish from a hung process.
#############################################################################

from __future__ import annotations

import asyncio
import heapq
import math
import re
import time
import uuid
from collections import Counter, defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseQdrantMemoryBlock
from asdrp.runtime import parse_json_payload


# Keep lexical matching simple and cheap while preserving numbers and identifiers.
_TOKEN_RE = re.compile(r"[a-z0-9]+(?:['’-][a-z0-9]+)*", re.IGNORECASE)

_TEMPORAL_RE = re.compile(
    r"\b(when|before|after|then|later|earlier|next|previous|first|last|recent|latest|"
    r"newest|current|currently|chronolog\w*|changed|updated|used to|at the time)\b",
    re.IGNORECASE,
)
_LATEST_RE = re.compile(
    r"\b(latest|newest|most recent|current|currently|now|still|as of now|today)\b",
    re.IGNORECASE,
)
_HISTORICAL_RE = re.compile(
    r"\b(before|earlier|previous|previously|originally|first|used to|at the time|"
    r"history|historical)\b",
    re.IGNORECASE,
)

# Common question words add little retrieval value and can swamp useful exact terms.
_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "hers",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "is",
        "it",
        "its",
        "me",
        "my",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "she",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "they",
        "this",
        "to",
        "was",
        "we",
        "were",
        "what",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
        "yours",
    }
)


@dataclass(slots=True)
class _Episode:
    # One event and the details needed to place it in context.

    event: str  # What happened or was stated.
    participants: list[str]  # People or entities involved.
    time: str  # Explicit or relative time.
    location: str  # Place where the event happened.
    outcome: str  # Result or later state.
    causal_context: str  # Cause or event that led to it.
    source_id: str  # ID of the supporting source entry.
    source_ordinal: int  # Position of the supporting entry.


class EpisodicMemoryBlock(BaseQdrantMemoryBlock):
    # Store the source as separate events with exact-source retrieval as a safety net.
    #
    # Dense episode vectors handle semantic matches. A small local lexical index keeps
    # exact names, dates, numbers, IDs, and wording available without a second model or
    # embedding call. Raw source entries remain local so extraction can improve search
    # structure without becoming a single point of failure for recall.

    memory_name = "episodic"  # Prefix for this task's Qdrant collection.

    def __init__(
        self,
        *,
        extraction_input_tokens: int = 12_000,
        extraction_concurrency: int = 8,
        use_llm_extraction: bool = False,
        local_episode_max_chars: int = 12_000,
        **kwargs: Any,
    ) -> None:
        if extraction_input_tokens < 1:
            raise ValueError("extraction_input_tokens must be at least 1")
        if extraction_concurrency < 1:
            raise ValueError("extraction_concurrency must be at least 1")
        if local_episode_max_chars < 1_000:
            raise ValueError("local_episode_max_chars must be at least 1000")

        super().__init__(**kwargs)

        self._extraction_input_tokens = extraction_input_tokens  # One prompt limit.
        self._extraction_semaphore = asyncio.Semaphore(extraction_concurrency)
        self._use_llm_extraction = use_llm_extraction
        self._local_episode_max_chars = local_episode_max_chars

        # Local copies make lexical retrieval and neighbor expansion network-free.
        self._memories_by_id: dict[str, RetrievedMemory] = {}
        self._episode_ids_by_ordinal: dict[int, list[str]] = defaultdict(list)
        self._source_memories: list[RetrievedMemory] = []

        # The postings map stores each term's frequency in only the documents using it.
        self._postings: dict[str, dict[str, int]] = defaultdict(dict)
        self._doc_lengths: dict[str, int] = {}
        self._doc_count = 0
        self._avg_doc_length = 1.0
        self._max_ordinal = 1

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Extract event records while keeping a local exact copy of every source entry.

        started = time.perf_counter()
        self._reset_local_index()  # Supports a clean rebuild if this object is reused.
        print(
            f"[episodic] build start | entries={len(entries):,} | "
            f"extraction={'llm' if self._use_llm_extraction else 'local'}",
            flush=True,
        )

        if self._use_llm_extraction:
            # Facts already contain their durable memory and do not need model extraction.
            direct_episodes = [
                self._fallback_episode(entry)
                for entry in entries
                if entry.metadata.get("entry_kind") == "fact"
            ]
            extraction_entries = [
                entry for entry in entries if entry.metadata.get("entry_kind") != "fact"
            ]
            packs = self._pack_entries(extraction_entries)
            print(
                f"[episodic] extracting | packs={len(packs):,}",
                flush=True,
            )
            extraction_results = await asyncio.gather(
                *(self._extract_pack(pack) for pack in packs),
                return_exceptions=True,
            )

            # One independently cancelled pack should not cancel every task in the run.
            extracted_packs: list[list[_Episode]] = []
            cancelled_packs = 0
            for pack, result in zip(packs, extraction_results, strict=True):
                if isinstance(result, asyncio.CancelledError):
                    cancelled_packs += 1
                    extracted_packs.append(
                        [self._fallback_episode(entry) for entry in pack]
                    )
                    continue
                if isinstance(result, BaseException):
                    raise result
                extracted_packs.append(result)

            if cancelled_packs:
                print(
                    f"[episodic] recovered {cancelled_packs:,} cancelled extraction "
                    f"pack{'s' if cancelled_packs != 1 else ''} with exact-source fallback",
                    flush=True,
                )
            extracted_episodes = [
                episode for pack in extracted_packs for episode in pack
            ]
            episodes = self._dedupe_episodes(direct_episodes + extracted_episodes)
        else:
            # Fast path: source entries are already chronological episodic observations.
            # Keep their wording losslessly and split only unusually large entries so one
            # embedding record never becomes needlessly huge.
            episodes = [
                episode for entry in entries for episode in self._local_episodes(entry)
            ]

        records: list[dict[str, Any]] = []  # Qdrant episode records to embed and store.
        record_occurrences: dict[tuple[str, str], int] = defaultdict(int)
        for episode in episodes:
            if not episode.event.strip():
                continue

            record_key = (episode.source_id, episode.event)
            occurrence = record_occurrences[record_key]
            record_occurrences[record_key] += 1
            id_parts = [episode.source_id, episode.event]
            if occurrence:
                id_parts.append(f"duplicate:{occurrence}")
            record_id = self._stable_memory_id("episode", *id_parts)
            record: dict[str, Any] = {
                "id": record_id,
                "text": self._episode_text(episode),
                "ordinal": episode.source_ordinal,
                "source_id": episode.source_id,
                "entry_kind": "episode",
            }
            if self._use_llm_extraction:
                # Structured extraction needs these fields for answer-time reconstruction.
                record.update(
                    {
                        "event": episode.event,
                        "participants": episode.participants,
                        "time": episode.time,
                        "location": episode.location,
                        "outcome": episode.outcome,
                        "causal_context": episode.causal_context,
                    }
                )
            records.append(record)

        # Let embedding/storage issue its first network request, then build the local sparse
        # index while that request is in flight. This overlaps useful work without mutating
        # memory state from a background thread.
        print(f"[episodic] storing | episodes={len(records):,}", flush=True)
        store_task = asyncio.create_task(self._store_text_records(records))
        await asyncio.sleep(0)
        try:
            self._build_local_index(entries, records)
            await store_task
        except BaseException:
            store_task.cancel()
            await asyncio.gather(store_task, return_exceptions=True)
            raise
        print(
            f"[episodic] build done | episodes={len(records):,} | "
            f"elapsed={time.perf_counter() - started:.1f}s",
            flush=True,
        )

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Retrieve a small, high-recall timeline with one dense query plus local ranking.

        if plan.mode in {"all", "global"}:
            # Exact source entries are both smaller and safer than replaying every derived
            # episode through global reduction. Fall back to Qdrant only if needed.
            if self._source_memories:
                return [
                    self._copy_memory(item, score=1.0) for item in self._source_memories
                ]
            return await self._retrieve_all()

        if plan.top_k <= 0:
            return []

        # A wider candidate pool is cheap in Qdrant and lets rank fusion recover near-miss
        # semantic results without passing that extra noise to the answer model.
        multiplier = max(1, int(plan.candidate_multiplier))
        candidate_limit = max(16, plan.top_k, plan.top_k * multiplier)
        lexical_limit = max(candidate_limit, plan.top_k * 3)

        # Qdrant/network work and local BM25 can run at the same time.
        dense, lexical = await asyncio.gather(
            self._query_similar(query, limit=candidate_limit),
            asyncio.to_thread(self._lexical_search, query, limit=lexical_limit),
        )

        if not dense and not lexical:
            return []

        fused = self._fuse_candidates(
            dense=dense,
            lexical=lexical,
            query=query,
            prefer_latest=plan.prefer_latest,
        )
        seeds = fused[: plan.top_k]  # Strong matches before chronology expansion.

        temporal_query = self._is_temporal_query(query)
        window = max(plan.neighbor_window, 1 if temporal_query else 0)
        if window <= 0:
            return sorted(seeds, key=lambda item: (item.ordinal, -item.score))

        return self._expand_neighbors(seeds, window=window, top_k=plan.top_k)

    async def close(self) -> None:
        # Release Qdrant state and drop the task-local lexical index at the same time.

        try:
            await super().close()
        finally:
            self._reset_local_index()

    def _pack_entries(self, entries: Sequence[MemoryEntry]) -> list[list[MemoryEntry]]:
        # Pack complete source entries while respecting the extraction input budget.

        packs: list[list[MemoryEntry]] = []  # Finished extraction packs.
        current: list[MemoryEntry] = []  # Entries in the pack being filled.
        tokens = 500  # Leaves room for the extraction instructions and JSON schema.

        for entry in entries:
            size = self.runtime.llm_counter.count(entry.text) + 32  # Text and label.
            if current and tokens + size > self._extraction_input_tokens:
                packs.append(current)
                current = []
                tokens = 500
            current.append(entry)
            tokens += size

        if current:
            packs.append(current)
        return packs

    async def _extract_pack(self, entries: list[MemoryEntry]) -> list[_Episode]:
        # Extract durable event units and guarantee that every source stays represented.

        if not entries:
            return []

        # Include source IDs and ordinals so extracted events can keep exact provenance.
        source = "\n\n".join(
            f"[ENTRY id={entry.entry_id} ordinal={entry.ordinal}]\n{entry.text}"
            for entry in entries
        )
        prompt = f"""
            Convert the benchmark context into episodic long-term memories. The context is
            the only source of truth.
            
            Extract every durable fact that could matter to a later question, including:
            - events and interactions
            - explicit facts, names, numbers, dates, quantities, IDs, and locations
            - preferences, constraints, purchases, plans, requests, and recommendations
            - decisions, corrections, conflicts, updates, reversals, and resulting states
            - causes or dependencies when the source states them
            
            Keep each event self-contained and preserve exact details, negation, ownership, and
            who said or did what. Do not merge separate occasions. Do not infer missing facts or
            add world knowledge. Skip only pure conversational filler with no durable information.
            
            Return JSON only: an array of objects with keys:
            - event: concise self-contained event or fact
            - participants: array of named people/entities involved
            - time: explicit or relative time, otherwise empty string
            - location: location, otherwise empty string
            - outcome: result or resulting state, otherwise empty string
            - causal_context: stated cause/dependency/preceding event, otherwise empty string
            - source_id: supporting ENTRY id
            - source_ordinal: supporting ENTRY ordinal as an integer
            
            CONTEXT:
            {source}
        """.strip()

        async with self._extraction_semaphore:
            response = await self.runtime.complete(prompt, phase="memory")

        # Invalid JSON should reduce extraction quality, not crash the full benchmark task.
        try:
            payload = parse_json_payload(response)
        except (TypeError, ValueError):
            return [self._fallback_episode(entry) for entry in entries]

        if not isinstance(payload, list):
            return [self._fallback_episode(entry) for entry in entries]

        known_by_id = {entry.entry_id: entry for entry in entries}
        known_by_ordinal = {entry.ordinal: entry for entry in entries}
        entry_terms = {
            entry.entry_id: set(self._tokenize(entry.text, remove_stopwords=True))
            for entry in entries
        }

        episodes: list[_Episode] = []  # Valid events returned by the extraction model.
        covered_sources: set[str] = set()  # Entries that received at least one event.

        for item in payload:
            if not isinstance(item, dict):
                continue

            event = str(item.get("event") or "").strip()
            if not event:
                continue

            source_entry = self._resolve_source_entry(
                item=item,
                event=event,
                entries=entries,
                known_by_id=known_by_id,
                known_by_ordinal=known_by_ordinal,
                entry_terms=entry_terms,
            )
            covered_sources.add(source_entry.entry_id)

            participants = item.get("participants") or []
            if not isinstance(participants, list):
                participants = [str(participants)]

            episodes.append(
                _Episode(
                    event=event,
                    participants=[
                        str(value).strip()
                        for value in participants
                        if str(value).strip()
                    ],
                    time=str(item.get("time") or "").strip(),
                    location=str(item.get("location") or "").strip(),
                    outcome=str(item.get("outcome") or "").strip(),
                    causal_context=str(item.get("causal_context") or "").strip(),
                    source_id=source_entry.entry_id,
                    source_ordinal=source_entry.ordinal,
                )
            )

        # Extraction is allowed to compress, but it is never allowed to erase a source.
        for entry in entries:
            if entry.entry_id not in covered_sources:
                episodes.append(self._fallback_episode(entry))

        return episodes

    def _resolve_source_entry(
        self,
        *,
        item: dict[str, Any],
        event: str,
        entries: list[MemoryEntry],
        known_by_id: dict[str, MemoryEntry],
        known_by_ordinal: dict[int, MemoryEntry],
        entry_terms: dict[str, set[str]],
    ) -> MemoryEntry:
        # Keep model provenance when valid, then recover locally when it is not.

        requested_source_id = str(item.get("source_id") or "").strip()
        if requested_source_id in known_by_id:
            return known_by_id[requested_source_id]

        try:
            requested_ordinal = int(item.get("source_ordinal"))
        except (TypeError, ValueError):
            requested_ordinal = -1
        if requested_ordinal in known_by_ordinal:
            return known_by_ordinal[requested_ordinal]

        # Last resort: match the extracted event to the source sharing the most useful
        # lexical terms. This is more accurate than assigning every bad ID to entry zero.
        event_terms = set(self._tokenize(event, remove_stopwords=True))
        if event_terms:
            return max(
                entries,
                key=lambda entry: len(event_terms & entry_terms[entry.entry_id]),
            )
        return entries[0]

    def _build_local_index(
        self,
        entries: Sequence[MemoryEntry],
        records: Sequence[dict[str, Any]],
    ) -> None:
        # Build exact-source anchors, episode lookup tables, and BM25 postings once.

        # Local episodes already preserve every source word, so indexing the full source
        # again would nearly double sparse-index work and return duplicate candidates. LLM
        # extraction still indexes source anchors because its summaries may omit wording.
        exact_episode_sources: set[str] = set()
        if self._use_llm_extraction:
            entry_text_by_id = {
                entry.entry_id: " ".join(entry.text.casefold().split())
                for entry in entries
            }
            exact_episode_sources = {
                str(record.get("source_id") or "")
                for record in records
                if " ".join(str(record.get("event") or "").casefold().split())
                == entry_text_by_id.get(str(record.get("source_id") or ""), "")
            }

        for entry in entries:
            text = entry.text.strip()
            if not text:
                continue

            source_memory = RetrievedMemory(
                memory_id=self._stable_memory_id("source", entry.entry_id),
                text=text,
                score=0.0,
                ordinal=entry.ordinal,
                metadata={
                    "entry_kind": "source_episode",
                    "source_id": entry.entry_id,
                    "timestamp": self._entry_time(entry),
                },
            )
            self._source_memories.append(source_memory)
            self._memories_by_id[source_memory.memory_id] = source_memory
            if self._use_llm_extraction and entry.entry_id not in exact_episode_sources:
                self._index_memory(source_memory)
            self._max_ordinal = max(self._max_ordinal, entry.ordinal)

        for record in records:
            memory_id = str(record["id"])
            metadata = {
                key: value
                for key, value in record.items()
                if key not in {"id", "text", "ordinal"}
            }
            metadata.setdefault("record_id", memory_id)

            memory = RetrievedMemory(
                memory_id=memory_id,
                text=str(record["text"]),
                score=0.0,
                ordinal=int(record["ordinal"]),
                metadata=metadata,
            )
            self._memories_by_id[memory_id] = memory
            self._episode_ids_by_ordinal[memory.ordinal].append(memory_id)
            self._index_memory(memory)
            self._max_ordinal = max(self._max_ordinal, memory.ordinal)

        self._source_memories.sort(key=lambda item: (item.ordinal, item.memory_id))
        if self._doc_count:
            self._avg_doc_length = sum(self._doc_lengths.values()) / self._doc_count

    def _index_memory(self, memory: RetrievedMemory) -> None:
        # Add one local document to the sparse lexical index.

        terms = self._tokenize(memory.text, remove_stopwords=True)
        if not terms:
            terms = self._tokenize(memory.text, remove_stopwords=False)
        if not terms:
            return

        counts = Counter(terms)
        self._doc_lengths[memory.memory_id] = len(terms)
        self._doc_count += 1

        for term, frequency in counts.items():
            self._postings[term][memory.memory_id] = frequency

    def _lexical_search(
        self,
        query: str,
        *,
        limit: int,
    ) -> list[tuple[RetrievedMemory, float]]:
        # Run a compact BM25-style search over exact source text and episode text.

        if limit <= 0 or not self._doc_count:
            return []

        query_terms = self._tokenize(query, remove_stopwords=True)
        if not query_terms:
            query_terms = self._tokenize(query, remove_stopwords=False)
        if not query_terms:
            return []

        k1 = 1.2  # Standard BM25 term-frequency saturation.
        b = 0.75  # Standard BM25 length normalization.
        scores: dict[str, float] = defaultdict(float)
        matched_terms: dict[str, int] = defaultdict(int)
        unique_query_terms = set(query_terms)

        for term in unique_query_terms:
            postings = self._postings.get(term)
            if not postings:
                continue

            document_frequency = len(postings)
            idf = math.log(
                1.0
                + (self._doc_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )
            if any(character.isdigit() for character in term):
                idf *= (
                    1.20  # Exact numeric details are unusually valuable in benchmarks.
                )

            for memory_id, frequency in postings.items():
                document_length = self._doc_lengths.get(memory_id, 1)
                denominator = frequency + k1 * (
                    1.0 - b + b * document_length / self._avg_doc_length
                )
                scores[memory_id] += idf * (frequency * (k1 + 1.0) / denominator)
                matched_terms[memory_id] += 1

        if not scores:
            return []

        # Reward records covering several different query terms, not one repeated term.
        query_term_count = max(1, len(unique_query_terms))
        for memory_id, matched_count in matched_terms.items():
            scores[memory_id] += 0.35 * matched_count / query_term_count

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

    def _fuse_candidates(
        self,
        *,
        dense: list[RetrievedMemory],
        lexical: list[tuple[RetrievedMemory, float]],
        query: str,
        prefer_latest: bool,
    ) -> list[RetrievedMemory]:
        # Fuse rank positions because dense cosine and BM25 scores are not comparable.

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

        # Recency helps only when the question asks for the current/latest state. It is
        # deliberately not applied to generic temporal questions such as "before".
        if (
            prefer_latest or self._is_latest_query(query)
        ) and not self._is_historical_query(query):
            for memory_id, memory in chosen.items():
                recency = min(
                    1.0,
                    max(0, memory.ordinal) / max(1, self._max_ordinal),
                )
                fused_scores[memory_id] += 0.04 * recency

        ranked = sorted(
            chosen.values(),
            key=lambda memory: (
                fused_scores[memory.memory_id],
                memory.ordinal,
            ),
            reverse=True,
        )

        return [
            self._copy_memory(memory, score=fused_scores[memory.memory_id])
            for memory in ranked
        ]

    def _expand_neighbors(
        self,
        seeds: list[RetrievedMemory],
        *,
        window: int,
        top_k: int,
    ) -> list[RetrievedMemory]:
        # Add nearby event records locally, then cap expansion before it becomes noise.

        merged = {memory.memory_id: memory for memory in seeds}

        for seed in seeds:
            for delta in range(-window, window + 1):
                ordinal = seed.ordinal + delta
                if ordinal < 0:
                    continue

                # Same-source sibling episodes can matter even when delta is zero.
                distance = abs(delta)
                neighbor_fraction = 0.30 / (distance + 1)
                for memory_id in self._episode_ids_by_ordinal.get(ordinal, []):
                    if memory_id in merged:
                        continue
                    local = self._memories_by_id[memory_id]
                    merged[memory_id] = self._copy_memory(
                        local,
                        score=seed.score * neighbor_fraction,
                    )

        # Neighbor windows overlap heavily. Keep every seed, then retain only the most
        # useful surrounding events so chronology cannot consume the whole prompt.
        seed_ids = {memory.memory_id for memory in seeds}
        neighbors = [
            memory for memory_id, memory in merged.items() if memory_id not in seed_ids
        ]
        neighbors.sort(key=lambda item: (item.score, item.ordinal), reverse=True)

        max_total = max(len(seeds), min(40, max(top_k * 2, top_k + 6)))
        kept = seeds + neighbors[: max(0, max_total - len(seeds))]
        return sorted(
            kept, key=lambda item: (item.ordinal, -item.score, item.memory_id)
        )

    def _dedupe_episodes(self, episodes: Sequence[_Episode]) -> list[_Episode]:
        # Remove only exact normalized duplicates from the same source entry.

        seen: set[tuple[str, str]] = set()
        unique: list[_Episode] = []

        for episode in episodes:
            normalized_event = " ".join(episode.event.casefold().split())
            key = (episode.source_id, normalized_event)
            if not normalized_event or key in seen:
                continue
            seen.add(key)
            unique.append(episode)

        return unique

    @staticmethod
    def _fallback_episode(entry: MemoryEntry) -> _Episode:
        # Preserve the source exactly when extraction is unnecessary or incomplete.

        timestamp = (
            entry.metadata.get("timestamp")
            or entry.metadata.get("time")
            or entry.metadata.get("date")
            or ""
        )
        return _Episode(
            event=entry.text.strip(),
            participants=[],
            time=str(timestamp),
            location="",
            outcome="",
            causal_context="",
            source_id=entry.entry_id,
            source_ordinal=entry.ordinal,
        )

    def _local_episodes(self, entry: MemoryEntry) -> list[_Episode]:
        # Use the source itself as the episode and only chunk entries that are very large.

        text = entry.text.strip()
        if not text:
            return []
        if len(text) <= self._local_episode_max_chars:
            return [self._fallback_episode(entry)]

        timestamp = self._entry_time(entry)
        chunks: list[_Episode] = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = min(text_length, start + self._local_episode_max_chars)
            if end < text_length:
                search_start = start + self._local_episode_max_chars // 2
                split_at = max(
                    text.rfind("\n\n", search_start, end),
                    text.rfind("\n", search_start, end),
                    text.rfind(". ", search_start, end),
                )
                if split_at > start:
                    end = split_at + (
                        2 if text.startswith(("\n\n", ". "), split_at) else 1
                    )

            chunk = text[start:end].strip()
            if chunk:
                chunks.append(
                    _Episode(
                        event=chunk,
                        participants=[],
                        time=timestamp,
                        location="",
                        outcome="",
                        causal_context="",
                        source_id=entry.entry_id,
                        source_ordinal=entry.ordinal,
                    )
                )
            start = end
            while start < text_length and text[start].isspace():
                start += 1

        return chunks

    def _stable_memory_id(self, kind: str, *parts: str) -> str:
        # Stable UUIDs keep local and Qdrant memory IDs identical across retries.

        value = "\x1f".join(str(part) for part in parts)
        return str(
            uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{self.task_id}:episodic:{kind}:{value}",
            )
        )

    @staticmethod
    def _episode_text(episode: _Episode) -> str:
        # Build a compact search document; the agent reads structured metadata directly.

        fields = [episode.event]
        if episode.participants:
            fields.append(f"Participants: {', '.join(episode.participants)}")
        if episode.time:
            fields.append(f"Time: {episode.time}")
        if episode.location:
            fields.append(f"Location: {episode.location}")
        if episode.causal_context:
            fields.append(f"Cause: {episode.causal_context}")
        if episode.outcome:
            fields.append(f"Outcome: {episode.outcome}")
        return "\n".join(fields)

    @staticmethod
    def _entry_time(entry: MemoryEntry) -> str:
        # Use the first time-like field supplied by the dataset adapter.

        return str(
            entry.metadata.get("timestamp")
            or entry.metadata.get("time")
            or entry.metadata.get("date")
            or ""
        )

    @staticmethod
    def _copy_memory(memory: RetrievedMemory, *, score: float) -> RetrievedMemory:
        # RetrievedMemory is treated as immutable, so ranking creates a fresh record.

        return RetrievedMemory(
            memory_id=memory.memory_id,
            text=memory.text,
            score=float(score),
            ordinal=memory.ordinal,
            metadata=memory.metadata,
        )

    @staticmethod
    def _tokenize(text: str, *, remove_stopwords: bool) -> list[str]:
        # Lowercase lexical terms once and optionally drop low-information words.

        terms = _TOKEN_RE.findall(text.casefold())
        if not remove_stopwords:
            return terms
        return [term for term in terms if term not in _STOPWORDS]

    @staticmethod
    def _is_temporal_query(query: str) -> bool:
        # Detect order-sensitive wording without assuming that newer always means better.

        return bool(_TEMPORAL_RE.search(query))

    @staticmethod
    def _is_latest_query(query: str) -> bool:
        # Only these phrases justify a direct recency preference during retrieval.

        return bool(_LATEST_RE.search(query))

    @staticmethod
    def _is_historical_query(query: str) -> bool:
        # Historical questions need unnecessary states preserved instead of recency boosting.

        return bool(_HISTORICAL_RE.search(query))

    def _reset_local_index(self) -> None:
        # Clear all task-local sparse and source state without touching Qdrant.

        self._memories_by_id.clear()
        self._episode_ids_by_ordinal.clear()
        self._source_memories.clear()
        self._postings.clear()
        self._doc_lengths.clear()
        self._doc_count = 0
        self._avg_doc_length = 1.0
        self._max_ordinal = 1
