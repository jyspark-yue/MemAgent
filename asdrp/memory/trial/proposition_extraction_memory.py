#############################################################################
# File: proposition_extraction_memory.py
#
# Description:
#   Stores independently retrievable atomic propositions and their source
#   history. Current-state retrieval can select the latest value while
#   historical questions keep earlier versions available.
#
#   - Packs source entries into model-safe extraction groups to reduce request
#     overhead without dropping source content.
#   - Uses compact source sections and plain proposition lines instead of large
#     structured objects, reducing output tokens and formatting failures.
#   - Extracts only the highest-value propositions per source to trade a small
#     amount of recall for a large runtime reduction.
#   - Parses explicit benchmark facts locally even when they are mixed with
#     conversational entries, avoiding unnecessary model calls.
#   - Uses larger extraction packs and bounded per-task concurrency so long LLM
#     calls do not starve the shared embedding service.
#   - Accepts complete best-effort extraction output and retries only one likely
#     truncated source instead of recursively rereading failed packs.
#   - Keeps statement, subject, relation, object, time, fact key, and source
#     provenance for retrieval and current-state filtering.
#   - Embeds each proposition set once, then performs sequential larger Qdrant
#     writes so transport retries never repeat completed embedding work.
#   - Retries Qdrant errors even when qdrant-client wraps a transport ReadError
#     inside an empty ResponseHandlingException.
#   - Prints lightweight extraction progress while long memory builds run.
#   - Applies a fast local round-robin proposition cap instead of spending more
#     LLM calls on post-extraction condensation.
#   - Combines dense Qdrant search with a lightweight local BM25 index.
#   - Filters fused current-state candidates by fact key while preserving history.
#
# Authors:
#   @author     Eric Vincent Fernandes
#
# Date:
#   Modified:   August 8, 2026 (Eric Vincent Fernandes)
#############################################################################

from __future__ import annotations

import asyncio
import random
import re
import uuid
from collections.abc import Awaitable, Callable, Sequence
from typing import Any, TypeVar

from qdrant_client.models import PointStruct

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseQdrantMemoryBlock
from asdrp.memory.hybrid_retrieval import HybridTextRetriever
from asdrp.memory.extraction import (
    PropositionRecord,
    normalize_key,
    parse_explicit_fact,
)


T = TypeVar("T")


class PropositionMemoryBlock(BaseQdrantMemoryBlock):
    # Store small facts that can each be retrieved on their own.
    #
    # All proposition versions are retained. During ordinary current-state
    # retrieval, newer values for the same subject/relation key supersede older
    # values. Historical or temporal questions preserve the competing versions.
    # The speed-biased default keeps broad source coverage while limiting the
    # generation, embedding, and Qdrant work needed for large LongMemEval tasks.

    memory_name = "proposition"  # Prefix for this task's Qdrant collection.

    def __init__(
        self,
        *,
        max_propositions: int = 1_500,
        extraction_input_tokens: int = 80_000,
        extraction_concurrency: int = 3,
        max_propositions_per_entry: int = 4,
        **kwargs: Any,
    ) -> None:
        if max_propositions < 1:
            raise ValueError("max_propositions must be at least 1")
        if extraction_input_tokens < 1:
            raise ValueError("extraction_input_tokens must be at least 1")
        if extraction_concurrency < 1:
            raise ValueError("extraction_concurrency must be at least 1")
        if max_propositions_per_entry < 1:
            raise ValueError("max_propositions_per_entry must be at least 1")
        super().__init__(**kwargs)
        self._max_propositions = max_propositions  # Speed-biased total memory cap.
        self._extraction_input_tokens = (
            extraction_input_tokens  # Larger packs reduce LLM request overhead.
        )
        self._extraction_concurrency = (
            extraction_concurrency  # Leaves shared API capacity for embeddings.
        )
        self._max_propositions_per_entry = (
            max_propositions_per_entry  # Bound model output at the source.
        )
        self._hybrid_retriever = HybridTextRetriever()

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Parse explicit facts locally, extract conversations, then store all facts.

        self._hybrid_retriever.clear()  # Supports a clean rebuild if reused.

        # Explicit benchmark facts already have a stable sentence form. Parsing them
        # locally is lossless and avoids sending them through the LLM just because
        # they happen to share a task with conversational entries.
        explicit_propositions = [
            parse_explicit_fact(
                entry.text,
                ordinal=entry.ordinal,
                source_id=entry.entry_id,
            )
            for entry in entries
            if entry.metadata.get("entry_kind") == "fact"
        ]
        model_entries = [
            entry for entry in entries if entry.metadata.get("entry_kind") != "fact"
        ]
        packs = self._pack_entries(model_entries)  # Model-backed extraction groups.

        if packs:
            print(
                f"[proposition] extracting {len(model_entries)} conversational entries "
                f"in {len(packs)} packs"
                + (
                    f" ({len(explicit_propositions)} explicit facts parsed locally)"
                    if explicit_propositions
                    else ""
                ),
                flush=True,
            )
        elif explicit_propositions:
            print(
                f"[proposition] parsed {len(explicit_propositions)} explicit facts locally",
                flush=True,
            )
        else:
            return

        extracted: list[list[PropositionRecord]] = []  # Completed model-backed packs.
        if packs:
            # Keep each task from filling the global API queue with long generations.
            # With several evaluator workers this still saturates the shared API limit,
            # while completed tasks can get short embedding requests through promptly.
            semaphore = asyncio.Semaphore(self._extraction_concurrency)

            async def extract_with_local_limit(
                pack: list[MemoryEntry],
            ) -> list[PropositionRecord]:
                async with semaphore:
                    return await self._extract_pack(pack)

            tasks = [
                asyncio.create_task(extract_with_local_limit(pack)) for pack in packs
            ]
            completed = 0  # Number of model packs that have finished.
            proposition_count = 0  # Running count shown in progress output.
            progress_step = max(1, len(packs) // 4)  # About four updates per build.

            try:
                for task in asyncio.as_completed(tasks):
                    pack_result = await task
                    extracted.append(pack_result)
                    completed += 1
                    proposition_count += len(pack_result)
                    if completed == len(packs) or completed % progress_step == 0:
                        print(
                            f"[proposition] extracted {completed}/{len(packs)} packs "
                            f"({proposition_count} propositions)",
                            flush=True,
                        )
            except BaseException:
                # Stop unfinished sibling work when this memory build cannot continue.
                for task in tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*tasks, return_exceptions=True)
                raise

        # Apply the speed cap only to model-extracted conversation memory. Explicit
        # benchmark facts are already compact and should not be dropped by this tuning.
        model_propositions = [item for pack in extracted for item in pack]
        model_propositions = self._deduplicate_propositions(model_propositions)
        model_propositions = self._cap_each_source(model_propositions)
        model_propositions = self._cap_propositions(model_propositions)
        propositions = self._deduplicate_propositions(
            explicit_propositions + model_propositions
        )

        # Keep all fact parts and their source details beside the vector.
        records = [
            {
                "id": str(uuid.uuid4()),
                "text": proposition.statement,
                "ordinal": proposition.source_ordinal,
                "subject": proposition.subject,
                "relation": proposition.relation,
                "object": proposition.object,
                "temporal": proposition.temporal,
                "fact_key": proposition.fact_key,
                "source_id": proposition.source_id,
                "entry_kind": "proposition",
                **proposition.metadata,
            }
            for proposition in propositions
            if proposition.statement.strip()
        ]
        print(
            f"[proposition] storing {len(records)} propositions",
            flush=True,
        )
        await self._store_proposition_records(records)
        self._hybrid_retriever.build(
            records,
            memory_id_for_record=self._qdrant_point_id,
        )

    @staticmethod
    def _qdrant_error_chain(error: BaseException) -> list[BaseException]:
        # Qdrant wraps httpx transport failures in ResponseHandlingException.
        # Walk those wrappers so an empty outer message does not hide a retryable
        # ReadError, timeout, or connection failure.

        chain: list[BaseException] = []
        seen: set[int] = set()
        current: BaseException | None = error
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            chain.append(current)
            cause = current.__cause__ or current.__context__
            current = cause if isinstance(cause, BaseException) else None
        return chain

    @classmethod
    def _is_retryable_qdrant_error(cls, error: BaseException) -> bool:
        # Check both the Qdrant wrapper and every nested HTTP/transport error.

        retryable_statuses = {408, 409, 429, 500, 502, 503, 504}
        retryable_names = {
            "ConnectError",
            "ConnectTimeout",
            "PoolTimeout",
            "ReadError",
            "ReadTimeout",
            "RemoteProtocolError",
            "WriteError",
            "WriteTimeout",
        }
        fragments = (
            "request timeout",
            "timed out",
            "timeout",
            "connection reset",
            "connection refused",
            "connection aborted",
            "connection error",
            "all connection attempts failed",
            "temporarily unavailable",
            "service unavailable",
            "server disconnected",
            "broken pipe",
            "read error",
        )

        for item in cls._qdrant_error_chain(error):
            if type(item).__name__ in retryable_names:
                return True

            for attribute in ("status_code", "status"):
                status = getattr(item, attribute, None)
                if isinstance(status, int) and status in retryable_statuses:
                    return True

            response = getattr(item, "response", None)
            if response is not None:
                status = getattr(response, "status_code", None)
                if isinstance(status, int) and status in retryable_statuses:
                    return True

            message = str(item).casefold()
            if any(fragment in message for fragment in fragments):
                return True

        return False

    async def _qdrant_call(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        # Proposition runs can create many concurrent collections. Retry wrapped
        # transport failures here because qdrant-client often exposes only an empty
        # ResponseHandlingException while the useful ReadError lives in __cause__.

        last_error: Exception | None = None
        for attempt_index in range(self._retry_attempts):
            try:
                return await operation()
            except asyncio.CancelledError:
                raise
            except Exception as error:
                last_error = error
                final_attempt = attempt_index + 1 >= self._retry_attempts
                if final_attempt or not self._is_retryable_qdrant_error(error):
                    raise

                delay = min(
                    self._retry_base_delay * (2**attempt_index),
                    self._retry_max_delay,
                )
                delay *= random.uniform(0.8, 1.2)
                nested = self._qdrant_error_chain(error)
                root = nested[-1]
                print(
                    f"[qdrant] {operation_name} retry {attempt_index + 1}/"
                    f"{self._retry_attempts - 1} in {delay:.1f}s: "
                    f"{type(error).__name__} -> {type(root).__name__}: {root}",
                    flush=True,
                )
                await asyncio.sleep(delay)

        if last_error is not None:
            raise last_error
        raise RuntimeError(
            f"Qdrant operation {operation_name!r} ended without a result"
        )

    async def _store_proposition_records(
        self,
        records: Sequence[dict[str, Any]],
    ) -> None:
        # Embed proposition text once, then write the completed vectors to Qdrant.
        # This prevents a transient Qdrant error from causing another embedding job
        # and avoids several workers hammering local Qdrant with four writes each.

        if not records:
            return

        texts = [str(record["text"]) for record in records]
        print(
            f"[proposition] embedding {len(texts)} propositions",
            flush=True,
        )
        vectors = await self.runtime.embed_many(texts, phase="memory")
        if len(vectors) != len(records):
            raise RuntimeError("Embedding result count does not match record count")

        # Proposition vectors are small enough that 512-point writes substantially
        # reduce HTTP overhead without creating oversized local Qdrant requests.
        batch_size = min(max(self._qdrant_batch_size, 512), 1_024)
        total_batches = (len(records) + batch_size - 1) // batch_size
        for batch_index, start in enumerate(
            range(0, len(records), batch_size), start=1
        ):
            record_batch = records[start : start + batch_size]
            vector_batch = vectors[start : start + batch_size]
            points: list[PointStruct] = []

            for record, vector in zip(record_batch, vector_batch, strict=True):
                payload = dict(record)
                record_id = str(payload.pop("id"))
                point_id = self._qdrant_point_id(record_id)
                payload.setdefault("record_id", record_id)
                points.append(
                    PointStruct(
                        id=point_id,
                        vector=vector,
                        payload=payload,
                    )
                )

            # Writes are intentionally sequential for proposition memory. Across ten
            # evaluator workers Qdrant can still receive ten writes at once, while a
            # single task no longer contributes four concurrent multi-megabyte upserts.
            await self._qdrant_call(
                "upsert",
                lambda points=points: self._client.upsert(
                    collection_name=self.collection_name,
                    points=points,
                    wait=True,
                ),
            )
            if total_batches > 1 and (
                batch_index == total_batches or batch_index % 4 == 0
            ):
                print(
                    f"[proposition] stored {batch_index}/{total_batches} Qdrant batches",
                    flush=True,
                )

    async def close(self) -> None:
        # Release Qdrant state and the task-local lexical index.

        try:
            await super().close()
        finally:
            self._hybrid_retriever.clear()

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Retrieve facts and keep only the latest value when requested.

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

        historical = self._is_historical_query(
            query
        )  # Whether unnecessary values matter.
        if plan.prefer_latest and not historical:
            newest: dict[str, RetrievedMemory] = {}  # Latest fact for each key.
            without_key: list[RetrievedMemory] = []  # Facts that cannot be grouped.
            for item in candidates:
                key = str(item.metadata.get("fact_key", ""))  # Subject/relation pair.
                if not key or key.endswith("|asserts"):
                    without_key.append(item)
                    continue
                previous = newest.get(key)  # Newest version seen so far.
                if previous is None or item.ordinal > previous.ordinal:
                    newest[key] = item  # Replace an older version of this fact.
            candidates = list(newest.values()) + without_key  # Kept current facts.
            candidates.sort(key=lambda item: (item.score, item.ordinal), reverse=True)
        else:
            candidates.sort(key=lambda item: item.score, reverse=True)
        return candidates[: plan.top_k]

    def _pack_entries(self, entries: Sequence[MemoryEntry]) -> list[list[MemoryEntry]]:
        # Pack complete entries for extraction without crossing model-safe size.

        packs: list[list[MemoryEntry]] = []  # Finished extraction groups.
        current: list[MemoryEntry] = []  # Entries in the group being filled.
        token_count = 450  # Leaves room for the extraction instructions.
        for entry in entries:
            size = (
                self.runtime.llm_counter.count(entry.text) + 12
            )  # Text and short label.
            if current and token_count + size > self._extraction_input_tokens:
                packs.append(current)
                current = []  # Start the next extraction pack.
                token_count = 450  # Reset its instruction allowance.
            current.append(entry)
            token_count += size
        if current:
            packs.append(current)
        return packs

    async def _extract_pack(
        self,
        entries: list[MemoryEntry],
    ) -> list[PropositionRecord]:
        # Extract propositions from one conversation pack without fragile JSON/TSV.

        if not entries:
            return []

        # Print each short source label once, then let the model return plain lines
        # beneath that source. This is smaller and much harder to format incorrectly
        # than repeating four structured fields for every proposition.
        source_labels = {f"E{index}": entry for index, entry in enumerate(entries)}
        source = "\n\n".join(
            f"[{label}]\n{entry.text}" for label, entry in source_labels.items()
        )

        prompt = f"""
You are a precise proposition extraction system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review every source section in the conversation segment.
2. Extract specific, concrete propositions: facts, opinions, preferences, beliefs,
   experiences, requirements, constraints, goals, plans, decisions, and important
   information discovered in the conversation.
3. For EACH source section, return at most {self._max_propositions_per_entry} propositions.
   Rank them by future retrieval value and keep only the most useful ones. Prioritize:
   - user-specific facts, preferences, constraints, experiences, and goals;
   - concrete assistant recommendations, decisions, or conclusions specific to the user;
   - dates, numbers, negatives, conflicts, changed values, and unusual named details.
4. Preserve names, dates, numbers, negatives, conflicts, and changed values exactly
   enough to answer later questions.
5. Do not infer unsupported facts or duplicate the same proposition within a source.
6. Skip generic assistant exposition, filler, greetings, examples, and conversational
   scaffolding unless they contain information specific to this conversation.
7. Keep each proposition concise and self-contained. Prefer one strong proposition over
   several weaker propositions that repeat the same idea.

OUTPUT FORMAT:
- For every source label below, output that label once on its own line, then output
  each new proposition for that source on its own line.
- Copy source labels exactly, including brackets.
- A source with no useful propositions may have no proposition lines beneath it.
- Do not use JSON, tables, tabs, commentary, or code fences.
- After the final source section, output END on its own line.

Example:
[E0]
The user prefers dark chocolate.
The user plans to visit Boston in October.
[E1]
The assistant recommended taking the train.
END

Conversation segment:
---
{source}
---
""".strip()

        response = await self.runtime.complete(prompt, phase="memory")
        propositions, seen_labels, saw_end = self._parse_extraction_response(
            response, source_labels
        )

        # A complete response is accepted as-is even if the model omitted empty source
        # labels. Retrying every omitted label costs far more than the small recall gain.
        if saw_end:
            return propositions

        # If the response was truncated, retry only the final missing source once. This
        # salvages the most likely incomplete section without recursively rereading a
        # large fraction of the pack.
        missing = [
            entry for label, entry in source_labels.items() if label not in seen_labels
        ]
        if not missing:
            return propositions

        entry = missing[-1]
        fallback_prompt = f"""
Extract at most {self._max_propositions_per_entry} high-value propositions from this
conversation segment. Prioritize user-specific facts, preferences, constraints, goals,
dates, numbers, conflicts, updates, and concrete assistant recommendations. Skip generic
filler and exposition. Keep each proposition concise and self-contained.

Return ONLY one proposition per line. If there are none, return exactly:
NO NEW PROPOSITIONS

Conversation segment:
---
{entry.text}
---
""".strip()
        fallback = await self.runtime.complete(fallback_prompt, phase="memory")
        propositions.extend(self._parse_single_source_response(fallback, entry))
        return propositions

    @staticmethod
    def _record_from_statement(
        statement: str,
        source_entry: MemoryEntry,
    ) -> PropositionRecord:
        # Derive fact metadata locally; semantic retrieval still uses the full statement.

        parsed = parse_explicit_fact(
            statement,
            ordinal=source_entry.ordinal,
            source_id=source_entry.entry_id,
        )
        return PropositionRecord(
            statement=parsed.statement,
            subject=parsed.subject,
            relation=parsed.relation,
            object=parsed.object,
            temporal=parsed.temporal,
            source_ordinal=source_entry.ordinal,
            source_id=source_entry.entry_id,
            metadata=parsed.metadata,
        )

    @classmethod
    def _parse_extraction_response(
        cls,
        response: str,
        source_labels: dict[str, MemoryEntry],
    ) -> tuple[list[PropositionRecord], set[str], bool]:
        # Parse source sections independently. Ordinary text beneath a valid source
        # header is a proposition, so punctuation or delimiter mistakes cannot poison
        # an otherwise usable pack.

        propositions: list[PropositionRecord] = []
        seen_labels: set[str] = set()
        current_entry: MemoryEntry | None = None
        current_label: str | None = None
        saw_end = False

        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            if line.casefold().rstrip(".") == "end":
                saw_end = True
                break

            header = re.fullmatch(r"\[?(E\d+)\]?:?", line, flags=re.IGNORECASE)
            if header:
                label = header.group(1).upper()
                current_entry = source_labels.get(label)
                current_label = label if current_entry is not None else None
                if current_entry is not None:
                    seen_labels.add(label)
                continue

            # Backward compatibility with the earlier TSV/triple-bar protocol.
            legacy_parts = [part.strip() for part in line.split("\t", 3)]
            if len(legacy_parts) != 4:
                legacy_parts = [part.strip() for part in line.split("|||", 3)]
            if len(legacy_parts) == 4:
                label, _subject, _relation, statement = legacy_parts
                normalized_label = label.strip("[] ").upper()
                source_entry = source_labels.get(normalized_label)
                if source_entry is not None and statement:
                    seen_labels.add(normalized_label)
                    propositions.append(
                        cls._record_from_statement(statement, source_entry)
                    )
                    current_entry = source_entry
                    current_label = normalized_label
                continue

            if current_entry is None:
                continue

            # Bullets and numbering are harmless even though the prompt does not ask
            # for them. Preserve the actual proposition text after removing the marker.
            statement = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
            if not statement:
                continue
            if statement.casefold().rstrip(".") == "no new propositions":
                continue
            propositions.append(cls._record_from_statement(statement, current_entry))

        # If the final END marker is missing, the response may have been truncated in
        # the last source section. Retry only that final section rather than trusting
        # possibly incomplete evidence or recursively rereading the whole pack.
        if not saw_end and current_label is not None:
            seen_labels.discard(current_label)

        return propositions, seen_labels, saw_end

    @classmethod
    def _parse_single_source_response(
        cls,
        response: str,
        source_entry: MemoryEntry,
    ) -> list[PropositionRecord]:
        # A one-source fallback needs no source labels, so nearly any normal text output
        # can be recovered instead of raising after an expensive completed request.

        stripped = response.strip()
        if not stripped or stripped.casefold().rstrip(".") == "no new propositions":
            return []

        propositions: list[PropositionRecord] = []
        for raw_line in response.splitlines():
            line = raw_line.strip()
            if not line or line.startswith("```"):
                continue
            line = re.sub(r"^(?:[-*•]\s+|\d+[.)]\s+)", "", line).strip()
            if not line or line.casefold().rstrip(".") in {
                "end",
                "no new propositions",
            }:
                continue
            propositions.append(cls._record_from_statement(line, source_entry))
        return propositions

    @staticmethod
    def _deduplicate_propositions(
        propositions: list[PropositionRecord],
    ) -> list[PropositionRecord]:
        # Remove only exact same-source duplicates so chronology is never collapsed.

        seen: set[tuple[str, str]] = set()  # Source ID and normalized statement.
        unique: list[PropositionRecord] = []  # Propositions kept in source order.
        for proposition in sorted(propositions, key=lambda item: item.source_ordinal):
            key = (proposition.source_id, normalize_key(proposition.statement))
            if key in seen:
                continue
            seen.add(key)
            unique.append(proposition)
        return unique

    def _cap_each_source(
        self,
        propositions: list[PropositionRecord],
    ) -> list[PropositionRecord]:
        # The prompt asks the model to rank propositions by usefulness. Keep only the
        # first few from each source even if the model ignores that requested limit.

        counts: dict[str, int] = {}
        kept: list[PropositionRecord] = []
        for proposition in propositions:
            count = counts.get(proposition.source_id, 0)
            if count >= self._max_propositions_per_entry:
                continue
            counts[proposition.source_id] = count + 1
            kept.append(proposition)
        return kept

    def _cap_propositions(
        self,
        propositions: list[PropositionRecord],
    ) -> list[PropositionRecord]:
        # Keep broad source coverage without paying for another LLM condensation pass.
        # The extractor already orders each source's propositions by importance, so a
        # round-robin cap keeps the strongest fact from as many sources as possible
        # before taking second/third/fourth facts from the same source.

        if len(propositions) <= self._max_propositions:
            return propositions

        by_source: dict[str, list[PropositionRecord]] = {}
        source_order: list[str] = []
        for proposition in propositions:
            source_id = proposition.source_id
            if source_id not in by_source:
                by_source[source_id] = []
                source_order.append(source_id)
            by_source[source_id].append(proposition)

        kept: list[PropositionRecord] = []
        depth = 0
        while len(kept) < self._max_propositions:
            added = False
            for source_id in source_order:
                source_items = by_source[source_id]
                if depth < len(source_items):
                    kept.append(source_items[depth])
                    added = True
                    if len(kept) >= self._max_propositions:
                        break
            if not added:
                break
            depth += 1

        kept.sort(key=lambda item: item.source_ordinal)
        print(
            f"[proposition] capped {len(propositions)} -> {len(kept)} propositions "
            f"without extra LLM condensation",
            flush=True,
        )
        return kept

    @staticmethod
    def _is_historical_query(query: str) -> bool:
        lowered = query.casefold()  # Makes phrase checks case-insensitive.
        return any(
            phrase in lowered
            for phrase in (
                "previous",
                "before",
                "earlier",
                "originally",
                "first",
                "used to",
                "changed",
                "at the time",
                "what was",
            )
        )
