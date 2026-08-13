#############################################################################
# File: graph_extraction.py
#
# Description:
#   Extracts one source-grounded entity-relation graph for both graph
#   storage variants. It combines local parsing for explicit facts with
#   direct OpenAI-backed JSON extraction for unstructured context.
#
#   - Packs complete source entries under a fixed extraction budget.
#   - Parses explicit facts locally even when a task also contains free text.
#   - Runs independent extraction packs under a concurrency limit.
#   - Recovers minor JSON syntax mistakes and retries invalid model output.
#   - Merges duplicate entity names while retaining the most detailed
#     description.
#   - Creates missing entity nodes for valid relation endpoints.
#   - Validates relation provenance against entries in the current
#     prompt pack.
#############################################################################

from __future__ import annotations

import ast
import asyncio
import json
from collections.abc import Sequence

from asdrp.eval_schemas import MemoryEntry
from asdrp.memory.extraction import (
    EntityRecord,
    RelationRecord,
    normalize_key,
    parse_explicit_fact,
)
from asdrp.runtime import OpenAIRuntime, parse_json_payload


class GraphExtractor:
    # Turn source entries into entities and relations supported by the source.

    def __init__(
        self,
        *,
        runtime: OpenAIRuntime,
        input_token_limit: int = 12_000,
        concurrency: int = 8,
    ) -> None:
        if input_token_limit < 1:
            raise ValueError("input_token_limit must be at least 1")
        if concurrency < 1:
            raise ValueError("concurrency must be at least 1")
        self._runtime = runtime  # Makes extraction calls and counts prompt tokens.
        self._input_token_limit = input_token_limit  # Limit for one extraction pack.
        self._semaphore = asyncio.Semaphore(concurrency)  # Limits model calls.

    async def extract(
        self,
        entries: Sequence[MemoryEntry],
    ) -> tuple[list[EntityRecord], list[RelationRecord]]:
        # Extract packs concurrently and merge duplicate entity descriptions.

        fact_entries = [
            entry for entry in entries if entry.metadata.get("entry_kind") == "fact"
        ]
        model_entries = [
            entry for entry in entries if entry.metadata.get("entry_kind") != "fact"
        ]

        results: list[tuple[list[EntityRecord], list[RelationRecord]]] = []
        if fact_entries:
            # Facts are already structured, even when the same task also has free text.
            results.append(await self._extract_pack(fact_entries))

        packs = self._pack(model_entries)  # Only unstructured entries need the model.
        tasks = [asyncio.create_task(self._extract_pack(pack)) for pack in packs]
        try:
            results.extend(await asyncio.gather(*tasks))
        except BaseException:
            # Do not leave sibling extraction calls running after one pack fails.
            for task in tasks:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        entity_map: dict[str, EntityRecord] = {}  # Canonical name to best record.
        relations: list[RelationRecord] = []  # Edges from every pack.
        for entities, pack_relations in results:
            relations.extend(pack_relations)
            for entity in entities:
                key = normalize_key(entity.name)  # Matches spelling and case variants.
                if not key:
                    continue
                existing = entity_map.get(key)  # Keep the longer description.
                if existing is None or len(entity.description) > len(
                    existing.description
                ):
                    entity_map[key] = entity  # First or most detailed version.

        # Relations can introduce entities the extractor omitted from its entity
        # list. Create them rather than dropping valid edges.
        for relation in relations:
            for name in (relation.source, relation.target):
                key = normalize_key(name)  # Same key used by extracted entities.
                if key and key not in entity_map:
                    entity_map[key] = EntityRecord(name=name)  # Minimal missing node.
        return list(entity_map.values()), relations

    def _pack(self, entries: Sequence[MemoryEntry]) -> list[list[MemoryEntry]]:
        # Pack complete source entries below the extraction prompt budget.

        packs: list[list[MemoryEntry]] = []  # Finished extraction packs.
        current: list[MemoryEntry] = []  # Entries in the pack being filled.
        tokens = 450  # Leaves space for extraction instructions.
        for entry in entries:
            size = self._runtime.llm_counter.count(entry.text) + 30  # Text and label.
            if current and tokens + size > self._input_token_limit:
                packs.append(current)
                current = []  # Start the next extraction pack.
                tokens = 450  # Reset its instruction allowance.
            current.append(entry)
            tokens += size
        if current:
            packs.append(current)
        return packs

    async def _request_json_payload(self, prompt: str) -> dict[str, object]:
        # Retry model-format mistakes locally instead of failing the whole benchmark task.

        last_error: Exception | None = None
        for attempt in range(3):
            request_prompt = prompt
            if attempt:
                request_prompt += (
                    "\n\nIMPORTANT: The previous extraction response was not valid JSON. "
                    "Return exactly one complete JSON object matching the requested "
                    "entities/relations shape. Use double-quoted keys and strings, no "
                    "comments, no Markdown fences, and no trailing commas."
                )

            async with self._semaphore:
                response = await self._runtime.complete(request_prompt, phase="memory")

            try:
                payload = self._parse_model_payload(response)
                if not isinstance(payload, dict):
                    raise TypeError("Graph extraction did not return a JSON object")

                entities = payload.get("entities", [])
                relations = payload.get("relations", [])
                if entities is None:
                    entities = []
                if relations is None:
                    relations = []
                if not isinstance(entities, list) or not isinstance(relations, list):
                    raise TypeError(
                        "Graph extraction JSON must contain list-valued entities and relations"
                    )

                payload["entities"] = entities
                payload["relations"] = relations
                return payload
            except (json.JSONDecodeError, TypeError) as error:
                last_error = error

        assert last_error is not None
        raise ValueError(
            "Graph extraction failed to return valid JSON after 3 attempts"
        ) from last_error

    @staticmethod
    def _parse_model_payload(response: str) -> object:
        # Accept strict JSON first, then a safe Python-literal form for minor syntax slips.

        try:
            return parse_json_payload(response)
        except json.JSONDecodeError as json_error:
            object_start = response.find("{")
            object_end = response.rfind("}")
            if object_start < 0 or object_end <= object_start:
                raise json_error

            candidate = response[object_start : object_end + 1]
            try:
                return ast.literal_eval(candidate)
            except (SyntaxError, ValueError):
                raise json_error

    async def _extract_pack(
        self,
        entries: list[MemoryEntry],
    ) -> tuple[list[EntityRecord], list[RelationRecord]]:
        # Parse explicit facts locally; use the LLM for unstructured context.

        if all(entry.metadata.get("entry_kind") == "fact" for entry in entries):
            fact_entities: dict[str, EntityRecord] = {}  # Nodes from parsed facts.
            relations: list[RelationRecord] = []  # Edges from parsed facts.
            for entry in entries:
                fact = parse_explicit_fact(  # Local parsing avoids a model call.
                    entry.text,
                    ordinal=entry.ordinal,
                    source_id=entry.entry_id,
                )
                subject_key = normalize_key(fact.subject) if fact.subject else ""
                object_key = normalize_key(fact.object) if fact.object else ""
                if subject_key:
                    fact_entities[subject_key] = EntityRecord(name=fact.subject)
                if object_key:
                    fact_entities[object_key] = EntityRecord(name=fact.object)
                if subject_key and object_key:
                    relations.append(
                        RelationRecord(
                            source=fact.subject,
                            relation=fact.relation,
                            target=fact.object,
                            statement=fact.statement,
                            temporal=fact.temporal,
                            source_ordinal=fact.source_ordinal,
                            source_id=fact.source_id,
                        )
                    )
            return list(fact_entities.values()), relations

        # Show the model each entry's ID and source order.
        source = "\n\n".join(
            f"[ENTRY id={entry.entry_id} ordinal={entry.ordinal}]\n{entry.text}"
            for entry in entries
        )
        # Request graph records in a strict JSON shape.
        prompt = f"""
            Build a source-grounded knowledge graph from the benchmark context. The context
            is the only source of truth, even if it conflicts with real-world knowledge.
            Capture named entities, products, people, places, works, concepts, preferences,
            decisions, events, and explicit relationships. Preserve changed or conflicting
            facts as separate relations with their source order. Do not infer unsupported
            world knowledge.
            
            Return one compact JSON object only. Do not use Markdown, comments, trailing commas,
            or text before or after the object. Keep descriptions and statements concise, and do
            not repeat the same source-grounded fact more than once.
            {{
              "entities": [
                {{"name": "...", "type": "...", "description": "source-grounded description"}}
              ],
              "relations": [
                {{
                  "source": "entity name",
                  "relation": "concise normalized relation",
                  "target": "entity or literal value",
                  "statement": "self-contained source-grounded sentence",
                  "temporal": "date/time/order qualifier or empty string",
                  "source_id": "supporting ENTRY id",
                  "source_ordinal": 0
                }}
              ]
            }}
            
            CONTEXT:
            {source}
        """.strip()
        payload = await self._request_json_payload(prompt)

        known = {entry.entry_id: entry for entry in entries}  # Source lookup.
        fallback = entries[0]  # Used when the model gives no valid source ID.
        extracted_entities: list[EntityRecord] = []  # Valid response nodes.
        for item in payload.get("entities", []):
            if not isinstance(item, dict) or not str(item.get("name") or "").strip():
                continue
            extracted_entities.append(
                EntityRecord(
                    name=str(item["name"]).strip(),
                    entity_type=str(item.get("type") or "unknown").strip(),
                    description=str(item.get("description") or "").strip(),
                )
            )

        relations = []  # Valid edges from the response.
        for item in payload.get("relations", []):
            if not isinstance(item, dict):
                continue
            source_name = str(item.get("source") or "").strip()  # Edge start.
            target_name = str(item.get("target") or "").strip()  # Edge end.
            statement = str(item.get("statement") or "").strip()  # Support text.
            if not source_name or not target_name or not statement:
                continue
            requested_source_id = str(item.get("source_id") or fallback.entry_id)
            source_entry = known.get(requested_source_id, fallback)
            source_id = (
                source_entry.entry_id
            )  # Keep provenance inside this prompt pack.
            relations.append(
                RelationRecord(
                    source=source_name,
                    relation=str(item.get("relation") or "related_to").strip(),
                    target=target_name,
                    statement=statement,
                    temporal=str(item.get("temporal") or "").strip(),
                    source_ordinal=source_entry.ordinal,
                    source_id=source_id,
                )
            )
        return extracted_entities, relations
