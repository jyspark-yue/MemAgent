#############################################################################
# File: graph_memory.py
#
# Description:
#   Stores extracted entities and versioned directed relations in a
#   task-local in-memory graph. Retrieval selects likely entity entry
#   points and follows a bounded number of graph links.
#
#   - Builds a MultiDiGraph so several relation versions can be
#     retained.
#   - Embeds entity names and descriptions for dense seed selection.
#   - Marks older source-relation edges as inactive while preserving
#     history.
#   - Walks incoming and outgoing edges without exceeding the requested
#     hop count.
#   - Ranks walked edges by seed relevance, distance, activity, and
#     source order.
#   - Preserves temporal/source provenance in complete graph retrieval.
#   - Distinguishes conversation recall wording from historical queries.
#############################################################################

from __future__ import annotations

import math
import re
import uuid
from collections import deque
from collections.abc import Sequence
from typing import Any

import networkx as nx
import numpy as np

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseMemBlock
from asdrp.memory.extraction import normalize_key
from asdrp.memory.graph_extraction import GraphExtractor


class GraphMemoryBlock(BaseMemBlock):
    # Store entities and directed relations, then search by walking the graph.
    #
    # This true graph stays in memory instead of Qdrant. Nodes use cleaned entity
    # names. Directed edges keep relations, source order, time, and update state.
    # A query finds likely starting entities by vector similarity, then walks the
    # graph instead of searching every edge as flat text.

    memory_name = "graph"  # Public name used by the agent registry.

    def __init__(
        self,
        *,
        task_id: str,
        runtime: Any,
        extraction_input_tokens: int = 12_000,
        extraction_concurrency: int = 8,
        **_: object,
    ) -> None:
        super().__init__(task_id=task_id, runtime=runtime)
        self._graph = nx.MultiDiGraph()  # Allows several edges between two nodes.
        self._extractor = GraphExtractor(  # Finds nodes and edges in source text.
            runtime=runtime,
            input_token_limit=extraction_input_tokens,
            concurrency=extraction_concurrency,
        )
        self._entity_embeddings: dict[str, np.ndarray] = {}  # Node search vectors.
        self._latest_edge_by_key: dict[str, tuple[str, str, str]] = {}  # Newest edges.
        self._multi_value_fact_keys: set[str] = set()  # Relations with parallel values.

    async def initialize(self) -> None:
        # Initialize an empty task-local graph; no global graph is shared.

        self._graph.clear()
        self._entity_embeddings.clear()
        self._latest_edge_by_key.clear()
        self._multi_value_fact_keys.clear()

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Extract entities/relations, embed entities, and build update-aware edges.

        entities, relations = await self._extractor.extract(entries)
        for entity in entities:
            node_id = normalize_key(entity.name)  # Stable node key.
            if not node_id:
                continue
            self._graph.add_node(
                node_id,
                name=entity.name,
                entity_type=entity.entity_type,
                description=entity.description,
            )

        node_ids = list(self._graph.nodes)  # Fixed order for vector pairing.
        embedding_texts = [self._node_text(node_id) for node_id in node_ids]
        vectors = await self.runtime.embed_many(embedding_texts, phase="memory")
        self._entity_embeddings = {  # Pair node IDs with their search vectors.
            node_id: np.asarray(vector, dtype=np.float32)
            for node_id, vector in zip(node_ids, vectors, strict=True)
        }

        for relation in sorted(relations, key=lambda item: item.source_ordinal):
            source = normalize_key(relation.source)  # Canonical start node.
            target = normalize_key(relation.target)  # Canonical end node.
            if not source or not target:
                continue
            if source not in self._graph:
                self._graph.add_node(
                    source, name=relation.source, entity_type="unknown", description=""
                )
            if target not in self._graph:
                self._graph.add_node(
                    target, name=relation.target, entity_type="unknown", description=""
                )

            edge_id = uuid.uuid4().hex  # Unique key for this graph edge.
            fact_key = f"{source}|{normalize_key(relation.relation)}"
            previous = self._latest_edge_by_key.get(fact_key)  # Older edge version.
            if previous is not None and fact_key not in self._multi_value_fact_keys:
                old_source, old_target, old_key = previous  # Previous edge location.
                old_data = self._graph.edges[old_source, old_target, old_key]
                old_ordinal = int(old_data.get("ordinal", 0))
                if old_ordinal == relation.source_ordinal:
                    # Several values stated together are parallel facts, not updates.
                    self._multi_value_fact_keys.add(fact_key)
                else:
                    old_data["active"] = False  # Keep the older value as history.
                    old_data["superseded_by"] = edge_id  # Link to the new version.

            self._graph.add_edge(
                source,
                target,
                key=edge_id,
                relation=relation.relation,
                statement=relation.statement,
                temporal=relation.temporal,
                ordinal=relation.source_ordinal,
                source_id=relation.source_id,
                active=True,
                fact_key=fact_key,
            )
            self._latest_edge_by_key[fact_key] = (source, target, edge_id)

        # Embed any relation-introduced nodes that were absent from the extractor's
        # explicit entity list.
        missing = [  # Relation-only nodes that still need search vectors.
            node_id for node_id in self._graph if node_id not in self._entity_embeddings
        ]
        if missing:
            vectors = await self.runtime.embed_many(
                [self._node_text(node_id) for node_id in missing],
                phase="memory",
            )
            for node_id, vector in zip(missing, vectors, strict=True):
                self._entity_embeddings[node_id] = np.asarray(vector, dtype=np.float32)

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Find starting entities, walk up to k links, and rank the source edges.

        if self._graph.number_of_edges() == 0:
            return []
        if plan.mode in {"all", "global"}:
            return self._all_edges(include_inactive=True)

        query_vector = np.asarray(  # Vector used to choose graph entry points.
            await self.runtime.embed_one(query, phase="query"),
            dtype=np.float32,
        )
        seed_count = max(4, min(12, plan.top_k // 2 + 2))  # Entry points to walk.
        seeds = self._rank_seed_nodes(query, query_vector)[:seed_count]
        historical = self._is_historical_query(
            query
        )  # Whether unnecessary edges matter.

        edge_scores: dict[tuple[str, str, str], float] = {}  # Best score per edge.
        # Start a breadth-first walk from each matching node.
        queue: deque[tuple[str, int, float]] = deque(
            (node_id, 0, score) for node_id, score in seeds
        )
        best_visit_score: dict[str, float] = {}  # Strongest traversal state per node.
        while queue:
            node_id, depth, seed_score = queue.popleft()  # Next node in the walk.
            # A seed's incident edges are one hop away. Stop before a node whose
            # outgoing edges would exceed the requested path length.
            if depth >= plan.graph_hops:
                continue
            visit_score = seed_score / (1.0 + depth)  # Includes path-length decay.
            previous_score = best_visit_score.get(node_id, -math.inf)
            if visit_score <= previous_score:
                continue
            best_visit_score[node_id] = visit_score

            # Search outgoing and incoming edges so either named endpoint can seed.
            incident = list(self._graph.out_edges(node_id, keys=True, data=True))
            incident += list(self._graph.in_edges(node_id, keys=True, data=True))
            for source, target, key, data in incident:
                if (
                    not historical
                    and plan.prefer_latest
                    and not data.get("active", True)
                ):
                    continue
                active_bonus = 0.05 if data.get("active", True) else 0.0
                score = visit_score + active_bonus  # Final traversal score.
                edge_ref = (source, target, key)  # Stable edge lookup tuple.
                edge_scores[edge_ref] = max(edge_scores.get(edge_ref, -math.inf), score)
                other = target if source == node_id else source  # Next node to visit.
                if depth + 1 < plan.graph_hops:
                    queue.append((other, depth + 1, seed_score * 0.8))

        # Sort walked edges by score, then by source order for ties.
        ranked = sorted(
            edge_scores.items(),
            key=lambda item: (
                item[1],
                self._graph.edges[item[0]]["ordinal"],
            ),
            reverse=True,
        )[: max(plan.top_k, plan.top_k * plan.candidate_multiplier)]

        results = []  # Neutral memory records returned to the agent.
        seen_edges: set[tuple[object, ...]] = set()  # Removes only exact edge dupes.
        for (source, target, key), score in ranked:
            data = self._graph.edges[source, target, key]  # Stored edge fields.
            statement = str(data["statement"])  # Text shown to the agent.
            dedupe_key = (
                source,
                target,
                normalize_key(str(data.get("relation") or "related_to")),
                statement,
                data.get("source_id", ""),
                int(data.get("ordinal", 0)),
            )
            if dedupe_key in seen_edges:
                continue
            seen_edges.add(dedupe_key)
            results.append(
                RetrievedMemory(
                    memory_id=key,
                    text=statement,
                    score=float(score),
                    ordinal=int(data["ordinal"]),
                    metadata={
                        "record_type": "edge",
                        "source_entity": self._graph.nodes[source].get("name", source),
                        "target_entity": self._graph.nodes[target].get("name", target),
                        "relation": data.get("relation", "related_to"),
                        "temporal": data.get("temporal", ""),
                        "source_id": data.get("source_id", ""),
                        "active": bool(data.get("active", True)),
                    },
                )
            )
            if len(results) >= plan.top_k:
                break
        return results

    async def close(self) -> None:
        # Release all task-local graph and embedding state.

        self._graph.clear()
        self._entity_embeddings.clear()
        self._latest_edge_by_key.clear()
        self._multi_value_fact_keys.clear()

    def _rank_seed_nodes(
        self,
        query: str,
        query_vector: np.ndarray,
    ) -> list[tuple[str, float]]:
        # Blend dense similarity with exact entity-name overlap.

        normalized_query = normalize_key(query)  # Same form used for node IDs.
        query_terms = set(normalized_query.split())  # Words used for name overlap.
        query_norm = float(np.linalg.norm(query_vector)) or 1.0  # Safe divisor.
        ranked = []  # Node IDs with their blended scores.
        for node_id, vector in self._entity_embeddings.items():
            vector_norm = float(np.linalg.norm(vector)) or 1.0  # Safe divisor.
            semantic = float(np.dot(query_vector, vector) / (query_norm * vector_norm))
            node_terms = set(node_id.split())  # Words in the entity name.
            lexical = len(query_terms & node_terms) / max(1, len(node_terms))
            exact = (
                1.0 if node_id and f" {node_id} " in f" {normalized_query} " else 0.0
            )
            ranked.append((node_id, 0.78 * semantic + 0.17 * lexical + 0.05 * exact))
        ranked.sort(key=lambda item: item[1], reverse=True)
        return ranked

    def _node_text(self, node_id: str) -> str:
        # Produce source-grounded text used solely for entity embeddings.

        data = self._graph.nodes[node_id]  # Saved name, type, and description.
        description = str(data.get("description") or "").strip()
        return f"{data.get('name', node_id)}. {description}".strip()

    def _all_edges(self, *, include_inactive: bool) -> list[RetrievedMemory]:
        # Return graph statements in source order for explicit all-memory tasks.

        results = []  # Every kept edge converted to a memory record.
        for source, target, key, data in self._graph.edges(keys=True, data=True):
            if not include_inactive and not data.get("active", True):
                continue
            results.append(
                RetrievedMemory(
                    memory_id=key,
                    text=str(data.get("statement", "")),
                    score=1.0,
                    ordinal=int(data.get("ordinal", 0)),
                    metadata={
                        "record_type": "edge",
                        "source_entity": self._graph.nodes[source].get("name", source),
                        "target_entity": self._graph.nodes[target].get("name", target),
                        "relation": data.get("relation", "related_to"),
                        "temporal": data.get("temporal", ""),
                        "source_id": data.get("source_id", ""),
                        "active": bool(data.get("active", True)),
                    },
                )
            )
        return sorted(results, key=lambda item: (item.ordinal, item.memory_id))

    @staticmethod
    def _query_without_conversation_reference(query: str) -> str:
        # "Previous conversation" recalls unnecessary context; it does not request an unnecessary fact.

        lowered = query.casefold()
        return re.sub(
            r"\b(previous|earlier|last|prior|past)\s+"
            r"(conversation|chat|discussion|session|message|exchange|talk)\b",
            "",
            lowered,
        )

    @classmethod
    def _is_temporal_query(cls, query: str) -> bool:
        # Detect actual chronology/update requests after removing recall phrasing.

        lowered = cls._query_without_conversation_reference(query)
        return bool(
            re.search(
                r"\b(before|after|earlier|later|latest|newest|previous|first|last|"
                r"when|timeline|chronolog\w*|changed|updated|used to|at the time|"
                r"originally|history|historical)\b",
                lowered,
            )
        )

    @classmethod
    def _is_historical_query(cls, query: str) -> bool:
        # Identify questions that intentionally need superseded graph edges.

        lowered = cls._query_without_conversation_reference(query)
        return bool(
            re.search(
                r"\b(before|earlier|previous|originally|first|used to|at the time|"
                r"history|historical)\b",
                lowered,
            )
        )
