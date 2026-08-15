#############################################################################
# File: hvm.py
#
# Description:
#   Builds Hierarchical Vector Memory as a RAPTOR-style summary tree. It
#   stores source leaves and every summary level so retrieval can combine
#   global structure with exact evidence.
#
#   - Embeds source leaves and recursively groups them into parent
#     summaries.
#   - Uses PCA reduction and BIC-selected Gaussian mixtures for clustering.
#   - Fits groups to summary prompts and guarantees progress toward one
#     root.
#   - Stores node levels, source ranges, parent links, child links,
#     text, and vectors in Qdrant.
#   - Traverses relevant branches from the root and merges them with
#     collapsed-tree retrieval.
#   - Returns a compact mix of summaries and leaves without duplicate
#     text.
#############################################################################

from __future__ import annotations

import asyncio
import math
import uuid
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseQdrantMemoryBlock


@dataclass(slots=True)
class _RaptorNode:
    # One source leaf or summary node in the RAPTOR tree.

    node_id: str  # Unique ID shared by Qdrant and parent links.
    text: str  # Source text or a summary.
    embedding: list[float]  # Vector used for clustering and retrieval.
    level: int  # Leaves are zero and summaries increase upward.
    start_ordinal: int  # First source entry covered by this node.
    end_ordinal: int  # Last source entry covered by this node.
    child_ids: list[str] = field(default_factory=list)  # Nodes directly below it.
    parent_id: str = ""  # Node directly above it, or blank for the root.


class HVMMemoryBlock(BaseQdrantMemoryBlock):
    # Build a RAPTOR tree from source leaves and repeated summaries.
    #
    # Leaves hold the complete source entries. Each level groups similar nodes and
    # summarizes each group until one root remains. Qdrant keeps every leaf and
    # summary. Search starts with broad summaries and follows them toward useful
    # source leaves.

    memory_name = "hvm"  # Prefix for this task's Qdrant collection.

    def __init__(
        self,
        *,
        max_cluster_size: int = 8,
        max_clusters: int = 32,
        max_levels: int = 8,
        summary_input_tokens: int = 12_000,
        summary_output_tokens: int = 900,
        summary_concurrency: int = 8,
        random_seed: int = 42,
        **kwargs: Any,
    ) -> None:
        if max_cluster_size < 2:
            raise ValueError("max_cluster_size must be at least 2")
        if max_clusters < 2:
            raise ValueError("max_clusters must be at least 2")
        if max_levels < 1:
            raise ValueError("max_levels must be at least 1")
        if summary_input_tokens < 1 or summary_output_tokens < 1:
            raise ValueError("summary token limits must be at least 1")
        if summary_concurrency < 1:
            raise ValueError("summary_concurrency must be at least 1")
        super().__init__(**kwargs)
        self._max_cluster_size = max_cluster_size  # Target leaves per cluster.
        self._max_clusters = max_clusters  # Cluster cap for one level.
        self._max_levels = max_levels  # Normal clustering passes.
        self._summary_input_tokens = summary_input_tokens  # One prompt limit.
        self._summary_output_tokens = summary_output_tokens  # Requested summary size.
        self._summary_semaphore = asyncio.Semaphore(summary_concurrency)
        self._random_seed = random_seed  # Keeps clustering repeatable.
        self._max_level = 0  # Highest level stored for the current task.

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Build the complete RAPTOR hierarchy and store every tree node once.

        usable = [entry for entry in entries if entry.text.strip()]  # Non-empty leaves.
        if not usable:
            return

        leaf_embeddings = await self.runtime.embed_many(  # Vectors for raw leaves.
            [entry.text for entry in usable],
            phase="memory",
        )
        leaves = [  # Level-zero nodes keep each source entry whole.
            _RaptorNode(
                node_id=str(uuid.uuid4()),
                text=entry.text,
                embedding=embedding,
                level=0,
                start_ordinal=entry.ordinal,
                end_ordinal=entry.ordinal,
            )
            for entry, embedding in zip(usable, leaf_embeddings, strict=True)
        ]

        all_nodes = list(leaves)  # Leaves and every summary made above them.
        current = leaves  # Nodes being grouped during this pass.
        level = 0  # Leaves start at level zero.
        while len(current) > 1 and level < self._max_levels:
            level += 1
            # Clustering is CPU work, so run it outside the event loop.
            groups = await asyncio.to_thread(self._cluster_nodes, current)
            groups = self._fit_groups_to_prompt(groups)  # Keep prompts under limit.
            if len(groups) >= len(current):
                groups = [  # Pair nodes when clustering did not reduce them.
                    current[index : index + 2] for index in range(0, len(current), 2)
                ]

            summaries = await asyncio.gather(  # One parent per fitted group.
                *(self._summarize_group(group, level) for group in groups)
            )
            summary_embeddings = await self.runtime.embed_many(
                [summary.text for summary in summaries],
                phase="memory",
            )
            parents = []  # Summary nodes that form the next level.
            for summary, embedding, group in zip(
                summaries,
                summary_embeddings,
                groups,
                strict=True,
            ):
                summary.embedding = embedding  # Vector used at the next level.
                for child in group:
                    child.parent_id = summary.node_id  # Link child to this parent.
                parents.append(summary)
            all_nodes.extend(parents)
            current = parents  # Build the next level from these summaries.

        # If the configured level cap is reached with multiple nodes, create one
        # final root in safe groups instead of losing the whole-source view.
        while len(current) > 1:
            level += 1
            groups = [  # Simple fixed groups finish the root after the level cap.
                current[index : index + self._max_cluster_size]
                for index in range(0, len(current), self._max_cluster_size)
            ]
            summaries = await asyncio.gather(
                *(self._summarize_group(group, level) for group in groups)
            )
            embeddings = await self.runtime.embed_many(
                [summary.text for summary in summaries],
                phase="memory",
            )
            for summary, embedding, group in zip(
                summaries, embeddings, groups, strict=True
            ):
                summary.embedding = embedding  # Final-level search vector.
                for child in group:
                    child.parent_id = summary.node_id  # Finish the tree link.
            all_nodes.extend(summaries)
            current = summaries  # Repeat until one root remains.

        self._max_level = max(node.level for node in all_nodes)  # Root level.
        records = [  # Qdrant payloads for leaves and summaries.
            {
                "id": node.node_id,
                "text": node.text,
                "ordinal": node.start_ordinal,
                "end_ordinal": node.end_ordinal,
                "level": node.level,
                "parent_id": node.parent_id,
                "child_ids": node.child_ids,
                "entry_kind": "raptor_leaf" if node.level == 0 else "raptor_summary",
            }
            for node in all_nodes
        ]
        await self._upsert_vector_records(
            records,
            [node.embedding for node in all_nodes],
        )

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Return global coverage or traverse relevant branches down the tree.

        if self._max_level == 0:
            if plan.mode in {"all", "global"}:
                return await self._retrieve_all()
            return await self._query_similar(query, limit=plan.top_k)

        if plan.mode in {"all", "global"}:
            levels = [self._max_level]  # Root gives the widest source view.
            if self._max_level > 0:
                levels.append(self._max_level - 1)
            return await self._retrieve_all(
                query_filter=Filter(
                    must=[FieldCondition(key="level", match=MatchAny(any=levels))]
                )
            )

        query_vector = await self.runtime.embed_one(query, phase="query")
        branch_width = max(2, min(6, math.ceil(math.sqrt(plan.top_k))))
        selected: list[RetrievedMemory] = []  # Nodes found across tree levels.

        current_level = self._max_level  # Begin at the root level.
        response = await self._qdrant_call(  # Best root-level branches.
            "query_points",
            lambda: self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=Filter(
                    must=[
                        FieldCondition(
                            key="level",
                            match=MatchValue(value=current_level),
                        )
                    ]
                ),
                limit=branch_width,
                with_payload=True,
                with_vectors=False,
            ),
        )
        frontier = [self._point_to_memory(point) for point in response.points]
        selected.extend(frontier)

        while current_level > 0 and frontier:
            parent_ids = [item.memory_id for item in frontier]  # Branches to follow.
            current_level -= 1
            child_limit = max(plan.top_k, branch_width * len(parent_ids))
            response = await self._qdrant_call(  # Children under this frontier.
                "query_points",
                lambda parents=parent_ids, limit=child_limit: self._client.query_points(
                    collection_name=self.collection_name,
                    query=query_vector,
                    query_filter=Filter(
                        must=[
                            FieldCondition(
                                key="parent_id",
                                match=MatchAny(any=parents),
                            )
                        ]
                    ),
                    limit=limit,
                    with_payload=True,
                    with_vectors=False,
                ),
            )
            frontier = [self._point_to_memory(point) for point in response.points]
            selected.extend(frontier[: max(plan.top_k, branch_width)])

        # RAPTOR also defines collapsed-tree retrieval: search all levels as one
        # index until the context budget is filled. Merging a compact collapsed
        # result with traversal improves robustness when the best evidence lies
        # outside an early branch choice, without another embedding request.
        collapsed = await self._qdrant_call(
            "query_points",
            lambda: self._client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                limit=max(plan.top_k, plan.top_k * plan.candidate_multiplier),
                with_payload=True,
                with_vectors=False,
            ),
        )
        selected.extend(self._point_to_memory(point) for point in collapsed.points)

        # Keep a compact multi-level context: the best summaries plus the most
        # relevant leaves. Duplicate text is removed without losing order data.
        selected.sort(
            key=lambda item: (
                item.score,
                int(item.metadata.get("level", 0)),
            ),
            reverse=True,
        )
        deduplicated: list[RetrievedMemory] = []  # Final mixed-level context.
        seen: set[str] = set()  # Stops repeated parent and child text.
        for item in selected:
            if item.text in seen:
                continue
            seen.add(item.text)
            deduplicated.append(item)
            if len(deduplicated) >= plan.top_k * 2:
                break
        return deduplicated

    def _cluster_nodes(self, nodes: list[_RaptorNode]) -> list[list[_RaptorNode]]:
        # Reduce embeddings with PCA, then select Gaussian-mixture clusters by BIC.

        count = len(nodes)  # Number of nodes at this tree level.
        if count <= self._max_cluster_size:
            return [nodes]

        embeddings = np.asarray([node.embedding for node in nodes], dtype=np.float32)
        n_components = max(2, min(10, count - 2))  # Reduced vector width.
        reduced = PCA(
            n_components=min(
                n_components,
                embeddings.shape[1],
                count - 1,
            ),
            random_state=self._random_seed,
        ).fit_transform(embeddings)

        expected = max(2, math.ceil(count / self._max_cluster_size))  # Rough count.
        upper = min(self._max_clusters, count - 1, max(expected * 2, 2))

        # Testing every component count is quadratic overhead on MemoryAgentBench
        # corpora with thousands of leaves. A compact BIC candidate set preserves
        # data-driven model selection while reducing dozens of GMM fits to at most
        # five. Diagonal covariance is substantially faster and stable in the
        # reduced PCA space used for clustering.
        candidate_components = sorted(  # Small set of cluster counts to compare.
            {
                1,
                max(2, min(upper, expected // 2)),
                max(2, min(upper, expected)),
                max(2, min(upper, math.ceil(expected * 1.5))),
                upper,
            }
        )
        best_model: GaussianMixture | None = None  # Lowest-BIC model found.
        best_bic = math.inf  # Any fitted model starts below this value.
        for components in candidate_components:
            model = GaussianMixture(  # Candidate grouping for this count.
                n_components=components,
                covariance_type="diag",
                random_state=self._random_seed,
                reg_covar=1e-6,
            )
            model.fit(reduced)
            bic = model.bic(reduced)  # Lower BIC gives the preferred fit.
            if bic < best_bic:
                best_bic = bic  # New lowest score.
                best_model = model  # Keep its fitted cluster model.
        assert best_model is not None
        labels = best_model.predict(reduced)  # Cluster number for every node.

        groups: dict[int, list[_RaptorNode]] = {}  # Label to source-ordered nodes.
        for label, node in zip(labels, nodes, strict=True):
            groups.setdefault(int(label), []).append(node)
        return [
            sorted(group, key=lambda node: node.start_ordinal)
            for group in groups.values()
        ]

    def _fit_groups_to_prompt(
        self,
        groups: list[list[_RaptorNode]],
    ) -> list[list[_RaptorNode]]:
        # Split large clusters without breaking an individual node.

        fitted: list[list[_RaptorNode]] = []  # Groups safe for summary prompts.
        for group in groups:
            current: list[_RaptorNode] = []  # Nodes in the group being filled.
            tokens = 350  # Leaves room for summary instructions.
            for node in group:
                size = self.runtime.llm_counter.count(node.text) + 25  # Text and label.
                if current and tokens + size > self._summary_input_tokens:
                    fitted.append(current)
                    current = []  # Start the next prompt group.
                    tokens = 350  # Reset its instruction allowance.
                current.append(node)
                tokens += size
            if current:
                fitted.append(current)
        return fitted

    async def _summarize_group(
        self,
        group: list[_RaptorNode],
        level: int,
    ) -> _RaptorNode:
        # Create one parent summary while keeping the source order.

        # Mark child IDs and source ranges so the summary keeps its place.
        source = "\n\n".join(
            f"[CHILD {node.node_id} | order {node.start_ordinal}-{node.end_ordinal}]\n{node.text}"
            for node in group
        )
        # Request one parent summary for this cluster.
        prompt = f"""
            Create a RAPTOR parent summary for the child nodes below. The benchmark context
            is the only source of truth. Preserve the cluster's important entities, exact
            facts, numbers, chronology, causal links, preferences, updates, conflicts, and
            plot events. The summary must support both high-level and multi-hop retrieval.
            Do not add external knowledge.
            
            Target at most {self._summary_output_tokens} tokens. Output only the summary.
            
            CHILD NODES:
            {source}
        """.strip()
        async with self._summary_semaphore:
            summary = await self.runtime.complete(prompt, phase="memory")
        return _RaptorNode(
            node_id=str(uuid.uuid4()),
            text=summary,
            embedding=[],
            level=level,
            start_ordinal=min(node.start_ordinal for node in group),
            end_ordinal=max(node.end_ordinal for node in group),
            child_ids=[node.node_id for node in group],
        )
