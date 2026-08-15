#############################################################################
# File: graph_agent.py
#
# Description:
#   Answers questions from a true directed knowledge graph. It presents
#   only walked edges and asks the model to verify every link used in the
#   answer.
#
#   - Selects graph depth and edge limits for direct, temporal,
#     conflict, and multi-hop questions.
#   - Orders retrieved edges by relevance and source position.
#   - Reduces all-edge queries while preserving direction, time, update
#     state, and paths.
#   - Applies active and superseded edge rules for current and
#     historical questions.
#   - Formats each edge as a directed relation with provenance.
#############################################################################

from __future__ import annotations

import time
from dataclasses import replace

from asdrp.agent.BaseAgent import AgentAnswer, BaseAgent
from asdrp.eval_schemas import EvaluationQuestion, RetrievalPlan, RetrievedMemory
from asdrp.memory.graph_memory import GraphMemoryBlock


class GraphAgent(BaseAgent):
    # Answer by following graph edges and checking each link in the path.
    #
    # The memory block finds likely entities and walks up to k graph links. This
    # agent reads the results as directed edges with update state and a source,
    # not as unrelated text pieces.

    memory_block_type = GraphMemoryBlock  # Stores and walks the task graph.

    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        # Retrieve a bounded subgraph and reason only over supported paths.

        plan, strategy = self._select_retrieval_plan(question, default_plan)
        memories, retrieval_seconds = await self._retrieve(question.text, plan)
        # Put the strongest graph edges first for the final prompt.
        ordered_memories = sorted(
            memories, key=lambda item: (-item.score, item.ordinal)
        )

        preparation_started = time.perf_counter()  # Starts the context timer.
        if plan.mode in {"all", "global"}:
            context, context_tokens = await self._reduce_global_context(
                question=question.text,
                memories=sorted(memories, key=lambda item: item.ordinal),
                final_token_budget=plan.max_context_tokens,
                formatter=self._format_edge,
                system_prompt=(
                    "You compress a benchmark knowledge graph into a faithful subgraph "
                    "description. The graph edges are the only source of truth."
                ),
                reduction_instruction=(
                    "Preserve entities, directed relations, edge statements, source order, "
                    "temporal qualifiers, active/superseded state, causal chains, and the "
                    "complete narrative or factual structure required by the question."
                ),
            )
        else:
            context, context_tokens = self._pack_context(
                ordered_memories,
                max_tokens=plan.max_context_tokens,
                formatter=self._format_edge,
            )
        preparation_seconds = time.perf_counter() - preparation_started  # Context time.

        # Explain how unnecessary and new versions of the same edge should be used.
        update_rule = (
            "For a current-state query, inactive/superseded edges are historical and an "
            "active later edge controls the answer."
            if plan.prefer_latest and not self._is_historical_question(question)
            else "Use active and historical edges according to the time requested."
        )
        # Set the graph reasoning rules.
        system_prompt = f"""
            You are the reasoning component of a TRUE KNOWLEDGE-GRAPH MEMORY agent.
            
            The retrieved records are directed edges from a task-local graph. To answer,
            identify the relevant start entity, follow only explicitly supplied relations,
            and verify every hop reaches the entity/value required by the question. Edge
            relevance scores select a subgraph but do not themselves prove a relation.
            
            Rules:
            - The graph is the only source of truth; ignore all outside knowledge.
            - Respect edge direction, relation labels, and entity identity.
            - {update_rule}
            - Do not infer a transitive relation unless the supplied path logically supports it.
            - Do not merge similarly named entities without an explicit graph connection.
            - For conflict questions, never average incompatible targets.
            - If no complete path supports an answer, state that it is not present.
            - Return only the final answer in the requested format.
        """.strip()
        # Add this question and its graph edges.
        user_prompt = f"""
            DATASET: {self.dataset}
            SOURCE FAMILY: {self.source}
            QUESTION TYPE: {question.metadata.get("question_type") or "unspecified"}
            GRAPH HOPS RETRIEVED: {plan.graph_hops}
            
            RETRIEVED SUBGRAPH EDGES:
            {context or "[No relevant graph path was retrieved]"}
            
            QUESTION:
            {question.text}
        """.strip()
        hypothesis, generation_seconds = await self._generate(  # Final model answer.
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return self._answer_result(
            hypothesis=hypothesis,
            memories=ordered_memories,
            context_tokens=context_tokens,
            retrieval_seconds=retrieval_seconds,
            preparation_seconds=preparation_seconds,
            generation_seconds=generation_seconds,
            plan=plan,
            strategy=strategy,
        )

    def _select_retrieval_plan(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> tuple[RetrievalPlan, str]:
        # Set graph depth from clear wording in the question.

        if default_plan.mode in {"all", "global"}:
            return default_plan, "all graph edges with relation-preserving reduction"

        hops = max(2, default_plan.graph_hops)  # Minimum path depth to search.
        top_k = default_plan.top_k  # Maximum number of graph records to return.
        reasons = [f"dataset default hops={hops}"]  # Explains the final plan.
        if self._is_multi_hop_question(question):
            hops = max(hops, 3)  # Allow one more link in the answer path.
            top_k = max(top_k, 24)  # Keep enough edges for several paths.
            reasons.append("multi-hop path expansion")
        if self._is_temporal_question(question):
            top_k = max(top_k, 24)  # Include unnecessary and new edge versions.
            reasons.append("versioned-edge coverage")
        if self.dataset == "mab_conflict_resolution":
            hops = max(hops, 3)  # Conflicts may connect through another entity.
            top_k = max(top_k, 40)  # The fact pool is much larger here.
            reasons.append("large update graph")
        if any(
            token in question.text.casefold()
            for token in ("through", "chain", "indirect")
        ):
            hops = max(hops, 4)  # The question asks for a longer path.
            reasons.append("explicit long path")

        return (
            replace(default_plan, graph_hops=hops, top_k=top_k),
            "; ".join(reasons),
        )

    @classmethod
    def _is_temporal_question(cls, question: EvaluationQuestion) -> bool:
        # Keep recall phrases like "previous conversation" from expanding retrieval.

        if cls._question_type(question) in {
            "temporal-reasoning",
            "knowledge-update",
            "temporal_update",
            "conflict_resolution",
        }:
            return True
        return GraphMemoryBlock._is_temporal_query(question.text)

    @staticmethod
    def _is_historical_question(question: EvaluationQuestion) -> bool:
        # Use the graph memory's conversation-aware history detector.

        return GraphMemoryBlock._is_historical_query(question.text)

    @staticmethod
    def _format_edge(memory: RetrievedMemory, index: int) -> str:
        # Render graph evidence as a directed triple plus the source statement.

        metadata = memory.metadata  # Holds the two entities and their relation.
        source = metadata.get("source_entity") or "unknown source"
        relation = metadata.get("relation") or "related_to"
        target = metadata.get("target_entity") or "unknown target"
        active = metadata.get("active", True)  # Current or superseded edge.
        return f"""
            [EDGE {index} | source_order={memory.ordinal} | relevance={memory.score:.4f} | active={active}]
            {source} --[{relation}]--> {target}
            Temporal qualifier: {metadata.get("temporal") or "unspecified"}
            Source record: {metadata.get("source_id") or memory.memory_id}
            Original statement: {memory.text}
        """.strip()
