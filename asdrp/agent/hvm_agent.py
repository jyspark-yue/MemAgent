#############################################################################
# File: hvm_agent.py
#
# Description:
#   Answers questions from RAPTOR-style hierarchical vector memory. It
#   keeps summary nodes and source leaves distinct so broad context does
#   not replace exact evidence.
#
#   - Widens tree retrieval for multi-branch, temporal, and EventQA
#     questions.
#   - Places summaries or leaves first based on the scope of the
#     question.
#   - Applies current-state rules across hierarchy levels.
#   - Builds the final grounded prompt from node levels, ranges,
#     parents, and children.
#   - Returns the shared answer and timing measurements.
#############################################################################

from __future__ import annotations

import time
from dataclasses import replace

from asdrp.agent.BaseAgent import AgentAnswer, BaseAgent
from asdrp.eval_schemas import EvaluationQuestion, RetrievalPlan, RetrievedMemory
from asdrp.memory.hvm import HVMMemoryBlock


class HVMAgent(BaseAgent):
    # Use RAPTOR summaries together with the source leaves below them.
    #
    # HVM retrieval intentionally returns multiple levels: broad parent summaries
    # provide global orientation while lower-level nodes provide exact details.
    # The agent keeps those roles explicit so duplicated parent/child content is
    # not mistaken for independent corroboration.

    memory_block_type = HVMMemoryBlock  # Stores the source as a summary tree.

    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        # Traverse the hierarchy, organize results by level, and answer once.

        plan, strategy = self._select_retrieval_plan(question, default_plan)
        memories, retrieval_seconds = await self._retrieve(question.text, plan)
        ordered_memories = self._order_hierarchical_context(question, memories)

        preparation_started = time.perf_counter()  # Starts the context timer.
        context, context_tokens = self._pack_context(
            ordered_memories,
            max_tokens=plan.max_context_tokens,
            formatter=self._format_node,
        )
        preparation_seconds = time.perf_counter() - preparation_started  # Context time.

        # Tell the model whether a later state should replace an older one.
        latest_rule = (
            "For current-state conflicts, use the latest explicitly supported leaf or "
            "summary state; do not let a broad older summary override a newer leaf."
            if plan.prefer_latest and not self._is_historical_question(question)
            else "Preserve the time or version requested by the question."
        )
        # Set the rules for using summary tree levels.
        system_prompt = f"""
            You are the reasoning component of a RAPTOR HIERARCHICAL VECTOR MEMORY agent.
            
            Retrieved nodes come from several abstraction levels. Level 0 nodes are source
            leaves; larger level numbers are recursively clustered summaries. Use summaries
            to identify the relevant branch and overall relationships, but use the most
            specific available child/leaf evidence for exact names, values, dates, and event
            order. Parent and child text may repeat the same evidence and must not be counted
            as separate confirmations.
            
            Rules:
            - The hierarchy is the only source of truth; never use outside knowledge.
            - {latest_rule}
            - For broad synthesis, cover all supplied high-level branches before adding detail.
            - For direct questions, prefer exact lower-level evidence over generalized wording.
            - For multi-hop questions, connect levels only where the hierarchy supplies support.
            - If the hierarchy does not contain the answer, state that it is not present.
            - Follow the requested output constraints and return only the final answer.
        """.strip()
        # Add this question and the chosen tree nodes.
        user_prompt = f"""
            DATASET: {self.dataset}
            SOURCE FAMILY: {self.source}
            QUESTION TYPE: {question.metadata.get("question_type") or "unspecified"}
            
            RETRIEVED RAPTOR NODES:
            {context or "[No relevant hierarchy nodes were retrieved]"}
            
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
        # Tune traversal breadth without replacing the HVM block's tree search.

        if default_plan.mode in {"all", "global"}:
            return default_plan, "top hierarchy levels for global RAPTOR coverage"

        top_k = default_plan.top_k  # Number of nodes returned at the end.
        candidate_multiplier = default_plan.candidate_multiplier  # Search breadth.
        reasons = [f"dataset default top_k={top_k}"]  # Explains the final plan.
        if self._is_multi_hop_question(question):
            top_k = max(top_k, 16)  # Return nodes from several branches.
            candidate_multiplier = max(candidate_multiplier, 5)  # Search wider first.
            reasons.append("broader multi-branch traversal")
        if self._is_temporal_question(question):
            top_k = max(top_k, 16)  # Keep summary and leaf versions of events.
            candidate_multiplier = max(candidate_multiplier, 5)  # Search more nodes.
            reasons.append("summary-plus-leaf temporal coverage")
        if self.source.startswith("eventqa"):
            top_k = max(top_k, 18)  # Small book events need more leaf coverage.
            reasons.append("fine-grained event branch coverage")

        return (
            replace(
                default_plan,
                top_k=top_k,
                candidate_multiplier=candidate_multiplier,
            ),
            "; ".join(reasons),
        )

    def _order_hierarchical_context(
        self,
        question: EvaluationQuestion,
        memories: list[RetrievedMemory],
    ) -> list[RetrievedMemory]:
        # Put abstraction or exact evidence first according to question scope.

        # Broad questions need parent summaries before exact leaf details.
        broad = self._is_multi_hop_question(question) or any(
            token in question.text.casefold()
            for token in ("summarize", "overall", "entire", "main points", "whole")
        )
        if broad:
            return sorted(
                memories,
                key=lambda item: (
                    -int(item.metadata.get("level", 0)),
                    item.ordinal,
                    -item.score,
                ),
            )
        return sorted(
            memories,
            key=lambda item: (
                int(item.metadata.get("level", 0)),
                -item.score,
                item.ordinal,
            ),
        )

    @staticmethod
    def _format_node(memory: RetrievedMemory, index: int) -> str:
        # Expose hierarchy level, child links, and represented source range.

        metadata = memory.metadata  # Holds this node's place in the tree.
        level = int(metadata.get("level", 0))  # Leaves sit at level zero.
        kind = "LEAF" if level == 0 else "SUMMARY"  # Clear prompt label.
        child_ids = metadata.get("child_ids") or []  # Direct nodes below this one.
        children = len(child_ids) if isinstance(child_ids, list) else "unknown"
        end_ordinal = metadata.get("end_ordinal", memory.ordinal)
        return f"""
            [RAPTOR NODE {index} | kind={kind} | level={level} | relevance={memory.score:.4f}]
            Source range: {memory.ordinal}-{end_ordinal}
            Parent ID: {metadata.get("parent_id") or "root/unknown"}
            Child count: {children}
            Content: {memory.text}
        """.strip()
