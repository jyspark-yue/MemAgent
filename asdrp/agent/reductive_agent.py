#############################################################################
# File: reductive_agent.py
#
# Description:
#   Answers questions from atomic proposition memory. It joins only
#   supported subject-relation-object facts and keeps changed values in
#   source order.
#
#   - Expands fact retrieval for multi-hop, temporal, and large conflict
#     tasks.
#   - Orders current facts newest-first while retaining chronological
#     order for history questions.
#   - Reduces global proposition sets while preserving distinct facts
#     and conflicts.
#   - Applies fact-key rules for current and historical values.
#   - Formats retrieved evidence compactly so final answer prompts do not
#     repeat the same proposition across several redundant fields.
#############################################################################

from __future__ import annotations

import time
from dataclasses import replace

from asdrp.agent.BaseAgent import AgentAnswer, BaseAgent
from asdrp.eval_schemas import EvaluationQuestion, RetrievalPlan, RetrievedMemory
from asdrp.memory.proposition_extraction_memory import PropositionMemoryBlock


class ReductiveAgent(BaseAgent):
    # Answer by selecting and joining facts taken from the source.
    #
    # Proposition memory stores independently retrievable subject-relation-object
    # statements and retains version history.  This agent therefore reasons over
    # fact keys, sources, time notes, and source order rather than
    # treating the results as conversational excerpts.

    memory_block_type = PropositionMemoryBlock  # Stores small standalone facts.

    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        # Retrieve atomic facts and resolve or preserve versions as requested.

        plan, strategy = self._select_retrieval_plan(question, default_plan)
        memories, retrieval_seconds = await self._retrieve(question.text, plan)

        # Current-state facts are easiest to inspect newest first; historical
        # and sequence questions retain chronological order.
        if plan.prefer_latest and not self._is_historical_question(question):
            ordered_memories = sorted(
                memories, key=lambda item: (item.ordinal, item.score), reverse=True
            )
        else:
            ordered_memories = sorted(
                memories, key=lambda item: (item.ordinal, -item.score)
            )

        preparation_started = time.perf_counter()  # Starts the context timer.
        if plan.mode in {"all", "global"}:
            context, context_tokens = await self._reduce_global_context(
                question=question.text,
                memories=ordered_memories,
                final_token_budget=plan.max_context_tokens,
                formatter=self._format_proposition,
                system_prompt=(
                    "You consolidate atomic benchmark propositions for a later answer. "
                    "Every supplied proposition is source evidence and no outside fact "
                    "may be introduced."
                ),
                reduction_instruction=(
                    "Preserve every distinct entity, relation, value, date, negative fact, "
                    "conflict, version change, causal dependency, and plot event. Merge "
                    "only exact redundancy and keep the ordering of changed values."
                ),
            )
        else:
            context, context_tokens = self._pack_context(
                ordered_memories,
                max_tokens=plan.max_context_tokens,
                formatter=self._format_proposition,
            )
        preparation_seconds = time.perf_counter() - preparation_started  # Context time.

        # Tell the model whether it should use the latest version of a fact.
        conflict_rule = (
            "For the same fact key, the proposition with the greatest source order is "
            "authoritative for the current state. Do not average or merge incompatible values."
            if plan.prefer_latest and not self._is_historical_question(question)
            else (
                "Retain proposition versions separately and select the version at "
                "the requested time."
            )
        )

        # Keep the final reasoning instructions strict but compact.
        system_prompt = f"""
            Answer using only the retrieved atomic propositions.
            
            Rules:
            - Treat the propositions as the only source of truth.
            - {conflict_rule}
            - Preserve negative, unusual, and false-looking source claims.
            - For multi-hop questions, verify every hop from supplied propositions.
            - Do not invent missing bridge facts or use outside knowledge.
            - If the supplied propositions do not support an answer, say the answer is not present.
            - Return only the final answer in the requested format.
        """.strip()

        # Add this question and the facts found for it.
        user_prompt = f"""
            DATASET: {self.dataset}
            SOURCE FAMILY: {self.source}
            QUESTION TYPE: {question.metadata.get("question_type") or "unspecified"}
            
            RETRIEVED PROPOSITIONS:
            {context or "[No relevant propositions were retrieved]"}
            
            QUESTION:
            {question.text}
        """.strip()

        # Final model answer.
        hypothesis, generation_seconds = await self._generate(
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
        # Allocate enough atomic facts for conflicts and supported multi-hop chains.

        if default_plan.mode in {"all", "global"}:
            return default_plan, "all propositions with fact-preserving reduction"

        top_k = default_plan.top_k  # Number of facts returned at the end.
        candidate_multiplier = max(default_plan.candidate_multiplier, 5)
        reasons = [f"dataset default top_k={top_k}"]  # Explains the final plan.
        if self._is_multi_hop_question(question):
            top_k = max(top_k, 24)  # Keep enough facts to form a full chain.
            candidate_multiplier = max(candidate_multiplier, 6)  # Search wider first.
            reasons.append("multi-hop fact-chain coverage")
        if self._is_temporal_question(question):
            top_k = max(top_k, 24)  # Include unnecessary and new values.
            candidate_multiplier = max(candidate_multiplier, 8)  # Search more versions.
            reasons.append("version-history coverage")
        if self.dataset == "mab_conflict_resolution":
            top_k = max(top_k, 48)  # Conflict tasks have large fact pools.
            candidate_multiplier = max(candidate_multiplier, 10)  # Recall matters here.
            reasons.append("5k-10k conflict pool")

        return (
            replace(
                default_plan,
                top_k=top_k,
                candidate_multiplier=candidate_multiplier,
                neighbor_window=0,
            ),
            "; ".join(reasons),
        )

    @staticmethod
    def _format_proposition(memory: RetrievedMemory, index: int) -> str:
        # Keep the self-contained statement plus only metadata needed for versions.

        metadata = memory.metadata  # Holds the fact key and optional time note.
        fact_key = metadata.get("fact_key") or "unkeyed"  # Groups changed values.
        temporal = str(metadata.get("temporal") or "").strip()  # Optional time detail.
        time_suffix = f" | time={temporal}" if temporal else ""
        return (
            f"[P{index} | order={memory.ordinal} | key={fact_key}{time_suffix}] "
            f"{memory.text}"
        )
