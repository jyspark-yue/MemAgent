#############################################################################
# File: summary_agent.py
#
# Description:
#   Answers questions from recursively condensed memory. It uses all
#   retained summaries for broad questions and a small dense subset for
#   direct lookup.
#
#   - Selects global coverage for temporal and cross-record questions.
#   - Orders summary units by their original source ranges.
#   - Applies current-state rules to unnecessary and new values retained in
#     summaries.
#   - Builds a grounded answer prompt without repeating ingestion-time
#     summarization.
#   - Formats each summary with its range, level, and relevance score.
#
# Authors:
#   @author     Eric Vincent Fernandes
#
# Date:
#   Modified:   August 7, 2026 (Eric Vincent Fernandes)
#############################################################################

from __future__ import annotations

import time
from dataclasses import replace

from asdrp.agent.BaseAgent import AgentAnswer, BaseAgent
from asdrp.eval_schemas import EvaluationQuestion, RetrievalPlan, RetrievedMemory
from memory.condensed_memory import CondensedMemoryBlock


class SummaryAgent(BaseAgent):
    # Answer from a smaller summary of the full source context.
    #
    # Condensed memory is intentionally lossy but globally aware.  The agent uses
    # all retained summary units for broad or multi-session questions and a small
    # dense subset for direct questions.  It does not perform the raw-memory
    # agent's query-time summary step because this memory was already condensed
    # when it was stored.

    memory_block_type = CondensedMemoryBlock  # Stores the final source summaries.

    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        # Retrieve the appropriate summary coverage and reconcile summary ranges.

        plan, strategy = self._select_retrieval_plan(question, default_plan)
        memories, retrieval_seconds = await self._retrieve(question.text, plan)
        # Keep the summaries in the order of the source ranges they cover.
        ordered_memories = sorted(
            memories,
            key=lambda item: (
                item.ordinal,
                int(item.metadata.get("end_ordinal", item.ordinal)),
            ),
        )

        preparation_started = time.perf_counter()  # Starts the context timer.
        context, context_tokens = self._pack_context(
            ordered_memories,
            max_tokens=plan.max_context_tokens,
            formatter=self._format_summary,
        )
        preparation_seconds = time.perf_counter() - preparation_started  # Context time.

        # Explain how unnecessary and new values inside a summary should be used.
        latest_rule = (
            "When a summary preserves an unnecessary and a new value, use the later value for "
            "a current-state question and the earlier value only when explicitly asked."
            if plan.prefer_latest and not self._is_historical_question(question)
            else "Respect every chronological state retained in the summaries."
        )
        # Set the rules for using condensed memory.
        system_prompt = f"""
You are the reasoning component of a CONDENSED MEMORY agent.

The memory consists of recursively produced chronological summaries. Each unit
may cover many source entries, so reason across its stated source range rather
than treating every sentence as a separate original observation. The summary
may omit incidental wording; never fill those gaps with outside knowledge.

Rules:
- The condensed memory is the only source of truth, even if it contradicts reality.
- {latest_rule}
- Reconcile overlapping summary units and avoid counting repeated summarized facts twice.
- Prefer explicit retained details over plausible inferences.
- For broad questions, synthesize across all supplied summary ranges.
- If a requested detail was not preserved, state that it is not present.
- Follow the requested format/length and return only the final answer.
""".strip()
        # Add this question and its summary units.
        user_prompt = f"""
DATASET: {self.dataset}
SOURCE FAMILY: {self.source}
QUESTION TYPE: {question.metadata.get("question_type") or "unspecified"}

CONDENSED MEMORY UNITS:
{context or "[No relevant condensed memory was retrieved]"}

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
        # Use global summary coverage when the question spans multiple records.

        if default_plan.mode in {"all", "global"}:
            return replace(default_plan, mode="all"), "all final condensed units"

        if self._is_multi_hop_question(question) or self._is_temporal_question(
            question
        ):
            return (
                replace(default_plan, mode="all"),
                "all final summaries for cross-range synthesis/update reasoning",
            )

        # Direct lookup needs only a few dense units.  Capping rather than
        # expanding top-k keeps direct condensed-memory lookups fast.
        top_k = min(max(default_plan.top_k, 4), 12)  # Keeps direct lookups small.
        return replace(default_plan, top_k=top_k), f"dense summary lookup top_k={top_k}"

    @staticmethod
    def _format_summary(memory: RetrievedMemory, index: int) -> str:
        # Show the chronology range represented by each recursive summary.

        start = memory.ordinal  # First source entry covered by this summary.
        end = memory.metadata.get("end_ordinal", start)  # Last covered entry.
        level = memory.metadata.get("level", "final")  # Reduction pass label.
        return f"""
[CONDENSED UNIT {index} | source_range={start}-{end} | level={level} | relevance={memory.score:.4f}]
{memory.text}
""".strip()
