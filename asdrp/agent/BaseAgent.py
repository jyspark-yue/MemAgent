#############################################################################
# File: BaseAgent.py
#
# Description:
#   Defines the shared answer flow used by each active memory agent. It
#   keeps common timing, context preparation, and question checks in one
#   place while each agent controls its own retrieval plan and prompt.
#
#   - Defines the standard answer record and its retrieval, context, and
#     generation measurements.
#   - Times memory retrieval and native OpenAI answer generation.
#   - Packs complete memory records into a fixed token budget and safely
#     shortens one oversized record.
#   - Reduces global evidence through question-aware map-reduce calls
#     when it does not fit in one prompt.
#   - Detects temporal, historical, and multi-hop questions from
#     benchmark labels and plain wording.
#   - Builds the final result shared by all agent implementations.
#############################################################################

from __future__ import annotations

import abc
import asyncio
import re
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import ClassVar

from asdrp.eval_schemas import (
    EvaluationQuestion,
    EvaluationTask,
    RetrievalPlan,
    RetrievedMemory,
)
from asdrp.memory.BaseMemBlock import BaseMemBlock
from asdrp.runtime import OpenAIRuntime

# A formatter turns one memory and its list number into prompt text.
MemoryFormatter = Callable[[RetrievedMemory, int], str]


@dataclass(slots=True, frozen=True)
class AgentAnswer:
    # This class holds the answer and the measurements saved with it.

    hypothesis: str  # The final answer returned by the model.
    retrieved_count: int  # The number of memories found for this question.
    retrieved_context_tokens: int  # The tokens kept in the final context.
    retrieval_seconds: float  # The time spent searching memory.
    context_preparation_seconds: float  # The time spent building the context.
    generation_seconds: float  # The time spent generating the answer.
    effective_retrieval_plan: RetrievalPlan  # The search limits that were used.
    retrieval_strategy: str  # A short label for the agent's search method.


class BaseAgent(abc.ABC):
    # This class holds the shared parts of each agent.
    #
    # The runner fills memory before it asks any questions. Answering a question
    # only reads that memory, so one answer cannot change the next answer.

    # Each child agent sets the memory class it needs.
    memory_block_type: ClassVar[type[BaseMemBlock]]

    def __init__(
        self,
        *,
        memory_block: BaseMemBlock,
        runtime: OpenAIRuntime,
        task: EvaluationTask,
    ) -> None:
        self.memory_block = memory_block  # Stores and searches the task context.
        self.runtime = runtime  # Makes model calls and counts their usage.
        self.dataset = task.dataset  # Keeps the dataset name for prompt rules.
        self.source = task.source  # Keeps the source name for prompt rules.

    @abc.abstractmethod
    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        # Answer one question with the child agent's own search steps.
        pass

    async def _retrieve(
        self,
        question: str,
        plan: RetrievalPlan,
    ) -> tuple[list[RetrievedMemory], float]:
        # Search this memory type and record how long it takes.

        started = time.perf_counter()  # Marks when the search starts.
        memories = await self.memory_block.get(question, plan)  # Finds matches.
        return memories, time.perf_counter() - started

    async def _generate(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
    ) -> tuple[str, float]:
        # Call the task-local LLM with a real system and user message.

        started = time.perf_counter()  # Marks when answer generation starts.
        # Send both prompt parts through the shared runtime.
        hypothesis = await self.runtime.chat(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            phase="query",
        )
        return hypothesis, time.perf_counter() - started

    def _pack_context(
        self,
        memories: list[RetrievedMemory],
        *,
        max_tokens: int,
        formatter: MemoryFormatter,
    ) -> tuple[str, int]:
        # Pack complete records until the final answer-context budget is full.
        #
        # Dataset adapters already keep embedding inputs within their limit. Keep
        # full memories here when possible. Only shorten the first memory when it
        # cannot fit on its own, so the answer still receives some context.

        if not memories or max_tokens <= 0:
            return "", 0

        parts: list[str] = []  # Holds each complete memory that still fits.
        used_tokens = 0  # Tracks the size of the context built so far.
        separator_tokens = 6  # Leaves room for the blank line between records.
        for index, memory in enumerate(memories, start=1):
            # Turn the memory into the format expected by this agent.
            rendered = formatter(memory, index).strip()
            if not rendered:
                continue
            rendered_tokens = self.runtime.llm_counter.count(rendered)  # Its size.
            required = rendered_tokens + (
                separator_tokens if parts else 0
            )  # Full cost.
            remaining = max_tokens - used_tokens  # Space left in the context.

            if required <= remaining:
                parts.append(rendered)
                used_tokens += required
                continue

            if not parts and remaining > 32:
                # Keep part of the best memory instead of returning no context.
                truncated = self.runtime.llm_counter.truncate(rendered, remaining)
                parts.append(truncated)
                used_tokens += self.runtime.llm_counter.count(truncated)
            break

        return "\n\n".join(parts), used_tokens

    async def _reduce_global_context(
        self,
        *,
        question: str,
        memories: list[RetrievedMemory],
        final_token_budget: int,
        formatter: MemoryFormatter,
        system_prompt: str,
        reduction_instruction: str,
        target_tokens_per_group: int = 1_600,
    ) -> tuple[str, int]:
        # Recursively reduce complete global evidence with agent-specific prompts.
        #
        # This helper only runs the repeated grouping work. Each agent explains
        # what its summaries must keep, such as event order, facts, or graph paths.

        # First check whether the memories already fit without a reduction call.
        packed, packed_tokens = self._pack_context(
            memories,
            max_tokens=final_token_budget,
            formatter=formatter,
        )
        total_tokens = sum(  # Counts all evidence, including text that did not fit.
            self.runtime.llm_counter.count(formatter(memory, index))
            for index, memory in enumerate(memories, start=1)
        )
        if total_tokens <= final_token_budget:
            return packed, packed_tokens

        # Keep each group well below the model limit. Separate groups can then run
        # at the same time under the evaluator's shared API limit.
        map_input_budget = min(40_000, max(8_000, final_token_budget // 2))
        # Keep the source order stable while the context becomes smaller.
        current = sorted(memories, key=lambda item: (item.ordinal, item.memory_id))
        level = 0  # Counts the number of reduction passes.

        while current:
            current_total = sum(  # Measures the output from the last pass.
                self.runtime.llm_counter.count(formatter(memory, index))
                for index, memory in enumerate(current, start=1)
            )
            if current_total <= final_token_budget:
                return self._pack_context(
                    current,
                    max_tokens=final_token_budget,
                    formatter=formatter,
                )

            # Split the current memories into groups that the model can handle.
            groups = self._pack_reduction_groups(
                current,
                token_budget=map_input_budget,
                formatter=formatter,
            )
            level += 1
            current_level = level  # Freezes this value for the async calls below.

            async def reduce_group(
                group_index: int,
                group: list[RetrievedMemory],
                reduction_level: int = current_level,
            ) -> RetrievedMemory:
                # Join this group's full records into one reduction prompt.
                source = "\n\n".join(
                    formatter(memory, index)
                    for index, memory in enumerate(group, start=1)
                )
                # Ask the model to keep only the evidence needed later.
                user_prompt = f"""
QUESTION THE FINAL ANSWER MUST SUPPORT:
{question}

REDUCTION REQUIREMENTS:
{reduction_instruction}

TARGET LENGTH:
At most {target_tokens_per_group} tokens.

MEMORY GROUP:
{source}
""".strip()
                # Reduce the group while keeping facts needed by the question.
                summary = await self.runtime.chat(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    phase="query",
                )
                return RetrievedMemory(
                    memory_id=f"query_reduction_{reduction_level}_{group_index}",
                    text=summary,
                    score=max((item.score for item in group), default=1.0),
                    ordinal=min(item.ordinal for item in group),
                    metadata={
                        "entry_kind": "query_reduction",
                        "level": reduction_level,
                        "source_count": len(group),
                    },
                )

            # Run independent groups together and keep their original order.
            reduced = list(
                await asyncio.gather(
                    *(
                        reduce_group(group_index, group)
                        for group_index, group in enumerate(groups)
                    )
                )
            )

            # Every ordinary pass combines at least two records.  This fallback
            # prevents an endless loop if one source record alone exceeds the map
            # budget or the settings create only one-record groups.
            if len(reduced) >= len(current):
                per_record = max(256, final_token_budget // max(1, len(reduced)))
                # Hard cuts guarantee that this stalled pass now fits.
                reduced = [
                    RetrievedMemory(
                        memory_id=item.memory_id,
                        text=self.runtime.llm_counter.truncate(item.text, per_record),
                        score=item.score,
                        ordinal=item.ordinal,
                        metadata=item.metadata,
                    )
                    for item in reduced
                ]
                return self._pack_context(
                    reduced,
                    max_tokens=final_token_budget,
                    formatter=formatter,
                )
            current = reduced  # Feed this smaller list into the next pass.

        return "", 0

    def _pack_reduction_groups(
        self,
        memories: list[RetrievedMemory],
        *,
        token_budget: int,
        formatter: MemoryFormatter,
    ) -> list[list[RetrievedMemory]]:
        # Pack chronological records into complete map/reduce input groups.

        groups: list[list[RetrievedMemory]] = []  # Holds each finished group.
        current: list[RetrievedMemory] = []  # Holds the group being filled.
        current_tokens = 350  # Leaves room for the reduction instructions.
        for memory in memories:
            rendered = formatter(memory, len(current) + 1)  # Formats this record.
            size = self.runtime.llm_counter.count(rendered) + 8  # Adds spacing.
            if current and current_tokens + size > token_budget:
                groups.append(current)
                current = []  # Start the next group with this memory.
                current_tokens = 350  # Reset the prompt allowance.
            current.append(memory)
            current_tokens += size
        if current:
            groups.append(current)

        if len(memories) > 1 and all(len(group) == 1 for group in groups):
            # Pair single records so each pass makes the list shorter.
            return [memories[index : index + 2] for index in range(0, len(memories), 2)]
        return groups

    @staticmethod
    def _answer_result(
        *,
        hypothesis: str,
        memories: list[RetrievedMemory],
        context_tokens: int,
        retrieval_seconds: float,
        preparation_seconds: float,
        generation_seconds: float,
        plan: RetrievalPlan,
        strategy: str,
    ) -> AgentAnswer:
        # Build the fixed result record returned to the evaluator.

        return AgentAnswer(
            hypothesis=hypothesis,
            retrieved_count=len(memories),
            retrieved_context_tokens=context_tokens,
            retrieval_seconds=retrieval_seconds,
            context_preparation_seconds=preparation_seconds,
            generation_seconds=generation_seconds,
            effective_retrieval_plan=plan,
            retrieval_strategy=strategy,
        )

    @staticmethod
    def _question_type(question: EvaluationQuestion) -> str:
        # Return the benchmark's normalized question-type label when present.

        return str(question.metadata.get("question_type") or "").casefold()

    @classmethod
    def _is_temporal_question(cls, question: EvaluationQuestion) -> bool:
        # Detect chronology/update language without an additional LLM call.

        if cls._question_type(question) in {
            "temporal-reasoning",
            "knowledge-update",
            "temporal_update",
            "conflict_resolution",
        }:
            return True
        return bool(
            re.search(
                r"\b(before|after|earlier|later|latest|newest|previous|first|last|"
                r"when|timeline|chronolog|changed|updated|used to|at the time)\b",
                question.text.casefold(),
            )
        )

    @classmethod
    def _is_multi_hop_question(cls, question: EvaluationQuestion) -> bool:
        # Detect questions likely to require combining multiple memory records.

        if cls._question_type(question) in {
            "multi-session",
            "multi_session_synthesis",
            "recommendation_from_memory",
            "conflict_resolution",
        }:
            return True
        return bool(
            re.search(
                r"\b(why|how|relationship|connect|combine|overall|based on|"
                r"recommend|in common|across|compare|sequence|followed)\b",
                question.text.casefold(),
            )
        )

    @staticmethod
    def _is_historical_question(question: EvaluationQuestion) -> bool:
        # Identify requests that intentionally need superseded facts retained.

        return bool(
            re.search(
                r"\b(before|earlier|previous|originally|first|used to|at the time|history)\b",
                question.text.casefold(),
            )
        )
