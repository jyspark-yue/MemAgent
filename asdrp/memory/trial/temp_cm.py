#############################################################################
# File: condensed_memory.py
#
# Description:
#   Builds a recursively condensed memory and stores only its final
#   summary units. This provides global coverage without retaining the
#   full RAPTOR hierarchy.
#
#   - Groups nearby source ranges under a model-safe input budget.
#   - Runs parallel chronological map summaries and repeats reduction
#     until the target size is reached.
#   - Stops safely if a reduction pass cannot make the unit list
#     smaller.
#   - Stores final ranges, levels, and summary text in task-local
#     Qdrant.
#   - Returns either all final summaries or a direct dense subset.
#
# Authors:
#   @author     Eric Vincent Fernandes
#
# Date:
#   Modified:   August 7, 2026 (Eric Vincent Fernandes)
#############################################################################

from __future__ import annotations

import asyncio
import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseQdrantMemoryBlock


@dataclass(slots=True)
class _SummaryUnit:
    # Internal unit used while recursively reducing the source context.

    text: str  # Summary or original source text.
    start_ordinal: int  # First source entry covered by the unit.
    end_ordinal: int  # Last source entry covered by the unit.
    level: int  # Number of summary passes above the source.


class CondensedMemoryBlock(BaseQdrantMemoryBlock):
    # Store the full context as a limited set of repeated summaries.
    #
    # The implementation performs parallel map summaries and then repeatedly
    # summarizes groups of summaries until the retained representation fits a
    # configurable global budget. Only the final condensed units are stored,
    # making it different from HVM/RAPTOR, which keeps source leaves and each
    # summary level.

    memory_name = "condensed"  # Prefix for this task's Qdrant collection.

    def __init__(
        self,
        *,
        summary_input_tokens=24_000,
        summary_output_tokens=700,
        target_total_tokens=16_000,
        max_final_units=16,
        summary_concurrency=4,
        **kwargs: Any,
    ) -> None:
        if summary_input_tokens < 1 or summary_output_tokens < 1:
            raise ValueError("summary token limits must be at least 1")
        if target_total_tokens < 1 or max_final_units < 1:
            raise ValueError("final summary limits must be at least 1")
        if summary_concurrency < 1:
            raise ValueError("summary_concurrency must be at least 1")
        super().__init__(**kwargs)
        self._summary_input_tokens = summary_input_tokens  # Limit for one group.
        self._summary_output_tokens = summary_output_tokens  # Requested summary size.
        self._target_total_tokens = target_total_tokens  # Limit across final units.
        self._max_final_units = max_final_units  # Maximum summaries left at the end.
        self._summary_semaphore = asyncio.Semaphore(summary_concurrency)

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Repeatedly summarize the source, then store the final units.

        # Start with one level-zero unit for each non-empty source entry.
        units = [
            _SummaryUnit(
                text=entry.text,
                start_ordinal=entry.ordinal,
                end_ordinal=entry.ordinal,
                level=0,
            )
            for entry in entries
            if entry.text.strip()
        ]
        if not units:
            return

        level = 1  # First pass turns raw entries into summaries.
        units = await self._summarize_level(units, level)  # First summary pass.
        while self._requires_reduction(units):
            previous_count = len(units)  # Used to catch a stalled reduction.
            level += 1
            units = await self._summarize_level(units, level)  # Next smaller level.
            # Defensive stop: malformed model output should never create an
            # endless reduction loop. Normal grouping strictly decreases count.
            if len(units) >= previous_count:
                break

        # Only final units are stored; source and middle levels are discarded.
        records = [
            {
                "id": str(uuid.uuid4()),
                "text": unit.text,
                "ordinal": unit.start_ordinal,
                "end_ordinal": unit.end_ordinal,
                "level": unit.level,
                "entry_kind": "condensed_summary",
            }
            for unit in units
        ]
        await self._store_text_records(records)

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Return all global summaries or the most relevant condensed units.

        if plan.mode in {"all", "global"}:
            return await self._retrieve_all()
        return await self._query_similar(query, limit=plan.top_k)

    def _requires_reduction(self, units: Sequence[_SummaryUnit]) -> bool:
        # Check both the count cap and the total retained token budget.

        total_tokens = sum(self.runtime.llm_counter.count(unit.text) for unit in units)
        return (
            len(units) > self._max_final_units
            or total_tokens > self._target_total_tokens
        )

    async def _summarize_level(
        self,
        units: Sequence[_SummaryUnit],
        level: int,
    ) -> list[_SummaryUnit]:
        # Pack nearby units and summarize the groups at the same time.

        groups = self._pack_adjacent(units)  # Neighboring source ranges stay together.

        async def summarize(group: list[_SummaryUnit]) -> _SummaryUnit:
            async with self._summary_semaphore:
                # Mark each source range so order remains clear in the prompt.
                source = "\n\n".join(
                    f"[SOURCE {unit.start_ordinal}-{unit.end_ordinal}]\n{unit.text}"
                    for unit in group
                )
                # Request one faithful summary for this group.
                prompt = f"""
You are building a long-term condensed memory from benchmark-provided context.
The supplied context is the only source of truth, even when it contradicts
real-world knowledge.

Create a faithful chronological summary of the source below. Preserve exact
names, numbers, dates, preferences, decisions, causal links, plot events,
examples, labels, and explicit conflicts. When facts change, preserve both the
old and new values and clearly identify which came later. Do not add outside
knowledge. Do not omit details merely because they seem unusual.

Target at most {self._summary_output_tokens} tokens. Output only the summary.

SOURCE:
{source}
""".strip()
                summary = await self.runtime.complete(prompt, phase="memory")
                return _SummaryUnit(
                    text=summary,
                    start_ordinal=group[0].start_ordinal,
                    end_ordinal=group[-1].end_ordinal,
                    level=level,
                )

        return list(await asyncio.gather(*(summarize(group) for group in groups)))

    def _pack_adjacent(self, units: Sequence[_SummaryUnit]) -> list[list[_SummaryUnit]]:
        # Pack chronological units below the model-safe summary input budget.

        groups: list[list[_SummaryUnit]] = []  # Finished groups for this pass.
        current: list[_SummaryUnit] = []  # Units in the group being filled.
        current_tokens = 0  # Size of the current group.
        fixed_prompt_allowance = 350  # Space for the summary instructions.

        for unit in units:
            unit_tokens = (
                self.runtime.llm_counter.count(unit.text) + 12
            )  # Text and label.
            would_exceed = (
                current
                and current_tokens + unit_tokens + fixed_prompt_allowance
                > self._summary_input_tokens
            )
            if would_exceed:
                groups.append(current)
                current = []  # Start a new group with this unit.
                current_tokens = 0  # Reset its text count.
            current.append(unit)
            current_tokens += unit_tokens

        if current:
            groups.append(current)

        # A reduction level must combine units. If token packing produced only
        # singleton groups, pair adjacent units and let the prompt remain close
        # to, rather than far beyond, the configured budget.
        if len(units) > 1 and all(len(group) == 1 for group in groups):
            groups = [  # Pair neighbors so the level becomes smaller.
                list(units[index : index + 2]) for index in range(0, len(units), 2)
            ]
        return groups
