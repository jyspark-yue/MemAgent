#############################################################################
# File: binary_router_agent.py
#
# Description:
#   Delegates question answering to the existing agent that corresponds to the
#   memory block chosen before question time by BinaryRouterMemoryBlock.
#
#   - Reuses HVMAgent and EpisodicAgent without changing either implementation.
#   - Never reclassifies or changes the selected memory after ingestion.
#   - Exposes the context-only routing decision to evaluator diagnostics.
#############################################################################

from __future__ import annotations

from typing import Any

from asdrp.agent.BaseAgent import AgentAnswer, BaseAgent
from asdrp.agent.episodic_agent import EpisodicAgent
from asdrp.agent.hvm_agent import HVMAgent
from asdrp.eval_schemas import EvaluationQuestion, EvaluationTask, RetrievalPlan
from asdrp.memory.binary_router_memory import BinaryRouterMemoryBlock
from asdrp.runtime import OpenAIRuntime


class BinaryRouterAgent(BaseAgent):
    memory_block_type = BinaryRouterMemoryBlock

    def __init__(
        self,
        *,
        memory_block: BinaryRouterMemoryBlock,
        runtime: OpenAIRuntime,
        task: EvaluationTask,
    ) -> None:
        if not isinstance(memory_block, BinaryRouterMemoryBlock):
            raise TypeError("BinaryRouterAgent requires BinaryRouterMemoryBlock")
        if task.dataset != "longmemeval":
            raise ValueError("Binary routing is currently defined only for LongMemEval")
        super().__init__(memory_block=memory_block, runtime=runtime, task=task)
        self._router_memory = memory_block

        selected = memory_block.selected_memory_name
        if selected == "hvm":
            delegate_type = HVMAgent
        elif selected == "episodic":
            delegate_type = EpisodicAgent
        else:
            raise ValueError(f"Unsupported routed agent destination: {selected!r}")

        # The evaluator constructs this object only after memory.put() has completed.
        self._delegate = delegate_type(
            memory_block=memory_block.selected_backend,
            runtime=runtime,
            task=task,
        )

    async def answer(
        self,
        question: EvaluationQuestion,
        default_plan: RetrievalPlan,
    ) -> AgentAnswer:
        return await self._delegate.answer(question, default_plan)

    @property
    def routing_metadata(self) -> dict[str, Any] | None:
        return self._router_memory.routing_metadata
