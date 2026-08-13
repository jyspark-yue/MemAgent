#############################################################################
# File: binary_router_memory.py
#
# Description:
#   Routes one complete task context to either HVM or episodic memory before any
#   evaluation question is exposed to the agent.
#
#   - Loads the fitted v4 session-pooled TF-IDF linear router once per process
#     through the shared model cache.
#   - Classifies only evaluator-prepared MemoryEntry context.
#   - Instantiates, initializes, and builds exactly one existing memory block.
#   - Delegates retrieval and cleanup unchanged to the selected memory block.
#   - Exposes routing diagnostics without modifying HVM or episodic memory.
#############################################################################

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

from asdrp.classification_algorithms.binary_router import (
    RoutingDecision,
    load_binary_router,
)
from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.memory.BaseMemBlock import BaseMemBlock
from asdrp.memory.episodic_memory import EpisodicMemoryBlock
from asdrp.memory.hvm import HVMMemoryBlock
from asdrp.runtime import OpenAIRuntime


_BACKENDS: dict[str, type[BaseMemBlock]] = {
    "hvm": HVMMemoryBlock,
    "episodic": EpisodicMemoryBlock,
}


class BinaryRouterMemoryBlock(BaseMemBlock):
    memory_name = "binary_router"

    def __init__(
        self,
        *,
        task_id: str,
        runtime: OpenAIRuntime,
        router_model_path: Path | str,
        qdrant_url: str,
        qdrant_timeout: int = 120,
        qdrant_batch_size: int = 256,
        ingest_concurrency: int = 4,
        retry_attempts: int = 8,
        retry_base_delay: float = 1.0,
        retry_max_delay: float = 20.0,
        delete_collection_on_close: bool = True,
    ) -> None:
        super().__init__(task_id=task_id, runtime=runtime)
        model_path = Path(router_model_path).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Binary router model does not exist: {model_path}")
        self._model_path = model_path
        self._backend_kwargs: dict[str, Any] = {
            "task_id": task_id,
            "runtime": runtime,
            "qdrant_url": qdrant_url,
            "qdrant_timeout": qdrant_timeout,
            "qdrant_batch_size": qdrant_batch_size,
            "ingest_concurrency": ingest_concurrency,
            "retry_attempts": retry_attempts,
            "retry_base_delay": retry_base_delay,
            "retry_max_delay": retry_max_delay,
            "delete_collection_on_close": delete_collection_on_close,
        }
        self._router = None
        self._backend: BaseMemBlock | None = None
        self._decision: RoutingDecision | None = None
        self._router_seconds = 0.0

    async def initialize(self) -> None:
        # Load/validate the small local model now so failures happen before ingestion.
        self._router = load_binary_router(str(self._model_path))

    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        if self._router is None:
            raise RuntimeError("Binary router memory must be initialized before put()")
        if self._backend is not None:
            raise RuntimeError("Binary router memory can only be built once per task")

        route_start = time.perf_counter()
        decision = self._router.predict_entries(entries)
        self._router_seconds = time.perf_counter() - route_start
        self._decision = decision

        try:
            backend_type = _BACKENDS[decision.selected_memory]
        except KeyError as error:
            raise ValueError(
                f"Router selected unsupported memory {decision.selected_memory!r}"
            ) from error

        backend = backend_type(**self._backend_kwargs)
        self._backend = backend
        print(
            f"[binary-router] {decision.predicted_label} -> {decision.selected_memory} | "
            f"score={decision.decision_score:+.4f} | "
            f"route={self._router_seconds * 1000.0:.1f}ms",
            flush=True,
        )
        await backend.initialize()
        await backend.put(entries)

    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        if self._backend is None:
            raise RuntimeError(
                "Binary router has not selected and built a memory block"
            )
        return await self._backend.get(query, plan)

    async def close(self) -> None:
        if self._backend is not None:
            await self._backend.close()
            self._backend = None

    @property
    def selected_backend(self) -> BaseMemBlock:
        if self._backend is None:
            raise RuntimeError("Binary router has not selected a memory block")
        return self._backend

    @property
    def selected_memory_name(self) -> str:
        if self._decision is None:
            raise RuntimeError("Binary router has not made a routing decision")
        return self._decision.selected_memory

    @property
    def routing_metadata(self) -> dict[str, Any] | None:
        if self._decision is None:
            return None
        return {
            "predicted_question_type": self._decision.predicted_label,
            "selected_memory": self._decision.selected_memory,
            "decision_score": self._decision.decision_score,
            "decision_score_positive_label": self._decision.positive_label,
            "classifier_name": getattr(self._router, "classifier_name", None),
            "router_seconds": self._router_seconds,
            "sampled_session_count": self._decision.sampled_session_count,
            "sampled_character_count": self._decision.sampled_character_count,
            "model_path": str(self._model_path),
        }
