#############################################################################
# File: BaseMemBlock.py
#
# Description:
#   Defines the common memory interfaces and Qdrant storage operations. It
#   keeps collection lifecycle, embedding, upsert, retrieval, and retry
#   logic out of the individual architectures.
#
#   - Defines the initialize, put, get, and close contract used by every
#     memory block.
#   - Creates a unique task-local Qdrant collection and optionally
#     deletes it after the task.
#   - Embeds and upserts records in bounded concurrent batches with
#     stable point IDs.
#   - Supports precomputed vectors for hierarchical memory nodes.
#   - Runs similarity queries, complete collection scans, and
#     source-neighbor lookup.
#   - Normalizes Qdrant payloads into the shared retrieved-memory
#     record.
#   - Retries short-lived Qdrant failures without restarting the full
#     ingestion job.
#############################################################################

from __future__ import annotations

import abc
import asyncio
import re
import uuid
from collections.abc import Awaitable, Callable, Iterable, Sequence
from typing import Any, TypeVar

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from asdrp.eval_schemas import MemoryEntry, RetrievalPlan, RetrievedMemory
from asdrp.qdrant_retry import retry_qdrant
from asdrp.runtime import OpenAIRuntime

# Keeps the return type of a retried Qdrant call unchanged.
T = TypeVar("T")


class BaseMemBlock(abc.ABC):
    # Defines the small set of actions every memory class supports.
    #
    # Only four public operations are required: initialize, put, get, and close.
    # Dataset formatting and output files stay outside the memory classes.

    memory_name = "base"  # Child classes use this in their collection names.

    def __init__(self, *, task_id: str, runtime: OpenAIRuntime) -> None:
        self.task_id = task_id  # Identifies the current benchmark task.
        self.runtime = runtime  # Handles model calls, embeddings, and usage.

    @abc.abstractmethod
    async def initialize(self) -> None:
        # Create task-local storage before any context is ingested.
        pass

    @abc.abstractmethod
    async def put(self, entries: Sequence[MemoryEntry]) -> None:
        # Transform and store all runner-prepared context entries.
        pass

    @abc.abstractmethod
    async def get(self, query: str, plan: RetrievalPlan) -> list[RetrievedMemory]:
        # Retrieve the context required to answer one independent question.
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        # Release storage and delete task-local state.
        pass


class BaseQdrantMemoryBlock(BaseMemBlock):
    # Holds the Qdrant setup and vector work shared by memory classes.

    vector_size = 1536  # Size returned by the chosen embedding model.

    def __init__(
        self,
        *,
        task_id: str,
        runtime: OpenAIRuntime,
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

        if qdrant_batch_size < 1:
            raise ValueError("qdrant_batch_size must be at least 1")
        if ingest_concurrency < 1:
            raise ValueError("ingest_concurrency must be at least 1")
        if retry_attempts < 1:
            raise ValueError("retry_attempts must be at least 1")
        if retry_base_delay < 0:
            raise ValueError("retry_base_delay cannot be negative")
        if retry_max_delay < 0:
            raise ValueError("retry_max_delay cannot be negative")

        # Collection names only use characters accepted by Qdrant.
        safe_task_id = re.sub(
            r"[^a-zA-Z0-9_-]+",
            "_",
            task_id,
        )[:48]

        # A random suffix keeps concurrent tasks from sharing a collection.
        self.collection_name = (
            f"{self.memory_name}_{safe_task_id}_{uuid.uuid4().hex[:12]}"
        ).lower()

        self._client = AsyncQdrantClient(  # Async client for this task only.
            url=qdrant_url,
            timeout=qdrant_timeout,
        )
        self._qdrant_batch_size = qdrant_batch_size  # Records per upsert.
        self._ingest_concurrency = ingest_concurrency  # Batches sent at once.
        self._retry_attempts = retry_attempts  # Tries for one network call.
        self._retry_base_delay = retry_base_delay  # First retry wait.
        self._retry_max_delay = retry_max_delay  # Longest retry wait.
        self._delete_collection_on_close = delete_collection_on_close  # Cleanup choice.
        self._initialized = False  # Tracks whether the collection was created.

    async def _qdrant_call(
        self,
        operation_name: str,
        operation: Callable[[], Awaitable[T]],
    ) -> T:
        # Execute one retryable Qdrant network operation.

        return await retry_qdrant(
            operation,
            operation_name=operation_name,
            attempts=self._retry_attempts,
            base_delay=self._retry_base_delay,
            max_delay=self._retry_max_delay,
        )

    async def initialize(self) -> None:
        # Create one unique collection for exactly one evaluation task.

        await self._qdrant_call(
            "create_collection",
            lambda: self._client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.vector_size,
                    distance=Distance.COSINE,
                ),
            ),
        )
        self._initialized = True  # Close may now delete this collection.

    async def close(self) -> None:
        # Delete the isolated collection and close the async HTTP client.
        #
        # A deletion failure is allowed to propagate so the evaluator can record
        # it as a cleanup error. The HTTP client is still closed in every case.

        try:
            if self._initialized and self._delete_collection_on_close:
                await self._qdrant_call(
                    "delete_collection",
                    lambda: self._client.delete_collection(
                        collection_name=self.collection_name,
                    ),
                )
        finally:
            await self._client.close()
            self._initialized = False  # The task owns no open collection now.

    async def _store_text_records(
        self,
        records: Sequence[dict[str, Any]],
    ) -> None:
        # Embed and upsert records with a limit on work running at once.
        #
        # Every record must contain ``id``, ``text``, and ``ordinal``. Additional
        # fields are stored unchanged as payload metadata.
        #
        # Retry each Qdrant upsert on its own. Do not restart the whole memory
        # build when one batch fails for a short time.

        if not records:
            return

        semaphore = asyncio.Semaphore(self._ingest_concurrency)  # Limits API work.

        async def store_batch(
            batch: Sequence[dict[str, Any]],
        ) -> None:
            async with semaphore:
                texts = [str(record["text"]) for record in batch]  # Embed together.
                vectors = await self.runtime.embed_many(  # One vector per record.
                    texts,
                    phase="memory",
                )

                if len(vectors) != len(batch):
                    raise RuntimeError(
                        "Embedding result count does not match record count"
                    )

                points: list[PointStruct] = []  # Qdrant objects for this batch.

                for record, vector in zip(
                    batch,
                    vectors,
                    strict=True,
                ):
                    payload = dict(record)  # Copy so the caller's record stays intact.
                    record_id = str(payload.pop("id"))  # Original memory ID.
                    point_id = self._qdrant_point_id(record_id)  # UUID for Qdrant.
                    payload.setdefault("record_id", record_id)

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=vector,
                            payload=payload,
                        )
                    )

                # ``points`` is created before retrying. Every retry therefore
                # reuses the same stable point IDs and payloads.
                await self._qdrant_call(
                    "upsert",
                    lambda: self._client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True,
                    ),
                )

        await asyncio.gather(
            *(
                store_batch(records[start : start + self._qdrant_batch_size])
                for start in range(
                    0,
                    len(records),
                    self._qdrant_batch_size,
                )
            )
        )

    async def _upsert_vector_records(
        self,
        records: Sequence[dict[str, Any]],
        vectors: Sequence[Sequence[float]],
    ) -> None:
        # Upsert records whose vectors were already computed by the memory type.

        if len(records) != len(vectors):
            raise ValueError("records and vectors must have the same length")

        if not records:
            return

        semaphore = asyncio.Semaphore(self._ingest_concurrency)  # Limits upserts.

        async def upsert_batch(
            record_batch: Sequence[dict[str, Any]],
            vector_batch: Sequence[Sequence[float]],
        ) -> None:
            async with semaphore:
                points: list[PointStruct] = []  # Qdrant objects for this batch.

                for record, vector in zip(
                    record_batch,
                    vector_batch,
                    strict=True,
                ):
                    payload = dict(record)  # Copy before removing the local ID.
                    record_id = str(payload.pop("id"))  # Original memory ID.
                    point_id = self._qdrant_point_id(record_id)  # UUID for Qdrant.
                    payload.setdefault("record_id", record_id)

                    points.append(
                        PointStruct(
                            id=point_id,
                            vector=list(vector),
                            payload=payload,
                        )
                    )

                await self._qdrant_call(
                    "upsert",
                    lambda: self._client.upsert(
                        collection_name=self.collection_name,
                        points=points,
                        wait=True,
                    ),
                )

        await asyncio.gather(
            *(
                upsert_batch(
                    records[start : start + self._qdrant_batch_size],
                    vectors[start : start + self._qdrant_batch_size],
                )
                for start in range(
                    0,
                    len(records),
                    self._qdrant_batch_size,
                )
            )
        )

    def _qdrant_point_id(self, record_id: str) -> str:
        # Return a stable UUID while preserving source IDs in the payload.

        try:
            return str(uuid.UUID(record_id))
        except ValueError:
            # Deterministic across retries and repeated ingestion attempts for
            # the same task-local record.
            return str(
                uuid.uuid5(
                    uuid.NAMESPACE_URL,
                    f"{self.task_id}:{record_id}",
                )
            )

    async def _query_similar(
        self,
        query: str,
        *,
        limit: int,
        query_filter: Filter | None = None,
    ) -> list[RetrievedMemory]:
        # Run one dense Qdrant search and normalize its scored points.

        if limit <= 0:
            return []

        # Embedding retries are handled independently by OpenAIEmbedder.
        vector = await self.runtime.embed_one(  # Turns the question into a vector.
            query,
            phase="query",
        )

        response = await self._qdrant_call(  # Returns the closest scored points.
            "query_points",
            lambda: self._client.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=limit,
                with_payload=True,
                with_vectors=False,
            ),
        )

        return [self._point_to_memory(point) for point in response.points]

    async def _retrieve_all(
        self,
        *,
        query_filter: Filter | None = None,
    ) -> list[RetrievedMemory]:
        # Scroll the collection without embedding a query or losing order.

        items: list[RetrievedMemory] = []  # Records collected from every page.
        offset: Any = None  # Qdrant cursor for the next page.

        while True:
            # ``offset`` does not change until a request succeeds, so retrying
            # this page cannot accidentally skip a page.
            page_offset = offset  # Freezes the cursor for retries of this page.
            points, next_offset = await self._qdrant_call(  # One scroll page.
                "scroll",
                lambda offset=page_offset: self._client.scroll(
                    collection_name=self.collection_name,
                    scroll_filter=query_filter,
                    limit=256,
                    offset=offset,
                    with_payload=True,
                    with_vectors=False,
                ),
            )

            items.extend(
                self._point_to_memory(
                    point,
                    default_score=1.0,
                )
                for point in points
            )

            offset = next_offset  # None means the final page was read.
            if offset is None:
                break

        items.sort(
            key=lambda item: (
                item.ordinal,
                item.memory_id,
            )
        )
        return items

    async def _retrieve_ordinals(
        self,
        ordinals: Iterable[int],
    ) -> list[RetrievedMemory]:
        # Fetch adjacent ordered records for sequential retrieval.

        unique = sorted({int(value) for value in ordinals if int(value) >= 0})

        if not unique:
            return []

        query_filter = Filter(  # Matches any of the requested source positions.
            must=[
                FieldCondition(
                    key="ordinal",
                    match=MatchAny(any=unique),
                )
            ]
        )

        return await self._retrieve_all(query_filter=query_filter)

    @staticmethod
    def _point_to_memory(
        point: Any,
        default_score: float | None = None,
    ) -> RetrievedMemory:
        # Convert a Qdrant point into the shared result type.

        payload = dict(point.payload or {})  # Copy stored fields from Qdrant.
        text = str(payload.pop("text", ""))  # Main memory text.
        ordinal = int(payload.pop("ordinal", 0))  # Original source position.
        score = getattr(point, "score", None)  # Missing on scroll results.

        effective_score = default_score if score is None else score
        if effective_score is None:
            effective_score = 0.0  # Safe fallback when neither source has a score.

        return RetrievedMemory(
            memory_id=str(point.id),
            text=text,
            score=float(effective_score),
            ordinal=ordinal,
            metadata=payload,
        )
