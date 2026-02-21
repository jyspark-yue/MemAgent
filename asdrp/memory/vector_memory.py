#############################################################################
# File: vector_memory.py
#
# Description:
#   Qdrant-backed vector memory block for storing and retrieving conversation
#   turns with semantic search.
#############################################################################

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from typing import Any, List, Optional

from llama_index.core import Document, StorageContext, VectorStoreIndex
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.utils import count_tokens
from pydantic import Field


class VectorMemoryBlock(BaseMemoryBlock[str]):
    """Vector memory block that stores chat turns and retrieves relevant context."""

    name: str = Field(default="vector_memory", description="Memory block name.")
    vector_store: Any = Field(description="Underlying vector store (QdrantVectorStore).")
    embed_model: Any = Field(default=None, description="Embedding model for vector indexing.")
    similarity_top_k: int = Field(default=3, description="Top-k nodes to retrieve.")
    input_tokens: int = Field(default=0, description="Tokens processed while loading chat history.")
    output_tokens: int = Field(default=0, description="Completion tokens while loading history (none for vector memory).")
    load_chat_history_time: float = Field(default=0.0, description="Cumulative time spent loading chat history.")

    def model_post_init(self, __context: Any) -> None:
        storage_context = StorageContext.from_defaults(vector_store=self.vector_store)
        self._index = VectorStoreIndex.from_vector_store(
            vector_store=self.vector_store,
            embed_model=self.embed_model,
            storage_context=storage_context,
        )

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Store incoming messages into the vector index."""
        if not messages:
            return

        start_time = time.time()
        batch_tokens = 0

        for message in messages:
            content = (message.content or "").strip()
            if not content:
                continue

            batch_tokens += count_tokens(content)
            doc = Document(
                text=content,
                metadata={
                    "id": str(uuid.uuid4()),
                    "role": str(message.role),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                },
            )
            self._index.insert(doc)

        self.input_tokens += batch_tokens
        self.load_chat_history_time += (time.time() - start_time)

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Retrieve relevant memory snippets for the latest query message."""
        query = self._extract_query(messages)
        if not query:
            return ""

        retriever = self._index.as_retriever(similarity_top_k=self.similarity_top_k)
        nodes = retriever.retrieve(query)
        if not nodes:
            return ""

        return "\n".join(node.node.text for node in nodes if getattr(node, "node", None))

    @staticmethod
    def _extract_query(messages: Optional[List[ChatMessage]]) -> str:
        if not messages:
            return ""
        for message in reversed(messages):
            content = (message.content or "").strip()
            if content:
                return content
        return ""
