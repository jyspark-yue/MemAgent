#############################################################################
# File: condensed_memory.py
#
# Description:
#   A condensed memory block that maintains context while staying within reasonable memory limits.
#
# Authors:
#   @author     Theodore Mui (theodoremui@gmail.com)
#               - Created summary_agent.py
#   @author     Eric Vincent Fernandes
#               - Implemented tracking for token/cost metrics
#               - Modified code to be compatible with Gemini (GenAI)
#   @author     Varenya Garg
#               - Modified code to store and retrieve messages using Qdrant vector database
#
# Date:
#   Created:    July 2, 2025  (Theodore Mui)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#   Modified:   April 9, 2026 (Varenya Garg)
#############################################################################

import time
from typing import Any, List, Optional
import qdrant_client
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client.models import Distance, VectorParams, PointIdsList
from llama_index.core.schema import TextNode
from llama_index.core import VectorStoreIndex, StorageContext, Settings
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.utils import count_tokens
from pydantic import Field
import json

DEFAULT_TOKEN_LIMIT = 60000

class CondensedMemoryBlock(BaseMemoryBlock[str]):
    """
    This class is a smart conversation buffer that maintains context while
    staying within reasonable memory limits.

    It condenses the conversation history into a single string, while 
    maintaining a token limit.

    It also includes additional kwargs, like tool calls, when needed.
    """

    def __init__(self, 
    collection: str = "agent_condensed_mem",
    host: str = "localhost", 
    port: int = 6333,
    **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        self._collection = collection
        # Connect to the Qdrant collection
        self._client = qdrant_client.QdrantClient(host=host, port=port)
        self._aclient = qdrant_client.AsyncQdrantClient(host=host, port=port)
        # Check if there is an existing collection 
        existing = [c.name for c in self._client.get_collections().collections]
        # Create a new collection if none exists
        if self._collection not in existing:
            self._client.create_collection(
                collection_name=self._collection,
                vectors_config=VectorParams(
                    size=1536,        # must match embedding model's output
                    distance=Distance.COSINE
                )
            )

        self._vector_store = QdrantVectorStore(client = self._client, collection_name=collection, aclient=self._aclient)
       
        Settings.llm = OpenAI(model="o4-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")
        
        self._storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)

        self._index = VectorStoreIndex(nodes=[], storage_context = self._storage_ctx)


    current_memory: List[str] = Field(default_factory=list)
    token_limit: int = Field(default=DEFAULT_TOKEN_LIMIT)
    input_tokens: int = Field(default=0, description="The number of tokens passed into the LLM when loading the chat history.")
    output_tokens: int = Field(default=0, description="The number of tokens returned by the LLM when loading the chat history. (Unneeded Here)")
    load_chat_history_time: float = Field(default=0.0, description="The duration of time it took to load the chat history.")

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Return the current memory block contents."""
        if messages is None:
            return ""

        # Use most recent query as the search query
        query = messages[-1].content
        query_vector = await Settings.embed_model.aget_text_embedding(query)
        top_k = block_kwargs.get("top_k", 3)
        
        results = self._client.query_points(
                collection_name=self._collection,
                query=query_vector,
                limit=top_k,
                with_payload=True
            ).points

        
        if not results: 
            return ""
        
        return "\n".join([
            json.loads(point.payload["_node_content"])["text"] 
            for point in results
        ])

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Push messages into the memory block. (Only handles text content)"""

        # Skip if no messages
        if not messages:
            return

        start_time = time.time()

        # construct a string for each message
        for message in messages:
            text_contents = "\n".join(
                block.text
                for block in message.blocks
                if isinstance(block, TextBlock)
            )
            memory_str = text_contents if text_contents else ""
            kwargs = {}
            for key, val in message.additional_kwargs.items():
                if key == "tool_calls":
                    val = [
                        {
                            "name": tool_call["function"]["name"],
                            "args": tool_call["function"]["arguments"],
                        }
                        for tool_call in val
                    ]
                    kwargs[key] = val
                elif key not in ("session_id", "tool_call_id"):
                    kwargs[key] = val
            memory_str += f"\n({kwargs})" if kwargs else ""

            nodes = TextNode(text=memory_str) 
            await self._index.ainsert_nodes(nodes)


            # Count tokens for this new message (input tokens)
            results, _ = self._aclient.scroll(
                collection_name=self._collection,
                limit=1000,
                with_payload=True
            )

            all_messages = "".join([
            json.loads(point.payload["_node_content"])["text"] 
            for point in results
        ])
        # ensure this memory block doesn't get too large
        message_length = count_tokens(all_messages)
        while message_length > self.token_limit:
            oldest_point = results[0].id

            self._client.delete(
                collection_name = self._collection,
                points_selector = PointIdsList(points = [oldest_point])
            )

            results = results[1:]

            all_messages = "".join([
                json.loads(point.payload["_node_content"])["text"] 
                for point in results
            ])

            message_length = count_tokens(all_messages)

        self.load_chat_history_time = time.time() - start_time