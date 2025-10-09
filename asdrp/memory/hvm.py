#############################################################################
# hvm.py
#
# class for implementing hierarchical vector memory using TreeIndex
# and Qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.com
#############################################################################


from __future__ import annotations

import datetime as dt
import os
import time
import uuid
from typing import Any, Dict, List
import re

import qdrant_client
import tiktoken
from typing import Optional
from llama_index.core import Document, StorageContext, Settings
from llama_index.core.indices.tree import TreeIndex
from llama_index.core.llms import LLM
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.vector_stores.qdrant import QdrantVectorStore


def create_llm(
    provider: str = "gemini",
    model: str = None,
    api_key: Optional[str] = None,
    **kwargs
) -> LLM:
    """
    Create an LLM instance based on the provider.
    
    Args:
        provider: LLM provider (only "gemini" supported in this version)
        model: Model name (if None, uses default for provider)
        api_key: API key (if None, uses environment variable)
        **kwargs: Additional arguments for the LLM
    
    Returns:
        LLM instance
    """
    
    if provider.lower() == "gemini":
        if model is None:
            model = "models/gemini-2.5-flash-lite" 

        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return Gemini(model=model, **kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Only 'gemini' is supported in this version.")


def create_embedding_model(
    provider: str = "gemini",
    model: str = None,
    api_key: Optional[str] = None,
    **kwargs
):
    """
    Create an embedding model instance based on the provider.
    
    Args:
        provider: Embedding provider (only "gemini" supported in this version)
        model: Model name (if None, uses default for provider)
        api_key: API key (if None, uses environment variable)
        **kwargs: Additional arguments for the embedding model
    
    Returns:
        Embedding model instance
    """
    
    if provider.lower() == "gemini":
        if model is None:
            model = "models/embedding-001"
        
        # Gemini 不需要显式传递 api_key，它自动从环境变量 GOOGLE_API_KEY 读取
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return GeminiEmbedding(model=model, **kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Only 'gemini' is supported in this version.")


class HierarchicalVectorMemory:
    """Long‑term memory that stores every turn as a leaf node.
    TreeIndex automatically builds a hierarchy above the leaves.
    Retrieval walks the hierarchy or uses the collapsed vector view, depending on `mode`."""

    def __init__(
        self,
        collection: str = "agent_mem_hvm",
        similarity_top_k: int = 3,
        mode: str = "tree_traversal",  # or "collapsed"
        host: str = "localhost",
        port: int = 6333,
        provider: str = "openai",
        model: str = "o4-mini",
    ) -> None:
        
        # Connect to the Qdrant collection
        self._client = qdrant_client.QdrantClient(host=host, port=port)
        self._vector_store = QdrantVectorStore(
            client=self._client, collection_name=collection
        )

        # Configure global LlamaIndex settings using factory
        Settings.llm = create_llm(provider=provider, model=model)
        Settings.embed_model = create_embedding_model(provider=provider)

        # Storage context glues the vector store to the index
        self._storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)

        # Empty tree index which grow with every `.store()` call
        self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True) # For testing purposes, num_children are low as 3 so tree can more easily form

        self._sim_top_k = similarity_top_k
        self._mode = mode
        
        # Token tracking for cost calculation
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.load_chat_history_time: float = 0.0
        self.tokenizer = tiktoken.get_encoding("o200k_base")


    async def _aput(self, messages, meta: Dict[str, Any] | None = None) -> None:
        """Insert the new turn as a leaf node."""
        
        # Handle both string and ChatMessage list inputs for compatibility
        if isinstance(messages, str):
            text = messages
        elif isinstance(messages, list):
            # Convert ChatMessage objects to string format
            text_parts = []
            for msg in messages:
                if hasattr(msg, 'content'):
                    role = getattr(msg, 'role', 'unknown')
                    content = msg.content
                    text_parts.append(f"{role.upper()}: {content}")
                else:
                    text_parts.append(str(msg))
            text = "\n".join(text_parts)
        else:
            text = str(messages)

        # Count tokens for this text
        self.input_tokens += len(self.tokenizer.encode(text))

        meta = meta or {}
        meta.update({
            "uuid": str(uuid.uuid4()),
            "tstamp": dt.datetime.utcnow().isoformat(),
        })

        doc = Document(text=text, metadata=meta)
        self._index.insert(doc)


    def _aget(self, query: str) -> List[str]:
        """Similarity search that follows the hierarchical structure."""

        qe = self._index.as_query_engine(
            similarity_top_k=self._sim_top_k,
            mode=self._mode,
        )
        nodes = qe.retrieve(query)
        return [n.node.text for n in nodes]


    def clear_memory(self) -> None:
        """Clear all memory from the Qdrant collection and reset the tree index."""
        try:
            # Delete the entire collection from Qdrant
            self._client.delete_collection(collection_name=self._vector_store.collection_name)
            print(f"✅ Cleared Qdrant collection: {self._vector_store.collection_name}")
            
            # Recreate the collection and vector store
            self._vector_store = QdrantVectorStore(
                client=self._client, collection_name=self._vector_store.collection_name
            )
            
            # Recreate the storage context
            self._storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
            
            # Recreate the empty tree index
            self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True)
            
            print("✅ Memory cleared and tree index reset")
            
        except Exception as e:
            print(f"❌ Error clearing memory: {e}")
            # If collection doesn't exist, that's fine - just recreate it
            try:
                self._vector_store = QdrantVectorStore(
                    client=self._client, collection_name=self._vector_store.collection_name
                )
                self._storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
                self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True)
                print("✅ Recreated empty memory structure")
            except Exception as recreate_error:
                print(f"❌ Error recreating memory structure: {recreate_error}")
                raise

    def print_tree(self, max_chars: int = 80) -> None:
        """Helper function just for better formatting when testing"""

        # Calls the built-in printer if exists
        if hasattr(self._index, "print_tree"):
            try:
                self._index.print_tree(max_chars=max_chars)
                return
            except Exception:
                pass

        # Manually traverse root to children
        try:
            graph = getattr(self._index, "_index_graph", None)
            root_nodes = getattr(graph, "root_nodes", []) if graph else []
        except Exception:
            root_nodes = []

        # Print the last few stored texts (if previous approach didn't work)
        if not root_nodes:
            
            print("(print_tree not supported – showing last inserts)")
            for doc_id, doc in list(self._index.docstore.docs.items())[-5:]:  # type: ignore[attr-defined]
                snippet = re.sub(r"\s+", " ", doc.text).strip()[:max_chars] #another display method: snippet = doc.text.replace("", " ")[:max_chars]
                print(f"- {doc_id[:6]}… {snippet}")
            return


        def _print(node, indent: int = 0):
            spacer = "  " * indent  # alignment
            snippet = re.sub(r"\s+", " ", doc.text).strip()[:max_chars]
            print(f"{spacer}┣━━ {snippet}")

            # children  kept in node.relationships[NodeRelationship.CHILD]
            rels = getattr(node, "relationships", {})
            from llama_index.core import NodeRelationship  # local import avoids top‑level dep

            children = rels.get(NodeRelationship.CHILD, [])
            child_ids = [c.node_id if hasattr(c, "node_id") else c for c in children]

            for cid in child_ids:
                child_node = self._index.docstore.get_node(cid)
                if child_node is not None:
                    _print(child_node, indent + 1)


        print("ROOTS:")
        for root in root_nodes:
            _print(root)
