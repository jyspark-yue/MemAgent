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
import asyncio
import os
import time
import uuid
from typing import Any, Dict, List
import re
import concurrent.futures

import tiktoken
from typing import Optional
from llama_index.core import Document, StorageContext, Settings
from llama_index.core import load_index_from_storage
from llama_index.core.indices.tree import TreeIndex
from llama_index.core.llms import LLM
from llama_index.llms.google_genai import GoogleGenAI


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
            model = "gemini-2.5-flash"  # According to LlamaIndex Google GenAI docs
        
        # Google GenAI reads API key from env var GOOGLE_API_KEY
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return GoogleGenAI(model=model, **kwargs)
    
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
        # Lazy import; only depend on embeddings when vector retrieval is enabled
        try:
            from llama_index.embeddings.google_genai import GoogleGenAIEmbedding as EmbeddingCls
            default_embed_model = "text-embedding-004"
        except Exception as e:
            # Require the newer Google GenAI path; do not fallback to legacy GeminiEmbedding to avoid old deps
            raise ImportError(
                "GoogleGenAIEmbedding is not available in your llama_index version. "
                "Please upgrade llama-index to a version that provides 'llama_index.embeddings.google_genai'."
            ) from e

        if model is None:
            model = default_embed_model

        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")

        return EmbeddingCls(model=model, **kwargs)
    
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
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        persist_dir: Optional[str] = "storage/hvm_tree",
    ) -> None:
        
        self._sim_top_k = similarity_top_k
        self._mode = mode

        # Always set LLM
        Settings.llm = create_llm(provider=provider, model=model)

        self._client = None
        self._vector_store = None

        self._persist_dir = persist_dir

        # Decide whether persisted files exist (first-run vs resume)
        persist_ready = False
        if self._persist_dir:
            try:
                os.makedirs(self._persist_dir, exist_ok=True)
            except Exception:
                pass
            needed = ["docstore.json", "index_store.json"]
            persist_ready = all(os.path.exists(os.path.join(self._persist_dir, n)) for n in needed)

        # Storage context: only attach vector store and embeddings when using collapsed mode
        if self._mode == "collapsed":
            # Create embedding model and vector store lazily
            Settings.embed_model = create_embedding_model(provider=provider)

            # Lazy import qdrant deps to avoid hard dependency when not needed
            import qdrant_client
            from llama_index.vector_stores.qdrant import QdrantVectorStore

            self._client = qdrant_client.QdrantClient(host=host, port=port)
            self._vector_store = QdrantVectorStore(
                client=self._client, collection_name=collection
            )
            if persist_ready:
                self._storage_ctx = StorageContext.from_defaults(
                    vector_store=self._vector_store,
                    persist_dir=self._persist_dir,
                )
            else:
                self._storage_ctx = StorageContext.from_defaults(
                    vector_store=self._vector_store,
                )
        else:
            # Tree traversal mode can work without embeddings/vector store
            if persist_ready:
                self._storage_ctx = StorageContext.from_defaults(
                    persist_dir=self._persist_dir,
                )
            else:
                self._storage_ctx = StorageContext.from_defaults()

        # Load existing index if persisted, else create empty and build
        index_loaded = False
        if persist_ready:
            try:
                self._index = load_index_from_storage(self._storage_ctx)
                index_loaded = True
                print(f"Loaded persisted TreeIndex from {self._persist_dir}")
            except Exception:
                index_loaded = False

        if not index_loaded:
            # Empty tree index which grow with every `.store()` call
            # Tune num_children to reduce merge depth; enable async building
            lightweight_tree_llm = create_llm(provider=provider, model="gemini-2.5-flash")
            self._index = TreeIndex(
                [],
                storage_context=self._storage_ctx,
                num_children=10,
                build_tree=True,
                use_async=True,
                llm=lightweight_tree_llm,
            )
            if self._persist_dir:
                print(f"Initialized new TreeIndex (persist_dir={self._persist_dir or 'None'})")
        
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
        session_id = meta.get("session_id")
        meta.update({
            "uuid": str(uuid.uuid4()),
            "tstamp": dt.datetime.utcnow().isoformat(),
        })

        # If a stable session_id is provided, and already ingested, skip to prevent duplicates
        if session_id:
            try:
                ref = self._storage_ctx.docstore.get_ref_doc_info(session_id)
                if ref is not None:
                    print(f"[HVM] Skip already ingested session_id={session_id}")
                    return
            except Exception:
                pass

        if session_id:
            doc = Document(text=text, metadata=meta, doc_id=session_id)
        else:
            doc = Document(text=text, metadata=meta)
        start_time = time.time()
        # Run blocking TreeIndex.insert in a worker thread to avoid nested event loop issues
        await asyncio.to_thread(self._index.insert, doc)
        self.load_chat_history_time += time.time() - start_time
        # Persist updated storage after insert to enable resume-on-restart
        try:
            if self._persist_dir:
                self._storage_ctx.persist(persist_dir=self._persist_dir)
                if session_id:
                    print(f"[HVM] Persisted session_id={session_id} -> {self._persist_dir}")
        except Exception:
            pass


    def _aget(self, query: str) -> List[str]:
        """Similarity search that follows the hierarchical structure."""

        qe = self._index.as_query_engine(
            similarity_top_k=self._sim_top_k,
            mode=self._mode,
        )
        nodes = qe.retrieve(query)
        return [n.node.text for n in nodes]


    def clear_memory(self) -> None:
        """Clear all memory and reset the tree index. If using Qdrant (collapsed mode), clear the collection; otherwise just rebuild in-memory index."""
        if self._vector_store is not None and self._client is not None:
            try:
                # Delete the entire collection from Qdrant
                collection_name = getattr(self._vector_store, "collection_name", None)
                if collection_name:
                    self._client.delete_collection(collection_name=collection_name)
                    print(f"Cleared Qdrant collection: {collection_name}")

                # Recreate vector store and storage context
                from llama_index.vector_stores.qdrant import QdrantVectorStore
                self._vector_store = QdrantVectorStore(
                    client=self._client, collection_name=collection_name or "agent_mem_hvm"
                )
                self._storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)
                self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True)
                print("Memory cleared and tree index reset")
            except Exception as e:
                print(f"Error clearing memory: {e}")
                # Fallback to in-memory reset
                self._storage_ctx = StorageContext.from_defaults()
                self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True)
                print("Recreated empty in-memory structure")
        else:
            # No external vector store; reset in-memory tree index
            self._storage_ctx = StorageContext.from_defaults()
            self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True)
            print("In-memory tree index reset")

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
            
            print("(print_tree not supported - showing last inserts)")
            for doc_id, doc in list(self._index.docstore.docs.items())[-5:]:  # type: ignore[attr-defined]
                snippet = re.sub(r"\s+", " ", doc.text).strip()[:max_chars] #another display method: snippet = doc.text.replace("", " ")[:max_chars]
                print(f"- {doc_id[:6]}... {snippet}")
            return


        def _print(node, indent: int = 0):
            spacer = "  " * indent  # alignment
            snippet = re.sub(r"\s+", " ", doc.text).strip()[:max_chars]
            print(f"{spacer}|-- {snippet}")

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
