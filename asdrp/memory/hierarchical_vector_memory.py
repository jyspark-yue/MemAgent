#############################################################################
# hierarchical_vector_memory.py
#
# class for implementing hierarchical vector memory using TreeIndex
# and Qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.com
#############################################################################


from __future__ import annotations

import datetime as dt
import uuid
from typing import Any, Dict, List
import re

import qdrant_client
from llama_index.core import Document, StorageContext, Settings
from llama_index.core.indices.tree import TreeIndex
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.qdrant import QdrantVectorStore


class HierarchicalVectorMemory:
    """Long‑term memory that stores every turn as a leaf node.
    TreeIndex utomatically builds a hierarchy above the leaves.
    Retrieval walks the hierarchy or uses the collapsed vector view, depending on `mode`."""

    def __init__(
        self,
        collection: str = "agent_mem_hvm",
        similarity_top_k: int = 3,
        mode: str = "tree_traversal",  # or "collapsed"
        host: str = "localhost",
        port: int = 6333,
    ) -> None:
        
        # Connect to the Qdrant collection
        self._client = qdrant_client.QdrantClient(host=host, port=port)
        self._vector_store = QdrantVectorStore(
            client=self._client, collection_name=collection
        )

        # Configure global LlamaIndex settings
        Settings.llm = OpenAI(model="o4-mini")
        Settings.embed_model = OpenAIEmbedding(model="text-embedding-3-small")

        # Storage context glues the vector store to the index
        self._storage_ctx = StorageContext.from_defaults(vector_store=self._vector_store)

        # Empty tree index which grow with every `.store()` call
        self._index = TreeIndex([], storage_context=self._storage_ctx, num_children=3, build_tree=True) # For testing purposes, num_children are low as 3 so tree can more easily form

        self._sim_top_k = similarity_top_k
        self._mode = mode


    def store(self, text: str, meta: Dict[str, Any] | None = None) -> None:
        """Insert the new turn as a leaf node."""

        meta = meta or {}
        meta.update({
            "uuid": str(uuid.uuid4()),
            "tstamp": dt.datetime.utcnow().isoformat(),
        })

        doc = Document(text=text, metadata=meta)
        self._index.insert(doc)


    def retrieve(self, query: str) -> List[str]:
        """Similarity search that follows the hierarchical structure."""

        qe = self._index.as_query_engine(
            similarity_top_k=self._sim_top_k,
            mode=self._mode,
        )
        nodes = qe.retrieve(query)
        return [n.node.text for n in nodes]


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
            spacer = "  " * indent  # alignment
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
