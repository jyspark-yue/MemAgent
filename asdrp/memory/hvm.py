#############################################################################
# hvm.py
#
# class for implementing hierarchical vector memory using TreeIndex
# and Qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.com
#############################################################################




# Standard Library Imports
from __future__ import annotations
import os
import re
import uuid
import threading
import asyncio
import datetime as dt
import random
import warnings
from enum import Enum
from concurrent.futures import ThreadPoolExecutor
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Callable,
    Sequence,
    Awaitable,
    Union,
)

"""
Adjusted from https://github.com/run-llama/llama_index/blob/main/llama-index-packs/llama-index-packs-raptor/llama_index/packs/raptor

Full credits to the original authors!
"""


"""
To use Qdrant locally, first install Docker from https://www.docker.com/get-started
then run the following commands:

docker pull qdrant/qdrant

# start qdrant with persistence

docker run -p 6333:6333 -p 6334:6334 \
    -v "$(pwd)/qdrant_storage:/qdrant/storage:z" \
    qdrant/qdrant

This will start a Qdrant instance on localhost:6333 with data persisted in the qdrant_storage directory.
"""

# Environment Configuration
warnings.filterwarnings("ignore", category=FutureWarning)



# Math / ML / Data Libraries
import numba
import numpy as np
import umap
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.preprocessing import StandardScaler




# LlamaIndex Core Components
from llama_index.core import (
    get_tokenizer,
    Document,
    StorageContext,
    Settings,
    VectorStoreIndex,
    get_response_synthesizer,
    load_index_from_storage,
)
from llama_index.core.schema import (
    BaseNode,
    NodeWithScore,
    QueryBundle,
    TextNode,
    TransformComponent,
)
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.base.response.schema import Response
from llama_index.core.base.base_retriever import BaseRetriever, QueryType
from llama_index.core.bridge.pydantic import BaseModel, Field
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.response_synthesizers import BaseSynthesizer
from llama_index.core.vector_stores.types import (
    MetadataFilter,
    MetadataFilters,
    BasePydanticVectorStore,
)
from llama_index.core.llama_pack.base import BaseLlamaPack
from llama_index.core.llms.llm import LLM
from llama_index.core.ingestion import run_transformations
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.indices.tree import TreeIndex

# ===============================
# 🤖 LlamaIndex LLMs & Embeddings
# ===============================
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.embeddings.google_genai import GoogleGenAIEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# ===============================
# ☁️ External APIs / SDKs
# ===============================
from google.genai import types
from google.genai.types import EmbedContentConfig


# Vector Store (Qdrant)
import qdrant_client
from qdrant_client import QdrantClient, AsyncQdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore


DEFAULT_SUMMARY_PROMPT = (
    "Summarize the provided text, including as many key details as needed."
)


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


DEFAULT_QDRANT_PERSIST_DIR = os.environ.get(
    "QDRANT_PERSIST_DIR", os.path.join(BASE_DIR, "qdrant_data")
)


if not os.path.exists(DEFAULT_QDRANT_PERSIST_DIR):
    print(f"Creating Qdrant persistence directory at {DEFAULT_QDRANT_PERSIST_DIR}")
    os.makedirs(DEFAULT_QDRANT_PERSIST_DIR)


# ---------------------------------------------------------------------------
# **CHANGED**: Shared TokenCountingHandler & CallbackManager for LLM+Embeddings
# ---------------------------------------------------------------------------
# Using ONE shared handler ensures token counts aggregate correctly across
# LLM completions and embedding calls; the evaluator expects combined usage.
TOKEN_HANDLER = TokenCountingHandler()  # **CHANGED**
CALLBACK_MANAGER = CallbackManager(handlers=[TOKEN_HANDLER])  # **CHANGED**

# 🔴 ADD THIS LINE — makes the handler global so internal components (like ResponseSynthesizer)
# created by LlamaIndex also get instrumented automatically.
from llama_index.core import Settings
Settings.callback_manager = CALLBACK_MANAGER



def create_default_llm(
    callback_manager: CallbackManager = None,
    model="gemini-2.5-flash-lite",
    temperature: float = 0.2,
    **kwargs,
) -> LLM:

    if callback_manager is None:
        callback_manager = CALLBACK_MANAGER  # **CHANGED**
    _safety_settings = [
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
            # threshold=types.HarmBlockThreshold.OFF,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
            # threshold=types.HarmBlockThreshold.OFF,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
            # threshold=types.HarmBlockThreshold.OFF,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
        types.SafetySetting(
            category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
            # threshold=types.HarmBlockThreshold.OFF,
            threshold=types.HarmBlockThreshold.BLOCK_NONE,
        ),
    ]

    _gen_cfg = types.GenerateContentConfig(
        safety_settings=_safety_settings, temperature=temperature
    )
    llm = GoogleGenAI(
        model=model,
        temperature=temperature,
        max_retries=100,
        callback_manager=callback_manager,  # **CHANGED** (ensure shared CM is passed)
        generation_config=_gen_cfg,
        **kwargs,
    )
    print("llm.callback_manager.handlers:", llm.callback_manager.handlers)
    print("llm:", llm)
    return llm



def create_default_embedding_model(
    model_name: str = "text-embedding-004",
    embed_batch_size: int = 100,
    callback_manager: CallbackManager = None,
    **kwargs,
):
    """
    Create a default embedding model.
    """
    if callback_manager is None:
        callback_manager = CALLBACK_MANAGER  # **CHANGED**
    embed_model = GoogleGenAIEmbedding(
        model_name=model_name,
        embed_batch_size=embed_batch_size,
        callback_manager=callback_manager,  # **CHANGED** (ensure shared CM is passed)
        **kwargs,
    )
    print(
        "embed_model.callback_manager.handlers:", embed_model.callback_manager.handlers
    )
    print("embed_model:", embed_model)
    return embed_model



class QueryModes(str, Enum):
    """Query modes."""

    tree_traversal = "tree_traversal"
    collapsed = "collapsed"



class SummaryModule(BaseModel):
    response_synthesizer: BaseSynthesizer = Field(description="LLM")
    summary_prompt: str = Field(
        default=DEFAULT_SUMMARY_PROMPT,
        description="Summary prompt.",
    )
    num_workers: int = Field(
        default=4, description="Number of workers to generate summaries."
    )
    show_progress: bool = Field(default=True, description="Show progress.")

    class Config:
        arbitrary_types_allowed = True

    def __init__(
        self,
        llm: Optional[LLM] = None,
        summary_prompt: str = DEFAULT_SUMMARY_PROMPT,
        num_workers: int = 4,
    ) -> None:
        response_synthesizer = get_response_synthesizer(
            response_mode="tree_summarize", use_async=True, llm=llm
        )
        super().__init__(
            response_synthesizer=response_synthesizer,
            summary_prompt=summary_prompt,
            num_workers=num_workers,
        )

    async def generate_summaries(
        self, documents_per_cluster: List[List[BaseNode]]
    ) -> List[str]:
        """
        Generate summaries of documents per cluster.

        Args:
            documents_per_cluster (List[List[BaseNode]]): List of documents per cluster

        Returns:
            List[str]: List of summary for each cluster

        """
        jobs = []
        total_docs = sum(len(documents) for documents in documents_per_cluster)
        total_clusters = len(documents_per_cluster)
        for documents in documents_per_cluster:
            with_scores = [NodeWithScore(node=doc, score=1.0) for doc in documents]
            jobs.append(
                self.response_synthesizer.asynthesize(self.summary_prompt, with_scores)
            )

        lock = asyncio.Semaphore(self.num_workers)
        responses = []

        # run the jobs while limiting the number of concurrent jobs to num_workers
        for job in jobs:
            async with lock:
                responses.append(await job)
        print(
            f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@LLM CALLED, generated {len(responses)} summaries. total docs: {total_docs}, total clusters: {total_clusters}"
        )
        # print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ responses:", responses)
        return [str(response) for response in responses]



def create_summary_module(llm: Optional[LLM] = None) -> SummaryModule:
    if llm is None:
        llm = create_default_llm()
    return SummaryModule(llm=llm)



def create_qdrant_client(
    qdrant_persist_dir: Optional[str] = DEFAULT_QDRANT_PERSIST_DIR,
    host: str = "localhost",
    port: int = 6333,
) -> AsyncQdrantClient:
    # prefer using host/port if available since it will provide better performance
    try:
        client = AsyncQdrantClient(host=host, port=port)
    except Exception as e:
        client = (
            AsyncQdrantClient(path=qdrant_persist_dir)
            if qdrant_persist_dir
            else AsyncQdrantClient()
        )
    return client



def create_vector_store(
    client: AsyncQdrantClient,
    collection_name: str = "agent_mem_hvm",
) -> QdrantVectorStore:
    vector_store = QdrantVectorStore(aclient=client, collection_name=collection_name)
    return vector_store



def create_vector_store_index(
    vector_store: QdrantVectorStore,
    embed_model: Optional[BaseEmbedding] = create_default_embedding_model(),
    transformations: Optional[List[TransformComponent]] = None,
) -> VectorStoreIndex:
    index = VectorStoreIndex(
        nodes=[],
        use_async=True,
        storage_context=StorageContext.from_defaults(vector_store=vector_store),
        embed_model=embed_model,
        transformations=transformations,
        show_progress=True,
    )
    return index



def create_hvm(
    client: AsyncQdrantClient,
    name: str = "hvm_memory",
    collection_name: str = "hvm_memory_collection",
    tree_depth: int = 3,
    llm: Optional[LLM] = None,
    embed_model: Optional[BaseEmbedding] = None,
    transformations: Optional[List[TransformComponent]] = None,
) -> "HierarchicalVectorMemory":
    if llm is None:
        llm = create_default_llm()
    summary_module = create_summary_module(llm=llm)
    if embed_model is None:
        embed_model = create_default_embedding_model()
    vector_store = create_vector_store(client=client, collection_name=collection_name)
    index = create_vector_store_index(
        vector_store=vector_store,
        embed_model=embed_model,
        transformations=transformations,
    )

    return HierarchicalVectorMemory(
        name=name,
        tree_depth=tree_depth,
        summary_module=summary_module,
        index=index,
        llm=llm,
        embed_model=embed_model,
    )



class HierarchicalVectorMemory(BaseMemoryBlock[str]):
    """Long-term memory that stores every turn as a leaf node.
    TreeIndex automatically builds a hierarchy above the leaves.
    Retrieval walks the hierarchy or uses the collapsed vector view, depending on `mode`.
    """

    input_tokens: int = Field(
        default=0,
        description="The number of tokens passed into the LLM when loading the chat history.",
    )
    output_tokens: int = Field(
        default=0,
        description="The number of tokens returned by the LLM when loading the chat history.",
    )

    embed_tokens: int = Field(
        default=0,
        description="The number of tokens used for embeddings.",
    )

    load_chat_history_time: float = Field(
        default=0.0,
        description="The duration of time it took to load the chat history.",
    )

    retrieval_time: float = Field(
        default=0.0,
        description="The duration of time it took to retrieve the chat history.",
    )

    llm: LLM = Field(description="LLM")
    embed_model: BaseEmbedding = Field(description="Embedding model")

    message_history: List[ChatMessage] = Field(
        default_factory=list, description="History of messages exchanged with the user."
    )

    tree_depth: int = Field(default=3, description="The depth of the tree.")
    summary_module: SummaryModule = Field(
        description="Module for summarizing clusters of nodes."
    )
    index: Optional[VectorStoreIndex] = Field(
        default=None, description="The TreeIndex instance."
    )

    verbose: bool = Field(default=False, description="Verbose mode.")

    def update_stats(self):
        llm_handler = None
        embed_handler = None
        try:
            llm_handlers = self.llm.callback_manager.handlers or []
            llm_handler = next((h for h in llm_handlers if isinstance(h, TokenCountingHandler)), None)
        except Exception:
            llm_handler = None
        try:
            embed_handlers = self.embed_model.callback_manager.handlers or []
            embed_handler = next((h for h in embed_handlers if isinstance(h, TokenCountingHandler)), None)
        except Exception:
            embed_handler = None

        # ⚠️ If you want update_stats to be "read-only", DO NOT overwrite self.input_tokens/… here.
        # If you do want absolute totals for debugging, store them under different attributes.
        # Example (debug-only):
        if llm_handler:
            self._abs_llm_prompt_tokens = llm_handler.prompt_llm_token_count
            self._abs_llm_completion_tokens = llm_handler.completion_llm_token_count
        if embed_handler:
            self._abs_embed_tokens = getattr(embed_handler, "total_embedding_token_count", 0)

        print(
            f"HVM stats (abs): prompt={getattr(self, '_abs_llm_prompt_tokens', 0)}, "
            f"completion={getattr(self, '_abs_llm_completion_tokens', 0)}, "
            f"embed={getattr(self, '_abs_embed_tokens', 0)}, "
            f"load_time={self.load_chat_history_time:.2f}s, retrieval_time={self.retrieval_time:.2f}s"
        )


    async def _get_embeddings_per_level(self, level: int = 0) -> List[float]:
        """
        Retrieve embeddings per level in the abstraction tree.

        Args:
            level (int, optional): Target level. Defaults to 0 which stands for leaf nodes.

        Returns:
            List[float]: List of embeddings

        """
        filters = MetadataFilters(filters=[MetadataFilter("level", level)])

        # kind of janky, but should work with any vector index
        source_nodes = await self.index.as_retriever(
            similarity_top_k=10000, filters=filters
        ).retrieve("retrieve")

        return [x.node for x in source_nodes]

    # inside class HierarchicalVectorMemory(BaseMemoryBlock[str]):

    async def _aput(self, messages: Optional[List[ChatMessage]], **kwargs) -> None:
        start_time = time.time()
        if not messages:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@@ No messages to store in HVM.")
            return
        print(f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Storing {len(messages)} in HVM.")
        self.message_history.extend(messages)

        # --- count DELTAS from the SAME LLM that the summarizer actually uses ---
        # Prefer the synthesizer's LLM (it is the one doing the summarization calls)
        llm_for_summary = getattr(
            getattr(self.summary_module, "response_synthesizer", None),
            "llm",
            self.llm,
        )
        try:
            llm_handlers = getattr(getattr(llm_for_summary, "callback_manager", None), "handlers", []) or []
            llm_handler = next((h for h in llm_handlers if isinstance(h, TokenCountingHandler)), None)
        except Exception:
            llm_handler = None

        try:
            embed_handlers = getattr(getattr(self.embed_model, "callback_manager", None), "handlers", []) or []
            embed_handler = next((h for h in embed_handlers if isinstance(h, TokenCountingHandler)), None)
        except Exception:
            embed_handler = None

        p0 = llm_handler.prompt_llm_token_count if llm_handler else 0
        c0 = llm_handler.completion_llm_token_count if llm_handler else 0
        e0 = getattr(embed_handler, "total_embedding_token_count", 0) if embed_handler else 0
        # ------------------------------------------------------------------------

        def _chat_to_node(msg: ChatMessage) -> TextNode:
            return TextNode(text=msg.content, metadata={"role": msg.role})

        documents: List[BaseNode] = [_chat_to_node(m) for m in messages]
        embed_model = self.index._embed_model
        transformations = self.index._transformations

        # Run any ingestion-time transformations
        cur_nodes = run_transformations(documents, transformations, in_place=False)

        # Build hierarchy level by level
        for level in range(self.tree_depth):
            # 1) Embed current nodes
            embeddings = await embed_model.aget_text_embedding_batch(
                [node.get_content(metadata_mode="embed") for node in cur_nodes]
            )
            assert len(embeddings) == len(cur_nodes)
            id_to_embedding = {node.id_: emb for node, emb in zip(cur_nodes, embeddings)}

            # 2) Cluster nodes (async)
            nodes_per_cluster = await aget_clusters(cur_nodes, id_to_embedding)

            # 3) Summarize each cluster using the synthesizer (this triggers LLM calls)
            summaries_per_cluster = await self.summary_module.generate_summaries(nodes_per_cluster)

            # 4) Create new summary nodes for the next level
            new_nodes = [
                TextNode(
                    text=summary,
                    metadata={"level": level},
                    excluded_embed_metadata_keys=["level"],
                    excluded_llm_metadata_keys=["level"],
                )
                for summary in summaries_per_cluster
            ]

            # 5) Insert child nodes with embeddings and parent links
            nodes_with_embeddings: List[BaseNode] = []
            for cluster, summary_doc in zip(nodes_per_cluster, new_nodes):
                for node in cluster:
                    node.metadata["parent_id"] = summary_doc.id_
                    node.excluded_embed_metadata_keys.append("parent_id")
                    node.excluded_llm_metadata_keys.append("parent_id")
                    node.embedding = id_to_embedding[node.id_]
                    nodes_with_embeddings.append(node)

            print(
                f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Inserting {len(nodes_with_embeddings)} nodes into the index at level {level}."
            )
            await self.index.ainsert_nodes(nodes_with_embeddings)

            # 6) Next level works on the summaries we just created
            cur_nodes = new_nodes

        # Insert the top-level summaries as well
        await self.index.ainsert_nodes(cur_nodes)

        # --- compute AFTER−BEFORE deltas and ACCUMULATE into memory counters ---
        if llm_handler:
            self.input_tokens  += max(0, llm_handler.prompt_llm_token_count     - p0)
            self.output_tokens += max(0, llm_handler.completion_llm_token_count - c0)
        if embed_handler:
            e1 = getattr(embed_handler, "total_embedding_token_count", 0)
            self.embed_tokens += max(0, e1 - e0)
        # ------------------------------------------------------------------------

        self.load_chat_history_time += time.time() - start_time

        # Optional: keep as debug only; avoid overwriting per-question counters
        try:
            self.update_stats()
        except Exception:
            pass



    async def collapsed_retrieval(
        self, query_str: str, similarity_top_k: int
    ) -> Response:
        """Query the index as a collapsed tree -- i.e. a single pool of nodes."""
        return await self.index.as_retriever(
            similarity_top_k=similarity_top_k
        ).aretrieve(query_str)

    async def tree_traversal_retrieval(
        self, query_str: str, similarity_top_k: int
    ) -> Response:
        """Query the index as a tree, traversing the tree from the top down."""
        # get top k nodes for each level, starting with the top
        parent_ids = None
        selected_node_ids = set()
        selected_nodes = []
        level = self.tree_depth - 1
        while level >= 0:
            # retrieve nodes at the current level
            if parent_ids is None:
                nodes = await self.index.as_retriever(
                    similarity_top_k=similarity_top_k,
                    filters=MetadataFilters(
                        filters=[MetadataFilter(key="level", value=level)]
                    ),
                ).aretrieve(query_str)

                for node in nodes:
                    if node.id_ not in selected_node_ids:
                        selected_nodes.append(node)
                        selected_node_ids.add(node.id_)

                parent_ids = [node.id_ for node in nodes]
                if self.verbose:
                    print(f"Retrieved parent IDs from level {level}: {parent_ids!s}")
            # retrieve nodes that are children of the nodes at the previous level
            elif parent_ids is not None and len(parent_ids) > 0:
                nested_nodes = await asyncio.gather(
                    *[
                        self.index.as_retriever(
                            similarity_top_k=similarity_top_k,
                            filters=MetadataFilters(
                                filters=[MetadataFilter(key="parent_id", value=id_)]
                            ),
                        ).aretrieve(query_str)
                        for id_ in parent_ids
                    ]
                )

                nodes = [node for nested in nested_nodes for node in nested]
                for node in nodes:
                    if node.id_ not in selected_node_ids:
                        selected_nodes.append(node)
                        selected_node_ids.add(node.id_)

                if self.verbose:
                    print(f"Retrieved {len(nodes)} from parents at level {level}.")

                level -= 1
                parent_ids = None

        return selected_nodes

    async def _aget(
        self,
        messages: Optional[List[ChatMessage]],
        mode: QueryModes = "tree_traversal",
        similarity_top_k: int = 2,
        last_n: int = 1,
        **kwargs,
    ) -> List[NodeWithScore]:
        """Retrieve nodes given query and mode."""
        import time  # local import to avoid module order issues
        t0 = time.time()  # **CHANGED**
        if len(messages) == 0:
            return ""
        if last_n > 0:
            messages = messages[-last_n:]
        else:
            messages = messages
        query_str = ""
        for msg in messages:
            query_str += msg.content + "\n"
        if mode == "tree_traversal":
            res = await self.tree_traversal_retrieval(query_str, similarity_top_k)
        elif mode == "collapsed":
            res = await self.collapsed_retrieval(query_str, similarity_top_k)
        else:
            raise ValueError(f"Invalid mode: {mode}")
        if not res:
            res = []
        print(f"Retrieved {len(res)} nodes from HVM.")
        res_str = ""
        for i, node in enumerate(res):
            res_str += f"<c_{i}>\n{node.node.text}\n</c_{i}>\n"
        self.retrieval_time += (time.time() - t0)  # **CHANGED**
        self.update_stats()
        return res_str

    async def aclose(self):
        # this is not working correctly, need to debug
        """Close any async resources (Qdrant aiohttp session, LLM/Emb clients, etc.)."""
        # --- Close Qdrant Async client (primary culprit for aiohttp warnings) ---
        qdrant_clients = []
        # Possible locations depending on LlamaIndex internals

        try:
            sc = getattr(self.index, "storage_context", None)
        except Exception:
            sc = None
        if sc is not None:
            try:
                vs = getattr(sc, "vector_store", None)
            except Exception:
                vs = None
        else:
            vs = None

        for cand in (vs,):
            if cand is None:
                continue
            # QdrantVectorStore stores the async client typically as "_aclient" (sometimes "aclient")
            aclient = (
                getattr(cand, "_aclient", None)
                or getattr(cand, "aclient", None)
                or getattr(cand, "client", None)
            )
            if aclient is not None:
                qdrant_clients.append(aclient)
        print(
            f"@@@@@@@@@@@@@@@@@@@@@@@@@@@@ Closing {len(qdrant_clients)} Qdrant async clients."
        )
        # Close unique clients
        seen_ids = set()
        for qc in qdrant_clients:
            if id(qc) in seen_ids:
                continue
            seen_ids.add(id(qc))
            try:
                close_res = getattr(qc, "close", None)
                if close_res is None:
                    close_res = getattr(qc, "aclose", None)
                if close_res is not None:
                    res = close_res()
                    if asyncio.iscoroutine(res):
                        await res
            except Exception:
                pass



# Set a random seed for reproducibility
RANDOM_SEED = 224
random.seed(RANDOM_SEED)


def global_cluster_embeddings(
    embeddings: np.ndarray,
    dim: int,
    n_neighbors: Optional[int] = None,
    metric: str = "cosine",
) -> np.ndarray:
    if n_neighbors is None:
        n_neighbors = int((len(embeddings) - 1) ** 0.5)
    return umap.UMAP(
        n_neighbors=n_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)



def local_cluster_embeddings(
    embeddings: np.ndarray, dim: int, num_neighbors: int = 10, metric: str = "cosine"
) -> np.ndarray:
    return umap.UMAP(
        n_neighbors=num_neighbors, n_components=dim, metric=metric
    ).fit_transform(embeddings)

#
# def get_optimal_clusters(
#     embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
# ) -> int:
#     max_clusters = min(max_clusters, len(embeddings))
#     n_clusters = np.arange(1, max_clusters)
#     bics = []
#     for n in n_clusters:
#         gm = GaussianMixture(n_components=n, random_state=random_state)
#         gm.fit(embeddings)
#         bics.append(gm.bic(embeddings))
#     return n_clusters[np.argmin(bics)]
#

# adjusted version of get_optimal_clusters to avoid numerical issue during clustering
def get_optimal_clusters(
    embeddings: np.ndarray, max_clusters: int = 50, random_state: int = RANDOM_SEED
) -> int:
    X = np.asarray(embeddings, dtype=np.float64)
    n, d = X.shape if X.ndim == 2 else (len(X), 1)
    if n <= 1:
        return 1

    # Robust caps:
    # - don't try more components than unique samples
    # - require ~2 samples per component to reduce singleton risk
    n_unique = len(np.unique(X, axis=0))
    hard_cap = max(1, min(max_clusters, n_unique, n // 2 if n >= 4 else 1))
    n_grid = np.arange(1, hard_cap + 1)

    # Feature scaling often fixes PD issues
    Xs = _stabilize_features(X)

    bics = []
    ks = []
    for k in n_grid:
        gm = _safe_fit_gmm(Xs, k, random_state)
        if gm is None:
            continue
        bics.append(gm.bic(Xs))
        ks.append(k)

    if not ks:
        # Last-resort: try a small Bayesian mixture that prunes components automatically
        try:
            bgm = BayesianGaussianMixture(
                n_components=min(5, max(1, n - 1)),
                covariance_type="diag",
                weight_concentration_prior_type="dirichlet_process",
                reg_covar=1e-3,
                random_state=random_state,
                max_iter=500,
            ).fit(Xs)
            # Effective components with non-trivial weight
            eff = (bgm.weights_ > (1.0 / max(10, n))).sum()
            return int(max(1, eff))
        except Exception:
            return 1

    return int(ks[int(np.argmin(bics))])


def GMM_cluster(embeddings: np.ndarray, threshold: float, random_state: int = 0):
    n_clusters = get_optimal_clusters(embeddings)
    gm = GaussianMixture(n_components=n_clusters, random_state=random_state)
    gm.fit(embeddings)
    probs = gm.predict_proba(embeddings)
    labels = [np.where(prob > threshold)[0] for prob in probs]
    return labels, n_clusters


# this is to avoid accesss issue when runing in multiple instance in async mode
_NUMBA_LOCK = threading.RLock()
# from numba import threading_layer
# print("Numba layer:", threading_layer())  # quick sanity check

def perform_clustering(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    with _NUMBA_LOCK:
        # If the number of embeddings is less than or equal to the dimension, return a list of zeros
        # This means all nodes are in the same cluster.
        # Otherwise, we will get an error when trying to cluster.
        if len(embeddings) <= dim + 1:
            return [np.array([0]) for _ in range(len(embeddings))]

        reduced_embeddings_global = global_cluster_embeddings(embeddings, dim)
        global_clusters, n_global_clusters = GMM_cluster(
            reduced_embeddings_global, threshold
        )

        all_local_clusters = [np.array([]) for _ in range(len(embeddings))]
        total_clusters = 0

        for i in range(n_global_clusters):
            global_cluster_embeddings_ = embeddings[
                np.array([i in gc for gc in global_clusters])
            ]

            if len(global_cluster_embeddings_) == 0:
                continue
            if len(global_cluster_embeddings_) <= dim + 1:
                local_clusters = [np.array([0]) for _ in global_cluster_embeddings_]
                n_local_clusters = 1
            else:
                reduced_embeddings_local = local_cluster_embeddings(
                    global_cluster_embeddings_, dim
                )
                local_clusters, n_local_clusters = GMM_cluster(
                    reduced_embeddings_local, threshold
                )

            for j in range(n_local_clusters):
                local_cluster_embeddings_ = global_cluster_embeddings_[
                    np.array([j in lc for lc in local_clusters])
                ]
                indices = np.where(
                    (embeddings == local_cluster_embeddings_[:, None]).all(-1)
                )[1]
                for idx in indices:
                    all_local_clusters[idx] = np.append(
                        all_local_clusters[idx], j + total_clusters
                    )

            total_clusters += n_local_clusters

        return all_local_clusters


def get_clusters(
    nodes: List[BaseNode],
    embedding_map: Dict[str, List[List[float]]],
    max_length_in_cluster: int = 10000,  # 10k tokens max per cluster
    tokenizer: Optional[
        Callable[[str], List[int]]
    ] = None,  # use tokenizer from llama_index
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length=None,  # to keep track of the total length of the previous clusters
) -> List[List[BaseNode]]:
    tokenizer = tokenizer or get_tokenizer()

    # get embeddings
    embeddings = np.array([np.array(embedding_map[node.id_]) for node in nodes])

    # Perform the clustering
    clusters = perform_clustering(
        embeddings, dim=reduction_dimension, threshold=threshold
    )

    # Initialize an empty list to store the clusters of nodes
    node_clusters = []

    # Iterate over each unique label in the clusters
    for label in np.unique(np.concatenate(clusters)):
        # Get the indices of the nodes that belong to this cluster
        indices = [i for i, cluster in enumerate(clusters) if label in cluster]

        # Add the corresponding nodes to the node_clusters list
        cluster_nodes = [nodes[i] for i in indices]

        # Base case: if the cluster only has one node, do not attempt to recluster it
        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        # Calculate the total length of the text in the nodes
        total_length = sum([len(tokenizer(node.text)) for node in cluster_nodes])

        # If the total length exceeds the maximum allowed length, recluster this cluster
        # If the total length did not change from the previous call then don't try again to avoid infinite recursion!
        if total_length > max_length_in_cluster and (
            prev_total_length is None or total_length < prev_total_length
        ):
            node_clusters.extend(
                get_clusters(
                    cluster_nodes,
                    embedding_map,
                    max_length_in_cluster=max_length_in_cluster,
                    tokenizer=tokenizer,
                    reduction_dimension=reduction_dimension,
                    threshold=threshold,
                    prev_total_length=total_length,
                )
            )
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters


# --- helper: robust GMM fit with retries ---
def _safe_fit_gmm(
    X: np.ndarray,
    n_components: int,
    random_state: int,
    *,
    cov_types: Sequence[str] = ("full", "diag", "spherical"),
    reg_grid: Sequence[float] = (1e-6, 1e-5, 1e-4, 1e-3),
    n_init: int = 2,
    max_iter: int = 300,
):
    X = np.asarray(X, dtype=np.float64)
    for cov in cov_types:
        for reg in reg_grid:
            try:
                gm = GaussianMixture(
                    n_components=n_components,
                    covariance_type=cov,
                    reg_covar=reg,
                    random_state=random_state,
                    n_init=n_init,
                    max_iter=max_iter,
                    init_params="kmeans",
                )
                gm.fit(X)
                return gm
            except Exception:
                continue
    return None


# --- optional: stabilize post-UMAP space (often helps small noisy batches) ---
def _stabilize_features(X: np.ndarray) -> np.ndarray:
    if X.size == 0:
        return X
    X = np.asarray(X, dtype=np.float64)
    # Standardize each dimension; avoids near-zero-variance axes breaking covariances
    return StandardScaler(with_mean=True, with_std=True).fit_transform(X)


# ---- assume the robust sync helpers you already have are imported here ----
# _safe_n_neighbors, global_cluster_embeddings, local_cluster_embeddings,
# get_optimal_clusters, GMM_cluster, perform_clustering  (all sync, robust)

TokenizeFn = Union[Callable[[str], List[int]], Callable[[str], Awaitable[List[int]]]]


async def _to_thread(func, /, *args, **kwargs):
    """Shorthand to offload sync CPU work to default threadpool."""
    return await asyncio.to_thread(func, *args, **kwargs)


async def _maybe_await_tokenize(tokenizer: TokenizeFn, text: str) -> List[int]:
    if asyncio.iscoroutinefunction(tokenizer):
        return await tokenizer(text)  # rare
    # Offload sync tokenization to a thread to avoid blocking loop when many nodes
    return await asyncio.to_thread(tokenizer, text)


async def _cluster_once_async(
    embeddings: np.ndarray,
    dim: int,
    threshold: float,
) -> List[np.ndarray]:
    # Offload the whole clustering pass; it calls NumPy/UMAP/sklearn in C
    return await _to_thread(perform_clustering, embeddings, dim, threshold)


async def _token_lengths_async(
    nodes: Sequence[BaseNode],
    tokenizer: TokenizeFn,
) -> List[int]:
    # Tokenize node.text concurrently (bounded by default threadpool size)
    tasks = [
        asyncio.create_task(_maybe_await_tokenize(tokenizer, n.text)) for n in nodes
    ]
    toks = await asyncio.gather(*tasks)
    return [len(t) for t in toks]


async def aget_clusters(
    nodes: List[BaseNode],
    embedding_map: Dict[str, List[float]],
    *,
    max_length_in_cluster: int = 10000,
    tokenizer: Optional[TokenizeFn] = None,
    reduction_dimension: int = 10,
    threshold: float = 0.1,
    prev_total_length: Optional[int] = None,
) -> List[List[BaseNode]]:
    """
    Async version of get_clusters:
      - Offloads UMAP/GMM and tokenization to threads
      - Safe for empty/degenerate cases
      - Preserves recursive re-clustering logic without blocking the event loop
    """
    if not nodes:
        return []

    tokenizer = tokenizer or get_tokenizer()

    # Build embeddings array off the loop (cheap, but keep IO loop clean)
    embeddings = np.array(
        [np.array(embedding_map[n.id_], dtype=float) for n in nodes], dtype=float
    )

    clusters = await _cluster_once_async(
        embeddings, dim=reduction_dimension, threshold=threshold
    )

    # Build label set safely
    label_set: set[int] = set()
    for arr in clusters:
        if arr.size:
            label_set.update(arr.tolist())

    if not label_set:
        # No labels → one big cluster
        return [nodes]

    node_clusters: List[List[BaseNode]] = []

    for label in sorted(label_set):
        indices = [i for i, arr in enumerate(clusters) if label in arr]
        cluster_nodes = [nodes[i] for i in indices]

        if len(cluster_nodes) == 1:
            node_clusters.append(cluster_nodes)
            continue

        # Token lengths without blocking loop
        lengths = await _token_lengths_async(cluster_nodes, tokenizer)
        total_length = sum(lengths)

        # Recluster if too large (and not stuck at same size)
        if total_length > max_length_in_cluster and (
            prev_total_length is None or total_length < prev_total_length
        ):
            # Recursive async call
            sub = await aget_clusters(
                cluster_nodes,
                embedding_map,
                max_length_in_cluster=max_length_in_cluster,
                tokenizer=tokenizer,
                reduction_dimension=reduction_dimension,
                threshold=threshold,
                prev_total_length=total_length,
            )
            node_clusters.extend(sub)
        else:
            node_clusters.append(cluster_nodes)

    return node_clusters


# SMOKE TESTS
# TODO will move the following code block in the future as regression test


def data_loader(file_name: str, is_abs_path: bool = False) -> list:
    """
    Load dataset from a JSON file.

    Args:
        file_name (str): Path to the JSON dataset file

    Returns:
        list: Dataset loaded from the JSON file

    Raises:
        FileNotFoundError: If the dataset file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    import json
    import os

    # Get the directory where this script lives
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    # If file_name is not an absolute path, make it relative to the script directory
    data_file = file_name

    # Check if file exists before loading
    if not os.path.exists(data_file):
        raise FileNotFoundError(f"Cannot find dataset file: {data_file}")

    print(f"Loading dataset from {data_file}...")

    try:
        with open(data_file, "r") as f:
            dataset = json.load(f)

        print(f"Dataset loaded with {len(dataset)} total questions")
        return dataset

    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(
            f"Invalid JSON in file {data_file}: {str(e)}", e.doc, e.pos
        )
    except Exception as e:
        raise Exception(f"Error loading dataset from {data_file}: {str(e)}")


import time


async def run_smoke_test(dataset, hvm: HierarchicalVectorMemory):
    print("Running smoke test...")
    print(f"Dataset contains {len(dataset)} items.")
    try:
        for q in dataset:
            print("--------------------------------")
            print("+++++++++++++++++++++++++++++++++++")
            print("--------------------------------")
            haystack_sessions = q.get("haystack_sessions", [])
            haystack_session_ids = q.get("haystack_session_ids")
            question_id = q.get("question_id")
            question = q.get("question")
            answer = q.get("answer")
            answer_session_ids = q.get("answer_session_ids", [])
            print(
                f"Processing question with {len(haystack_sessions)} haystack sessions."
            )

            total_turn = 0
            buffer = []
            for idx, session in enumerate(haystack_sessions):
                print(f"Processing haystack session {idx} with {len(session)} turns.")
                for turn in session:
                    content = turn["content"].replace(
                        "<|endoftext|>", ""
                    )  # Clean content to avoid tokenizer special-token errors

                    if turn["role"] == "user":
                        msg = ChatMessage(role="user", content=content)
                    elif turn["role"] == "assistant":
                        msg = ChatMessage(role="assistant", content=content)
                    else:
                        msg = None
                        print(f"Unknown role {turn['role']} in turn, skipping.")
                    if msg:
                        buffer.append(msg)
                        total_turn += 1
            await hvm.aput(buffer)
            print(
                f"Inserted all haystack sessions into HVM. Total turns inserted for this question: {total_turn}"
            )
            print(f"Question: {question}")
            response = await hvm.aget(
                messages=[ChatMessage(role="user", content=question)],
                mode=QueryModes.tree_traversal,
                similarity_top_k=1,
                last_n=1,
            )
            print(f"Retrieved context from HVM.\n{response}")
            print(f"Answer: {answer}")
    finally:
        print("Closing HVM...")
        await hvm.aclose()


if __name__ == "__main__":
    time_start = time.time()
    print("Creating HVM...")

    # Example usage of data_loader
    try:
        # Try to load the default dataset file
        dataset = data_loader("/Users/judyyu/memagents/asdrp/eval/data/custom_history/longmemeval_m_sample5_20.json")
        # dataset = data_loader("longmemeval_single_500.json")
        print(f"Successfully loaded dataset with {len(dataset)} items")

        # Print first item keys to show structure
        # if dataset:
        #     print(f"First item keys: {list(dataset[0].keys())}")

    except FileNotFoundError as e:
        print(f"Dataset file not found: {e}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
    q_client = create_qdrant_client()
    hvm = create_hvm(
        client=q_client,
        name="test_hvm",
        collection_name="test_hvm_collection" + str(uuid.uuid4()),
        tree_depth=3,
    )
    print("HVM created.")
    asyncio.run(run_smoke_test(dataset, hvm))
    time_end = time.time()
    print(f"Total time taken: {time_end - time_start:.2f} seconds")
