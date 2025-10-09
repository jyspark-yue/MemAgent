#############################################################################
# hvm_agent.py
#
# agent that uses HierarchicalVectorMemory block with qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.coom
#############################################################################




from __future__ import annotations

import asyncio
import os
from typing import List, Optional
import uuid

from asdrp.agent.base import AgentReply
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.llms.google_genai import GoogleGenAI
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from asdrp.memory.hvm import HierarchicalVectorMemory

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())


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
            model = "gemini-2.5-flash"  # Refer to LlamaIndex Google GenAI docs
        
        # Google GenAI reads API key from env var GOOGLE_API_KEY
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return GoogleGenAI(model=model, **kwargs)
    
    else:
        raise ValueError(f"Unsupported provider: {provider}. Only 'gemini' is supported in this version.")


def get_cost_per_1k_tokens(provider: str, model: str, token_type: str = "input") -> float:
    """
    Get cost per 1000 tokens for a given provider and model.
    
    Args:
        provider: LLM provider (only "gemini" supported in this version)
        model: Model name
        token_type: "input" or "output"
    
    Returns:
        Cost per 1000 tokens in USD
    """
    
    # Gemini pricing (as of 2024)
    gemini_pricing = {
        "models/gemini-2.5-flash-lite": {"input": 0.000075, "output": 0.0003},  # 使用与1.5-flash相同的定价
        "models/gemini-1.5-flash": {"input": 0.000075, "output": 0.0003},
        "models/gemini-1.5-pro": {"input": 0.00125, "output": 0.005},
        "models/gemini-1.0-pro": {"input": 0.0005, "output": 0.0015},
        "models/embedding-001": {"input": 0.000025, "output": 0.0},
    }
    
    if provider.lower() == "gemini":
        pricing = gemini_pricing.get(model, {"input": 0.000075, "output": 0.0003})  # Default fallback
    else:
        raise ValueError(f"Unsupported provider: {provider}. Only 'gemini' is supported in this version.")
    
    return pricing.get(token_type, 0.0)


class HVMAgent:
    """Conversational agent that uses hierarchical vector memory."""
    
    def __init__(
        self,
        provider: str = "gemini",
        model: str = "gemini-2.5-flash",
        top_k: int = 3,
        retrieval_mode: str = "collapsed",  # default to collapsed to enable Qdrant vectors
        tools: Optional[List[FunctionTool]] = None,
        temperature: float = 0.0,
        callback_manager: CallbackManager | None = None,
        # Qdrant / persistence options (only used in collapsed mode)
        qdrant_host: str = "localhost",
        qdrant_port: int = 6333,
        collection: Optional[str] = "agent_mem_hvm",
        persist_dir: Optional[str] = "storage/hvm_tree",
        reset_collection_on_init: bool = False,
    ) -> None:
        tools = tools or []

        # Configuration so memory reset can rebuild identical state
        self.provider = provider
        self.model = model
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

        # Inference configuration
        self.temperature = temperature
        self.callback_manager = callback_manager or CallbackManager(handlers=[TokenCountingHandler()])

        # Vector store / persistence configuration
        self.qdrant_host = qdrant_host
        self.qdrant_port = qdrant_port
        self.collection = collection
        self.persist_dir = persist_dir
        self.reset_collection_on_init = reset_collection_on_init

        # Create LLM using factory
        self.llm = create_llm(provider=provider, model=model, temperature=self.temperature, callback_manager=self.callback_manager)

        # Memory storage / state
        self.memory_block: HierarchicalVectorMemory | None = None
        self.memory = self._create_memory()
        self.agent = self._create_agent(self.memory, tools)
        
        # Token tracking for cost calculation
        import tiktoken
        self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("o200k_base")
        self.query_input_tokens = 0     # Number of tokens passed into the LLM within this agent
        self.query_output_tokens = 0    # Number of tokens returned by the LLM within this agent
        self.query_time = 0             # Duration of time the LLM took to respond


    async def achat(self, user_msg: str) -> AgentReply:
        import time
        import asyncio
        
        # Retrieve relevant snippets from memory (TreeIndex retriever uses sync LLM internally → run off loop)
        snippets = await asyncio.to_thread(self.memory_block._aget, user_msg)
        context = "\n".join(snippets)
        
        # Compose a single‑turn prompt
        prompt = (
            "You are a helpful, concise assistant.\n\n"
            f"Known context (may be empty):\n{context}\n\n"
            f"User: {user_msg}\n"
            "Assistant:"
        )
        
        # Track time
        initial_query_time = time.time()
        
        # Token deltas baseline
        handler = None
        try:
            handlers = getattr(self.llm.callback_manager, "handlers", []) or []
            handler = next((h for h in handlers if isinstance(h, TokenCountingHandler)), None)
        except Exception:
            handler = None
        p0 = handler.prompt_llm_token_count if handler else 0
        c0 = handler.completion_llm_token_count if handler else 0

        # Ask the LLM for a completion (native async)
        completion = await self.llm.acomplete(prompt)
        assistant_msg: str = (getattr(completion, "text", "") or "").strip()
        if not assistant_msg:
            # Signal to caller to retry following the evaluate script's logic
            raise ValueError("Response has no candidates")
        
        # Track tokens and time (prefer handler deltas; fallback to tokenizer)
        if handler:
            self.query_input_tokens = max(0, handler.prompt_llm_token_count - p0)
            self.query_output_tokens = max(0, handler.completion_llm_token_count - c0)
        else:
            self.query_input_tokens = len(self.tokenizer.encode(prompt))
            self.query_output_tokens = len(self.tokenizer.encode(assistant_msg))
        self.query_time = time.time() - initial_query_time
        
        # Store the turn to memory
        await self.memory_block._aput(f"USER: {user_msg}")
        await self.memory_block._aput(f"ASSISTANT: {assistant_msg}")
        
        return AgentReply(response_str=assistant_msg)


    def _create_memory(self) -> HierarchicalVectorMemory:
        """Initiate a new instance."""
        
        # If using a fixed collection in collapsed mode, optionally reset it before creating memory
        if self.retrieval_mode == "collapsed" and self.collection:
            try:
                import qdrant_client
                qclient = qdrant_client.QdrantClient(host=self.qdrant_host, port=self.qdrant_port)
                if self.reset_collection_on_init:
                    try:
                        qclient.delete_collection(collection_name=self.collection)
                    except Exception:
                        # Ignore if collection didn't exist
                        pass
            except Exception:
                # Qdrant not available or not needed when not using collapsed
                pass

        # Choose collection name
        collection_name = self.collection or f"agent_mem_hvm_{uuid.uuid4().hex}"
        memory_block = HierarchicalVectorMemory(
            collection=collection_name,
            similarity_top_k=self.top_k,
            mode=self.retrieval_mode,
            provider=self.provider,
            model=self.model,
            host=self.qdrant_host,
            port=self.qdrant_port,
            persist_dir=self.persist_dir,
        )

        self.memory_block = memory_block
        self.memory = memory_block
        return memory_block


    def _create_agent(self, memory: Memory, tools: List[FunctionTool]) -> FunctionAgent:
        """Recreate agent"""

        agent = FunctionAgent(
            llm=self.llm,
            memory=memory,
            tools=tools,
        )

        self.agent = agent
        return agent



    def print_memory_tree(self, max_chars: int = 80) -> None:
        """Print the current hierarchical memory tree"""
        self.memory_block.print_tree(max_chars=max_chars)
