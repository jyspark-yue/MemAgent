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
from llama_index.llms.gemini import Gemini
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
            model = "models/gemini-2.5-flash-lite" 
        
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY not found in environment variables")
        
        return Gemini(model=model, **kwargs)
    
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
        "models/gemini-2.5-flash-lite": {"input": 0.000075, "output": 0.0003},  
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
        provider: str = "openai",
        model: str = "o4-mini",
        top_k: int = 3,
        retrieval_mode: str = "tree_traversal",
        tools: Optional[List[FunctionTool]] = None,
    ) -> None:
        tools = tools or []

        # Configuration so memory reset can rebuild identical state
        self.provider = provider
        self.model = model
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

        # Create LLM using factory
        self.llm = create_llm(provider=provider, model=model)

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
        try:
            import time
            
            # Retrieve relevant snippets from memory
            snippets = self.memory_block._aget(user_msg)
            context = "\n".join(snippets)

            # Compose a single‑turn prompt
            prompt = (
                "You are a helpful, concise assistant.\n\n"
                f"Known context (may be empty):\n{context}\n\n"
                f"User: {user_msg}\n"
                "Assistant:"
            )

            # Track input tokens (prompt + user message)
            self.query_input_tokens = len(self.tokenizer.encode(prompt))
            
            # Track time
            initial_query_time = time.time()

            # Ask the LLM for a completion
            completion = self.llm.complete(prompt)
            assistant_msg: str = completion.text.strip()

            # Track output tokens and time
            self.query_output_tokens = len(self.tokenizer.encode(assistant_msg))
            self.query_time = time.time() - initial_query_time

            # Store the turn to memory
            await self.memory_block._aput(f"USER: {user_msg}")
            await self.memory_block._aput(f"ASSISTANT: {assistant_msg}")

            return AgentReply(response_str=assistant_msg)
        except Exception as e:
            # Reset token tracking on error
            self.query_time = 0
            self.query_input_tokens = 0
            self.query_output_tokens = 0
            print(f"Error in HVMAgent: {e}")
            return AgentReply(response_str="I'm sorry, I'm having trouble processing your request. Please try again.")


    def _create_memory(self) -> HierarchicalVectorMemory:
        """Initiate a new instance."""
        
        # Drop the previous collection
        if self.memory_block is not None:
            client = getattr(self.memory_block, "_client", None)
            vector_store = getattr(self.memory_block, "_vector_store", None)
            collection_name = getattr(vector_store, "collection_name", None)
            if client and collection_name:
                try:
                    client.delete_collection(collection_name=collection_name)
                except Exception:
                    pass
        
        collection_name = f"agent_mem_hvm_{uuid.uuid4().hex}"
        memory_block = HierarchicalVectorMemory(
            collection=collection_name,
            similarity_top_k=self.top_k,
            mode=self.retrieval_mode,
            provider=self.provider,
            model=self.model,
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
