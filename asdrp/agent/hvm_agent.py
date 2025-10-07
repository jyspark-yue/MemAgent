#############################################################################
# hvm_agent.py
#
# agent that uses HierarchicalVectorMemory block with qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.coom
#############################################################################




from __future__ import annotations
from llama_index.llms.openai import OpenAI

import asyncio

from typing import List, Optional
import uuid

from asdrp.agent.base import AgentReply
from llama_index.core.agent.workflow import FunctionAgent
from llama_index.core.memory import Memory
from llama_index.core.tools import FunctionTool
from asdrp.memory.hvm import HierarchicalVectorMemory

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class HVMAgent:
    """Conversational agent that uses hierarchical vector memory."""
    
    def __init__(
        self,
        model: str = "o4-mini",
        top_k: int = 3,
        retrieval_mode: str = "tree_traversal",
        tools: Optional[List[FunctionTool]] = None,
    ) -> None:
        tools = tools or []

        # Configuration so memory reset can rebuild identical state
        self.model = model
        self.top_k = top_k
        self.retrieval_mode = retrieval_mode

        # Llama‑Index’s OpenAI wrapper
        self.llm = OpenAI(model=model)

        # Memory storage / state
        self.memory_block: HierarchicalVectorMemory | None = None
        self.memory = self._create_memory()
        self.agent = self._create_agent(self.memory, tools)


    async def achat(self, user_msg: str) -> AgentReply:
        try:
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

            # Ask the LLM for a completion
            completion = self.llm.complete(prompt)
            assistant_msg: str = completion.text.strip()

            # Store the turn to memory
            self.memory_block.store(f"USER: {user_msg}")
            self.memory_block.store(f"ASSISTANT: {assistant_msg}")

            return AgentReply(response_str=assistant_msg)
        except Exception as e:
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
        )

        self.memory_block = memory_block
        self.memory = memory_block
        return memory_block


    def _create_agent(self, memory: Memory, tools: List[FunctionTool]) -> FunctionAgent:
        """Recreate agent"""

        agent = FunctionAgent(
            llm=self.llm,
            tools=tools,
        )

        setattr(agent, "memory", memory)
        self.agent = agent
        return agent



    def print_memory_tree(self, max_chars: int = 80) -> None:
        """Print the current hierarchical memory tree"""
        self.memory_block.print_tree(max_chars=max_chars)
