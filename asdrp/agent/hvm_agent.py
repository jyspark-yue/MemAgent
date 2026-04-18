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

from asdrp.agent.base import AgentReply
from llama_index.core.agent.workflow import FunctionAgent
from asdrp.memory.hvm import HierarchicalVectorMemory
from llama_index.core.memory import Memory

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

class HVMAgent:
    """Conversational agent that uses hierarchical vector memory."""
    
    def __init__(
        self,
        model: str = "o4-mini",
        top_k: int = 3,
        retrieval_mode: str = "tree_traversal",
    ) -> None:
        # Llama‑Index’s OpenAI wrapper
        self.llm = OpenAI(model=model)

        # Memory storage
        self.memory_block = HierarchicalVectorMemory(
            similarity_top_k=top_k, mode=retrieval_mode
        )


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


    # def _create_agent(self, memory: Memory, tools: List[FunctionTool]) -> FunctionAgent:
    #     return FunctionAgent(
    #         llm=self.llm,
    #         memory=memory,
    #         tools=tools,
    #     )
    

    # def _create_memory(self) -> Memory:
    #     return Memory.from_defaults(
    #         session_id="proposition_agent",
    #         token_limit=50,                       
    #         chat_history_token_ratio=0.7,        
    #         token_flush_size=10,                 
    #         insert_method=InsertMethod.SYSTEM,
    #         memory_blocks=[self.memory_block]
    #     )
        


    def print_memory_tree(self, max_chars: int = 80) -> None:
        """Print the current hierarchical memory tree"""
        self.memory_block.print_tree(max_chars=max_chars)
