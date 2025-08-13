#############################################################################
# hierarchical_vector_agent.py
#
# agent that uses HierarchicalVectorMemory block with qdrant database
#
# @author Judy Yu
# @email  jyspark.yue@gmail.coom
#############################################################################


from __future__ import annotations
from llama_index.llms.openai import OpenAI
from asdrp.memory.hierarchical_vector_memory import HierarchicalVectorMemory


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

        # Long‑term memory store
        self.mem = HierarchicalVectorMemory(
            similarity_top_k=top_k, mode=retrieval_mode
        )


    def chat(self, user_msg: str) -> str:
        """One conversation turn: returns the assistant's reply."""

        # Retrieve relevant snippets from memory
        snippets = self.mem.retrieve(user_msg)
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
        self.mem.store(f"USER: {user_msg}")
        self.mem.store(f"ASSISTANT: {assistant_msg}")

        return assistant_msg


    def print_memory_tree(self, max_chars: int = 80) -> None:
        """Print the current hierarchical memory tree"""
        self.mem.print_tree(max_chars=max_chars)
