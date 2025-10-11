from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import time
from asdrp.agent.base import AgentReply

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.memory import VectorMemory
from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding

# cost
INPUT_COST_PER_1K = 0.00010
OUTPUT_COST_PER_1K = 0.00040


class VectorAgent:
    """
    initiate vector_agent using built in llama_index VectorMemory
    uses gemini-2.5-flash-lite
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.2,
        similarity_top_k: int = 3,
    ):
        # LLM model
        self.llm = Gemini(model=model, temperature=temperature)

        # use VectorMemory
        self.memory_block = VectorMemory.from_defaults(
            embed_model=GeminiEmbedding(model_name="text-embedding-004"),
            retriever_kwargs={"similarity_top_k": similarity_top_k},
        )

        # track runtime and cost
        self.input_tokens = 0
        self.output_tokens = 0
        self.load_chat_history_time = 0.0
        self.input_cost = 0.0
        self.output_cost = 0.0
        self.total_cost = 0.0

        self.query_input_tokens = 0
        self.query_output_tokens = 0
        self.query_time = 0.0
        self.query_input_cost = 0.0
        self.query_output_cost = 0.0
        self.query_total_cost = 0.0

    # cost estimate
    def _tok(self, text):
        return max(1, len((text or "").split()))

    async def _aput(self, messages):
        """
        push chat history to memory
        takes each ChatMessage one-by-one into VectorMemory
        """
        start = time.perf_counter()
        added_input = 0

        for m in messages:
            await self.memory_block.aput(m)
            added_input += self._tok(m.content)

        elapsed = time.perf_counter() - start

        self.input_tokens += added_input
        self.input_cost += (added_input / 1000.0) * INPUT_COST_PER_1K
        self.load_chat_history_time += elapsed
        self.total_cost = self.input_cost + self.output_cost

    async def achat(self, user_msg):
        """
        question the agent using the LLM + vector memory
        """
        start = time.perf_counter()

        # retrieve relevant context
        retriever = self.memory_block.as_retriever()
        hits = await retriever.aretrieve(user_msg)
        context = "\n".join(h.node.text if getattr(h, "node", None) else getattr(h, "text", "") for h in hits if h)

        prompt = (
                "You are a helpful, concise assistant.\n\n"
                f"Known context (may be empty):\n{context}\n\n"
                f"User: {user_msg}\n"
                "Assistant:"
            )

        # cost tracking for input
        self.query_input_tokens = self._tok(prompt)
        self.query_input_cost = (self.query_input_tokens / 1000.0) * INPUT_COST_PER_1K

        comp = await self.llm.acomplete(prompt)
        text = (comp.text or "").strip()

        # cost tracking for output
        self.query_output_tokens = self._tok(text)
        self.query_output_cost = (self.query_output_tokens / 1000.0) * OUTPUT_COST_PER_1K
        self.query_total_cost = self.query_input_cost + self.query_output_cost

        # store this conversation into memory
        await self.memory_block.aput(ChatMessage(role="user", content=user_msg))
        await self.memory_block.aput(ChatMessage(role="assistant", content=text))

        self.query_time = time.perf_counter() - start
        return AgentReply(response_str=text)
    

