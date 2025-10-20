from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import time
from asdrp.agent.base import AgentReply
from llama_index.core.agent.workflow import FunctionAgent

#changed import
#from llama_index.core.memory import VectorMemoryBlock
from llama_index.core.base.llms.types import ChatMessage


from asdrp.memory.vector_memory import VectorMemoryBlock

from llama_index.llms.gemini import Gemini
from llama_index.embeddings.gemini import GeminiEmbedding
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM

from llama_index.vector_stores.qdrant import QdrantVectorStore
from qdrant_client import QdrantClient, AsyncQdrantClient

safety_settings = [
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
    )
]

gen_cfg = types.GenerateContentConfig(safety_settings=safety_settings, temperature=0.2)

def get_default_llm(callback_manager=CallbackManager(handlers=[TokenCountingHandler()])) -> LLM:
    return GoogleGenAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_retries=10,
        callback_manager=callback_manager,
        generation_config=gen_cfg,
    )

class VectorAgent:
    """
    initiate vector_agent using built in llama_index VectorMemory
    uses gemini-2.5-flash-lite
    """

    def __init__(
        self,
        top_k: int = 3,
        collection: str = "vector agent",
        host: str = "localhost",
        port: int = 6333,
    ):
        # LLM model
        self.llm = get_default_llm()

        self._client = QdrantClient(host=host, port=port)
        self._aclient = AsyncQdrantClient(host=host, port=port)
        self._vector_store = QdrantVectorStore(
            client=self._client, 
            aclient=self._aclient, 
            collection_name=collection
        )

        # use VectorMemory
        self.memory_block = VectorMemoryBlock(
            vector_store=self._vector_store,
            embed_model=GeminiEmbedding(model_name="models/embedding-001"),
            similarity_top_k=top_k
        )

        self.query_input_tokens = 0
        self.query_output_tokens = 0
        self.query_time = 0.0

    def _create_memory(self):
        """Create a new memory instance for this agent."""
        return self.memory_block

    def _create_agent(self, memory, messages):
        """Create a new agent instance with the given memory."""
        return self

    async def achat(self, user_msg):
        """
        question the agent using the LLM + vector memory
        """
        try:
            start = time.time()
            
            # retrieve relevant context
            user_message = ChatMessage(role="user", content=user_msg)
            snippets = await self.memory_block._aget([user_message])
            context = snippets if snippets else ""

            prompt = (
                    "You are a helpful, concise assistant.\n\n"
                    f"Known context (may be empty):\n{context}\n\n"
                    f"User: {user_msg}\n"
                    "Assistant:"
                )
            
            self.query_input_tokens = int(len(prompt) / 4)
            comp = await self.llm.acomplete(prompt)

            text = (comp.text or "").strip()
            self.query_output_tokens = int(len(text) / 4)

            self.query_time = time.time() - start
            return AgentReply(response_str=text)
        
        except Exception as e:
            self.query_input_tokens = 0
            self.query_output_tokens = 0
            self.query_time = 0.0

            print(f"error in VectorAgent: {e}")

            return AgentReply(response_str = "I'm sorry, I'm having trouble processing your request. Please try again.")
