# could implement a base class similar to this


from llama_index.core.memory import BaseMemoryBlock
from pydantic import Field

class BaseMemBlock(BaseMemoryBlock[str]):
    input_tokens: int = Field(
        default=0, description="The number of tokens passed into the LLM when loading the chat history."
    )
    output_tokens: int = Field(
        default=0, description="The number of tokens returned by the LLM when loading the chat history."
    )
    load_chat_history_time: float = Field(
        default=0.0, description="The duration of time it took to load the chat history."
    )
    
    llm: LLM = Field(description="LLM")

    embed_model: BaseEmbedding = Field(description="Embedding model")

    def update_stats(self):
        pass
