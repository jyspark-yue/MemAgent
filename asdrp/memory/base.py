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


    async def _retry_aput(self, buffer):
        last_exc = None
        for attempt in range(1, RETRY_ATTEMPTS + 1):
            try:
                await self._aput(buffer)
                last_exc = None
                break
            except ClientError as e:  # <-- explicitly catch Gemini API errors
                print(f"ClientError caught (attempt {attempt}): {e}")
                last_exc = e
                if getattr(e, "status", None) in [502, 503, 504]:
                    delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                    delay *= random.uniform(0.8, 1.2)  # add ±20% jitter
                    print(f"Transient server error {e.status}, backing off {delay}s...")
                    await asyncio.sleep(delay)
                    continue
                raise

            except RuntimeError as e:
                # specifically catch Gemini’s “Response was terminated early: MAX_TOKENS”
                print(f"Gemini hit content issue (attempt {attempt}): {e}")
                last_exc = e
                if "MAX_TOKENS" in str(e) or "PROHIBITED_CONTENT" in str(e):
                    # delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY - 5
                    )
                    delay = delay * random.uniform(0.8, 1.2)  # add ±20% jitter
                    print(f"Backing off {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                raise  # different RuntimeError

            except ValueError as e:
                print(f"No candidates detected (attempt {attempt}): {e}")
                last_exc = e
                if "no candidates" in str(e):
                    delay = min(
                        RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY + 10
                    )
                    delay = delay * random.uniform(0.8, 1.2)  # add ±20% jitter
                    print(f"Backing off {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                raise  # different RuntimeError

            except Exception as e:
                print(f"Error processing buffered turns (attempt {attempt}): {e}")
                traceback.print_exc()
                last_exc = e
                if "Rate limit" in str(e) or "429" in str(e):
                    # delay = RETRY_BASE_DELAY * (2 ** (attempt - 1))
                    delay = min(RETRY_BASE_DELAY * (2 ** (attempt - 1)), RETRY_MAX_DELAY)
                    delay = delay * random.uniform(0.8, 1.2)  # add ±20% jitter
                    print(f"RateLimit detected, backing off {delay}s before retry...")
                    await asyncio.sleep(delay)
                    continue
                raise
        if last_exc is not None:
            raise last_exc




    