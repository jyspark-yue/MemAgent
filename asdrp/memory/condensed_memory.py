#############################################################################
# File: condensed_memory.py
#
# Description:
#   A condensed memory block that maintains context while staying within reasonable memory limits.
#
# Authors:
#   @author     Theodore Mui (theodoremui@gmail.com)
#               - Created summary_agent.py
#   @author     Eric Vincent Fernandes
#               - Implemented tracking for token/cost metrics
#               - Modified code to be compatible with Gemini (GenAI)
#
# Date:
#   Created:    July 2, 2025  (Theodore Mui)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#############################################################################

import time
from typing import Any, List, Optional

from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.utils import count_tokens
from pydantic import Field

DEFAULT_TOKEN_LIMIT = 60000

class CondensedMemoryBlock(BaseMemoryBlock[str]):
    """
    This class is a smart conversation buffer that maintains context while
    staying within reasonable memory limits.

    It condenses the conversation history into a single string, while 
    maintaining a token limit.

    It also includes additional kwargs, like tool calls, when needed.
    """
    current_memory: List[str] = Field(default_factory=list)
    token_limit: int = Field(default=DEFAULT_TOKEN_LIMIT)
    input_tokens: int = Field(default=0, description="The number of tokens passed into the LLM when loading the chat history.")
    output_tokens: int = Field(default=0, description="The number of tokens returned by the LLM when loading the chat history. (Unneeded Here)")
    load_chat_history_time: float = Field(default=0.0, description="The duration of time it took to load the chat history.")

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Return the current memory block contents."""
        return "\n".join(self.current_memory)

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Push messages into the memory block. (Only handles text content)"""

        # Skip if no messages
        if not messages:
            return

        start_time = time.time()

        # construct a string for each message
        for message in messages:
            text_contents = "\n".join(
                block.text
                for block in message.blocks
                if isinstance(block, TextBlock)
            )
            memory_str = text_contents if text_contents else ""
            kwargs = {}
            for key, val in message.additional_kwargs.items():
                if key == "tool_calls":
                    val = [
                        {
                            "name": tool_call["function"]["name"],
                            "args": tool_call["function"]["arguments"],
                        }
                        for tool_call in val
                    ]
                    kwargs[key] = val
                elif key not in ("session_id", "tool_call_id"):
                    kwargs[key] = val
            memory_str += f"\n({kwargs})" if kwargs else ""

            self.current_memory.append(memory_str)

            # Count tokens for this new message (input tokens)

        # ensure this memory block doesn't get too large
        message_length = sum(count_tokens(message) for message in self.current_memory)
        while message_length > self.token_limit:
            self.current_memory = self.current_memory[1:]
            message_length = sum(count_tokens(message) for message in self.current_memory)

        self.load_chat_history_time = time.time() - start_time