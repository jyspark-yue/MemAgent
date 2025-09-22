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
#
# Date:
#   Created:    July 2, 2025  (Theodore Mui)
#   Modified:   September 20, 2025 (Eric Vincent Fernandes)
#############################################################################

import asyncio
import pprint
import time
from typing import Any, List, Optional

import tiktoken
from llama_index.core.llms import ChatMessage, TextBlock
from llama_index.core.memory import BaseMemoryBlock
from pydantic import Field

DEFAULT_TOKEN_LIMIT = 50000

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
    tokenizer: tiktoken.Encoding = tiktoken.get_encoding("o200k_base")
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
        self.input_tokens = 0

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
                elif key != "session_id" and key != "tool_call_id":
                    kwargs[key] = val
            memory_str += f"\n({kwargs})" if kwargs else ""

            self.current_memory.append(memory_str)

            # Count tokens for this new message (input tokens)
            self.input_tokens += len(self.tokenizer.encode(memory_str))

        # ensure this memory block doesn't get too large
        message_length = sum(
            len(self.tokenizer.encode(message))
            for message in self.current_memory
        )
        while message_length > self.token_limit:
            self.current_memory = self.current_memory[1:]
            message_length = sum(
                len(self.tokenizer.encode(message))
                for message in self.current_memory
            )

        self.load_chat_history_time = time.time() - start_time

#-------------------------------------
# Main: smoke tests
#-------------------------------------

async def smoke_test():
    print("\n--- CondensedMemoryBlock Smoke Tests ---\n")
    mem = CondensedMemoryBlock(name="smoke", token_limit=100)
    # Test 1: Add a single message
    msg1 = ChatMessage(
        blocks=[TextBlock(text="Hello, world!")],
        additional_kwargs={}
    )
    await mem._aput([msg1])
    print("Test 1 - Single message:")
    pprint.pprint(await mem._aget())

    # Test 2: Add multiple messages
    msg2 = ChatMessage(
        blocks=[TextBlock(text="How are you?")],
        additional_kwargs={}
    )
    await mem._aput([msg2])
    print("Test 2 - Multiple messages:")
    pprint.pprint(await mem._aget())

    # Test 3: Add message with tool_calls
    msg3 = ChatMessage(
        blocks=[TextBlock(text="Tool call message")],
        additional_kwargs={
            "tool_calls": [
                {"function": {"name": "search", "arguments": {"q": "test"}}}
            ],
            "irrelevant": "should be included"
        }
    )
    await mem._aput([msg3])
    print("Test 3 - Message with tool_calls:")
    pprint.pprint(await mem._aget())

    # Test 4: Add message with empty text and only kwargs
    msg4 = ChatMessage(
        blocks=[],
        additional_kwargs={"foo": "bar"}
    )
    await mem._aput([msg4])
    print("Test 4 - Empty text, only kwargs:")
    pprint.pprint(await mem._aget())

    # Test 5: Exceed token limit (force trimming)
    # Use short token limit for demonstration
    mem2 = CondensedMemoryBlock(name="trim", token_limit=10)
    for i in range(5):
        msg = ChatMessage(
            blocks=[TextBlock(text=f"msg{i}")],
            additional_kwargs={}
        )
        await mem2._aput([msg])
    print("Test 5 - Exceed token limit (should trim):")
    pprint.pprint(await mem2._aget())

    # Test 6: Add message with session_id/tool_call_id (should be ignored)
    msg5 = ChatMessage(
        blocks=[TextBlock(text="Session/tool_call id test")],
        additional_kwargs={"session_id": "abc", "tool_call_id": "def", "keep": "yes"}
    )
    mem3 = CondensedMemoryBlock(name="sidtest", token_limit=100)
    await mem3._aput([msg5])
    print("Test 6 - session_id/tool_call_id ignored:")
    pprint.pprint(await mem3._aget())

if __name__ == "__main__":
    asyncio.run(smoke_test())





