#############################################################################
# summary_agent.py
#
# agent for summarizing conversations
#
# @author Theodore Mui
# @email  theodoremui@gmail.com
# Fri Jul 04 11:30:53 PDT 2025
#############################################################################

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio
import time
from typing import List
import tiktoken

from llama_index.core.agent.workflow import FunctionAgent, AgentOutput
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.core.memory import (
    Memory, InsertMethod
)
from llama_index.llms.openai import OpenAI

from asdrp.agent.base import AgentReply
from asdrp.memory.condensed_memory import CondensedMemoryBlock

def get_default_llm(callback_manager=CallbackManager(handlers=[TokenCountingHandler()])) -> LLM:
    return OpenAI(model="o4-mini", callback_manager=callback_manager)

class SummaryAgent:
    def __init__(
        self,
        tools=None,
    ):
        if tools is None:
            tools = []
        self.llm = get_default_llm()
        self.memory_block = CondensedMemoryBlock(
            name="condensed_memory",
            token_limit=50
        )
        self.memory = self._create_memory()
        self.agent = self._create_agent(self.memory, tools)
        self.tokenizer: tiktoken.Encoding = tiktoken.get_encoding("o200k_base")
        self.query_input_tokens = 0
        self.query_output_tokens = 0
        self.query_time = 0

    async def achat(self, user_msg: str) -> AgentReply:
        try:
            # Measure tokens used by memory block/agent
            # self.query_input_tokens = len(self.tokenizer.encode(str(await self.memory_block._aget(None))))
            self.query_input_tokens = self.memory_block.input_tokens

            initial_query_time = time.time()

            response = await self.agent.run(user_msg=user_msg, memory=self.memory)

            # Compute query tokens and cost for this question
            self.query_time = time.time() - initial_query_time

            if isinstance(response, AgentOutput):
                self.query_output_tokens = len(self.tokenizer.encode(response.response.content))
                return AgentReply(response_str=response.response.content)
            elif isinstance(response, ChatMessage):
                self.query_output_tokens = len(self.tokenizer.encode(response.content))
                return AgentReply(response_str=response.content)
            else:
                self.query_output_tokens = len(self.tokenizer.encode(str(response)))
                return AgentReply(response_str=str(response))
        except Exception as e:
            self.query_time = 0
            self.query_input_tokens = 0
            self.query_output_tokens = 0
            print(f"Error in SummaryAgent: {e}")
            return AgentReply(response_str="I'm sorry, I'm having trouble processing your request. Please try again.")

    def _create_agent(self, memory: Memory, tools: List[FunctionTool]) -> FunctionAgent:
        return FunctionAgent(
            llm=self.llm,
            memory=memory,
            tools=tools,
        )
        
    def _create_memory(self) -> Memory:
        return Memory.from_defaults(
            session_id="summary_agent",
            token_limit=50,                       # size of the entire working memory 
            chat_history_token_ratio=0.7,         # ratio of chat history to total tokens
            token_flush_size=10,                  # number of tokens to flush when memory is full
            insert_method=InsertMethod.SYSTEM,
            memory_blocks=[self.memory_block]
        )
    

#-------------------------------------
# Main: smoke tests
#-------------------------------------

def print_result(test_name, passed):
    print(f"{test_name}: {'PASSED' if passed else 'FAILED'}")

async def smoke_test_summary_agent_basic():
    print("Running basic instantiation and chat test...")
    agent = SummaryAgent()
    reply = await agent.achat("Hello, agent!")
    print_result("Basic chat reply type", isinstance(reply, AgentReply))
    print(f"Reply: {reply.response_str}")

# async def smoke_test_summary_agent_with_memory():
#     print("Running memory persistence test...")
#     memory = Memory.from_defaults(session_id="test_session", token_limit=50)
#     agent = SummaryAgent(memory=memory)
#     msg1 = "Remember this: The sky is blue."
#     await agent.achat(msg1)
#     # Check memory contents
#     mem_items = list(memory.get_all())
#     print_result("Memory stores at least one item after chat", len(mem_items) > 0)
#     print(f"Memory contents: {mem_items}")

async def smoke_test_summary_agent_custom_tools():
    print("Running custom tools test...")
    def echo_tool(input_str: str) -> str:
        return f"Echo: {input_str}"
    tool = FunctionTool.from_defaults(fn=echo_tool, name="echo_tool", description="Echo the input string")
    agent = SummaryAgent(tools=[tool])
    reply = await agent.achat("Call the echo_tool with input_str='test123'")
    print_result("Custom tool reply contains 'Echo'", "Echo" in reply.response_str)
    print(f"Reply: {reply.response_str}")

async def main():
    await smoke_test_summary_agent_basic()
    # await smoke_test_summary_agent_with_memory()
    await smoke_test_summary_agent_custom_tools()
    print("All smoke tests completed.")


if __name__ == "__main__":
    asyncio.run(main())
    
