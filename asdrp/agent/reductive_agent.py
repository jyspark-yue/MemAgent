#############################################################################
# File: reductive_agent.py
#
# Description:
#   Agent for reductive reasoning by inferring propositions from conversation and then summarizing the conversation
#
# Authors:
#   @author     Theodore Mui (theodoremui@gmail.com)
#               - Created reductive_agent.py
#   @author     Eric Vincent Fernandes
#               - Implemented tracking for token/cost metrics
#
# Date:
#   Created:    July 4, 2025  (Theodore Mui)
#   Modified:   September 20, 2025 (Eric Vincent Fernandes)
#############################################################################

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio
from typing import List
import time
import re

from llama_index.core.agent.workflow import FunctionAgent, AgentOutput
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.core.memory import (
    Memory, InsertMethod
)
from llama_index.core.utils import count_tokens  # import token counter
from llama_index.llms.google_genai import GoogleGenAI  # use Gemini

from asdrp.agent.base import AgentReply
from asdrp.memory.proposition_extraction_memory import PropositionExtractionMemoryBlock

def get_default_llm(callback_manager=CallbackManager(handlers=[TokenCountingHandler()])) -> LLM:
    return GoogleGenAI(
        model="gemini-2.5-flash-lite",
        # temperature=0.0,
        max_retries=100,
        max_tokens=64000,
        callback_manager=callback_manager,
    )

class ReductiveAgent:
    def __init__(
        self,
        memory: Memory = None,
        tools=None,
    ):
        if tools is None:
            tools = []
        self.llm = get_default_llm()
        self.memory_block = PropositionExtractionMemoryBlock(
            name="proposition_extraction_memory",
            max_propositions=50,
        )
        self.memory = memory
        self.agent = self._create_agent(memory, tools)
        self.query_input_tokens = 0     # Number of tokens passed into the LLM within this agent
        self.query_output_tokens = 0    # Number of tokens returned by the LLM within this agent
        self.query_time = 0             # Duration of time the LLM took to respond

    async def achat(self, user_msg: str) -> AgentReply:
        try:
            # Prepend known propositions to the user message if available, with explicit instruction
            propositions = ""
            if self.memory and hasattr(self.memory, "memory_blocks"):
                for block in self.memory.memory_blocks:
                    if isinstance(block, PropositionExtractionMemoryBlock):
                        props = await block._aget()
                        if props:
                            propositions = (
                                "Known propositions from conversation so far:\n"
                                f"{props}\n"
                                "When answering, reference the known propositions above if relevant.\n"
                            )

            full_msg = propositions + user_msg

            # Track input tokens using count_tokens
            self.query_input_tokens = count_tokens(full_msg)

            initial_query_time = time.time()
            response = await self.agent.run(user_msg=full_msg, memory=self.memory)

            # Compute elapsed time for this question
            self.query_time = time.time() - initial_query_time

            # Track output tokens using count_tokens
            if isinstance(response, AgentOutput):
                self.query_output_tokens = count_tokens(response.response.content)
                return AgentReply(response_str=response.response.content)
            elif isinstance(response, ChatMessage):
                self.query_output_tokens = count_tokens(response.content)
                return AgentReply(response_str=response.content)
            else:
                self.query_output_tokens = count_tokens(str(response))
                return AgentReply(response_str=str(response))

        except Exception as e:
            self.query_time = 0
            self.query_input_tokens = 0
            self.query_output_tokens = 0
            print(f"Error in ReductiveAgent: {e}")
            return AgentReply(response_str="I'm sorry, I'm having trouble processing your request. Please try again.")

    def _create_agent(self, memory: Memory, tools: List[FunctionTool]) -> FunctionAgent:
        return FunctionAgent(
            llm=self.llm,
            memory=memory,
            tools=tools,
        )

    def _create_memory(self) -> Memory:
        return Memory.from_defaults(
            session_id="proposition_agent",
            token_limit=50,                       # size of the entire working memory
            chat_history_token_ratio=0.7,         # ratio of chat history to total tokens
            token_flush_size=10,                  # number of tokens to flush when memory is full
            insert_method=InsertMethod.SYSTEM,
            memory_blocks=[self.memory_block]
        )


#-----------------------------------------
# Main: proposition extraction smoke tests
#-----------------------------------------

import asyncio

def print_result(test_name, passed):
    print(f"{test_name}: {'PASSED' if passed else 'FAILED'}")

async def smoke_test_proposition_extraction():
    """Test that propositions are extracted and stored from user input."""
    memory = ReductiveAgent()._create_memory()
    agent = ReductiveAgent(memory=memory)
    await agent.achat("I believe the sky is blue.")
    # Find the proposition block
    block = next((b for b in memory.memory_blocks if isinstance(b, PropositionExtractionMemoryBlock)), None)
    props = await block._aget() if block else ""
    passed = ("sky is blue" in props.lower())
    print_result("Proposition extraction", passed)
    print(f"Extracted propositions: {props}")

async def smoke_test_proposition_deduplication():
    """Test that duplicate propositions are not stored multiple times."""
    memory = ReductiveAgent()._create_memory()
    agent = ReductiveAgent(memory=memory)
    await agent.achat("I believe the sky is blue.")
    await agent.achat("I believe the sky is blue.")
    block = next((b for b in memory.memory_blocks if isinstance(b, PropositionExtractionMemoryBlock)), None)
    props = await block._aget() if block else ""
    count = props.lower().count("sky is blue")
    passed = count == 1
    print_result("Proposition deduplication", passed)
    print(f"Extracted propositions: {props}")

async def smoke_test_multiple_propositions():
    """Test that multiple distinct propositions are extracted and stored."""
    memory = ReductiveAgent()._create_memory()
    agent = ReductiveAgent(memory=memory)
    await agent.achat("The sky is blue. Water is wet.")
    block = next((b for b in memory.memory_blocks if isinstance(b, PropositionExtractionMemoryBlock)), None)
    props = await block._aget() if block else ""
    props_lower = props.lower()
    passed = ("sky is blue" in props_lower and "water is wet" in props_lower)
    print_result("Multiple proposition extraction", passed)
    print(f"Extracted propositions: {props}")

async def smoke_test_proposition_limit():
    """Test that the max_propositions limit is respected."""
    block = PropositionExtractionMemoryBlock(
        name="proposition_extraction_memory",
        llm=get_default_llm(),
        max_propositions=2,
    )
    memory = Memory.from_defaults(
        session_id="test_session",
        token_limit=50,
        memory_blocks=[block],
    )
    agent = ReductiveAgent(memory=memory)
    await agent.achat("It is a fact that the sky is blue.")
    print("After 1st achat:", await block._aget())
    await agent.achat("It is a fact that water is wet.")
    print("After 2nd achat:", await block._aget())
    await agent.achat("It is a fact that grass is green.")
    props = await block._aget()
    print("After 3rd achat:", props)
    count = props.count("<proposition>")
    passed = count == 2
    print_result("Proposition limit respected", passed)
    print(f"Extracted propositions: {props}")

async def smoke_test_agent_reply_includes_propositions():
    """Test that the agent can reference extracted propositions in its reply."""
    memory = ReductiveAgent()._create_memory()
    agent = ReductiveAgent(memory=memory)
    await agent.achat("I believe the sky is blue.")
    reply = await agent.achat("What do you know about the sky?")
    passed = ("sky is blue" in reply.response_str.lower())
    print_result("Agent reply references proposition", passed)
    print(f"Agent reply: {reply.response_str}")

async def smoke_test_token_counting():
    """Test that input and output token counts are tracked correctly."""
    memory = ReductiveAgent()._create_memory()
    agent = ReductiveAgent(memory=memory)
    reply = await agent.achat("Testing token counting system.")
    passed_input = agent.query_input_tokens > 0
    passed_output = agent.query_output_tokens > 0
    print_result("Input token counting", passed_input)
    print_result("Output token counting", passed_output)
    print(f"Input tokens: {agent.query_input_tokens}, Output tokens: {agent.query_output_tokens}")

async def main():
    await smoke_test_proposition_extraction()
    await smoke_test_proposition_deduplication()
    await smoke_test_multiple_propositions()
    await smoke_test_proposition_limit()
    await smoke_test_agent_reply_includes_propositions()
    await smoke_test_token_counting()

if __name__ == "__main__":
    asyncio.run(main())

