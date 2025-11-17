#############################################################################
# File: summary_agent.py
#
# Description:
#   Agent for summarizing conversations
#
# Authors:
#   @author     Theodore Mui (theodoremui@gmail.com)
#               - Created summary_agent.py
#   @author     Eric Vincent Fernandes
#               - Implemented tracking for token/cost metrics
#               - Modified code to be compatible with Gemini (GenAI)
#
# Date:
#   Created:    July 4, 2025  (Theodore Mui)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#############################################################################

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import asyncio
import time
from typing import List

from llama_index.core.agent.workflow import FunctionAgent, AgentOutput
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.utils import count_tokens
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.tools import FunctionTool
from llama_index.core.llms import LLM
from llama_index.core.memory import (Memory, InsertMethod)
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types

from asdrp.agent.base import AgentReply
from asdrp.memory.condensed_memory import CondensedMemoryBlock


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
        temperature=0.2, # small randomness, still mostly deterministic
        max_retries=100,
        callback_manager=callback_manager,
        generation_config=gen_cfg,
    )


class SummaryAgent:
    def __init__(self, tools=None):
        if tools is None:
            tools = []
        self.llm = get_default_llm()
        self.memory_block = CondensedMemoryBlock(
            name="condensed_memory",
            token_limit=50
        )
        self.memory = self._create_memory()
        self.agent = self._create_agent(self.memory, tools)
        self.query_input_tokens = 0     # Number of tokens passed into the LLM within this agent
        self.query_output_tokens = 0    # Number of tokens returned by the LLM within this agent
        self.query_time = 0             # Duration of time the LLM took to respond

    async def achat(self, user_msg: str) -> AgentReply:
        try:

            full_msg = user_msg + str(await self.memory_block._aget())

            # Count tokens passed into the LLM within this agent
            self.query_input_tokens = count_tokens(full_msg)

            initial_query_time = time.time()

            response = await self.agent.run(user_msg=user_msg, memory=self.memory)

            # Compute elapsed time for this question
            self.query_time = time.time() - initial_query_time

            # Track output tokens using count_tokens
            if isinstance(response, AgentOutput):
                output_text = response.response.content
            elif isinstance(response, ChatMessage):
                output_text = response.content
            else:
                output_text = str(response)

            self.query_output_tokens = count_tokens(output_text)
            return AgentReply(response_str=output_text)

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
            token_limit=50,                       # size of the entire working memory (default defined by LlamaIndex)
            chat_history_token_ratio=0.7,         # ratio of chat history to total tokens
            token_flush_size=10,                  # number of tokens to flush when memory is full
            insert_method=InsertMethod.SYSTEM,
            memory_blocks=[self.memory_block]
        )

if __name__ == "__main__":

    # For running the agent with human input:
    agent = SummaryAgent()

    user_input = input("Enter your input: ")
    while user_input.strip() != "":
        reply = asyncio.run(agent.achat(user_input))
        print(f"Agent Response: {reply.response_str}")
        user_input = input("Enter your input: ")

    print("Thank you for chatting with me!")
    agent._create_memory()