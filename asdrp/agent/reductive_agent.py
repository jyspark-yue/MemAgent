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
#               - Modified code to be compatible with Gemini (GenAI)
#   @author     Varenya Garg
#               - Modified code to store and retrieve propositions using Qdrant vector database
#
# Date:
#   Created:    July 4, 2025  (Theodore Mui)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#   Modified:   April 5, 2026 (Varenya Garg)
#############################################################################

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

from asdrp.agent.BaseAgent import BaseAgent
import time
import asyncio
from asdrp.agent.base import AgentReply
from llama_index.core.base.llms.types import ChatMessage, MessageRole
from asdrp.memory.proposition_extraction_memory import PropositionExtractionMemoryBlock

class ReductiveAgent(BaseAgent):
    def __init__(self):
        super().__init__()
        self.memory_block = PropositionExtractionMemoryBlock(collection="agent_condensed_mem", max_propositions=50)

    async def achat(self, user_msg: str) -> AgentReply:
        try:
            initial_query_time = time.time()
            user_message = ChatMessage(role=MessageRole.USER, content=user_msg)

            # Prepend known propositions to the user message if available, with explicit instruction
            props = await self.memory_block._aget(messages=[user_message])
            if not props:
                return AgentReply(response_str="ERROR: NO PROPOSITIONS STORED")

            prompt = (
                f"User: {user_msg}\n"
                f"Known context (may be empty):\n{props}\n\n"
                "Assistant:"
            )

            # Track input tokens
            self.query_input_tokens = int(len(prompt) / 4)
            response = await self.llm.acomplete(prompt)
            self.query_output_tokens = int(len(response.text) / 4)
            output_text = response.text.strip()

            self.query_time = time.time() - initial_query_time      # Compute elapsed time for this question
            return AgentReply(response_str=output_text)

        except Exception as e:
            self.query_time = 0
            self.query_input_tokens = 0
            self.query_output_tokens = 0
            print(f"Error in ReductiveAgent: {e}")
            return AgentReply(response_str="I'm sorry, I'm having trouble processing your request. Please try again.")

if __name__ == "__main__":

    # For running the agent with human input:
    agent = ReductiveAgent()

    user_input = input("Enter your input: ")
    while user_input.strip() != "":
        reply = asyncio.run(agent.achat(user_input))
        print(f"Agent Response: {reply.response_str}")
        user_input = input("Enter your input: ")

    print("Thank you for chatting with me!")