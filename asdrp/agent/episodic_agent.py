#############################################################################
# File: episodic_agent.py
#
# Description:
#   This agent is equipped with episodic memory.
#
# Authors:
#   @author     Eric Vincent Fernandes
#               - Created episodic_agent.py
#
# Date:
#   Created:    August 5, 2025 (Eric Vincent Fernandes)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#############################################################################

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import time
import asyncio
from typing import List

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.llms import LLM
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.utils import count_tokens
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types

from asdrp.agent.base import AgentReply
from asdrp.memory.episodic_memory import EpisodicMemoryBlock

_safety_settings = [
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

_gen_cfg = types.GenerateContentConfig(safety_settings=_safety_settings, temperature=0.2)

def _get_default_llm(callback_manager=CallbackManager(handlers=[TokenCountingHandler()])) -> LLM:
    return GoogleGenAI(
        model="gemini-2.5-flash-lite",
        temperature=0.2,
        max_retries=100,
        callback_manager=callback_manager,
        generation_config=_gen_cfg,
    )

class EpisodicAgent:

    def __init__(self):
        self.llm = _get_default_llm()
        self.memory_block = EpisodicMemoryBlock(name="episodic_memory")
        self.query_input_tokens = 0     # Number of tokens passed into the LLM within this agent
        self.query_output_tokens = 0    # Number of tokens returned by the LLM within this agent
        self.query_time = 0             # Duration of time the LLM took to respond

    async def achat(self, user_msg: str) -> AgentReply:
        try:
            initial_query_time = time.time()

            # Retrieves relevant memory episodes
            episodes = self.parse_memories(text = await self.memory_block._aget([ChatMessage(role = "user", content = user_msg)]))

            # Parses relevant memory episodes into ChatMessage objects
            relevant_chat_history = []
            for entry in episodes:
                # entry is a dict with keys like "[USER INPUT]", "[AGENT OUTPUT]", "[OUTCOME]" etc
                if "[USER INPUT]" in entry:
                    relevant_chat_history.append(ChatMessage(role="user", content=entry.get("[USER INPUT]", "")))
                if "[AGENT OUTPUT]" in entry:
                    agent_msg_content = entry.get("[AGENT OUTPUT]", "")
                    # Append metadata fields in a human-readable way
                    extra_info = []
                    if entry.get("[OUTCOME]"):
                        extra_info.append(f"Outcome: {entry.get('[OUTCOME]')}")
                    if entry.get("[LOCATION]"):
                        extra_info.append(f"Location: {entry.get('[LOCATION]')}")
                    if entry.get("[REFLECTION]"):
                        extra_info.append(f"Reflection: {entry.get('[REFLECTION]')}")
                    if extra_info:
                        agent_msg_content += "\n" + "\n".join(extra_info)
                    relevant_chat_history.append(ChatMessage(role="assistant", content=agent_msg_content))

            # Track input tokens using count_tokens
            full_msg = " ".join([m.content for m in relevant_chat_history]) + " " + user_msg
            self.query_input_tokens = count_tokens(full_msg)

            context_text = "\n".join([f"{m.role.capitalize()}: {m.content}" for m in relevant_chat_history])
            prompt = (
                f"User: {user_msg}\n"
                f"Known context (may be empty):\n{context_text}\n\n"
                "Assistant:"
            )

            completion = await self.llm.acomplete(prompt)
            output_text = completion.text.strip()

            self.query_output_tokens = count_tokens(output_text)
            self.query_time = time.time() - initial_query_time      # Compute elapsed time for this question
            return AgentReply(response_str = output_text)

        except Exception as e:
            self.query_time = 0
            self.query_input_tokens = 0
            self.query_output_tokens = 0
            print(f"Error in EpisodicAgent: {e}")
            return AgentReply(response_str = "I'm sorry, I'm having trouble processing your request. Please try again.")

    def parse_memories(self, text: str) -> List[dict[str, str]] :
        """Converts the unformatted string (bundled up memory episodes) into a list of dictionaries with str->str keys."""

        relevant_memories: List[dict[str, str]] = []
        current_entry: dict[str, str] = {}

        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue

            if line.startswith("[USER INPUT]"):
                current_entry = {}  # Creates a new dictionary for each memory episode
            if ": " in line:
                key, value = line.split(": ", 1)
                current_entry[key.strip()] = value.strip()  # Stores key-value pair into dictionary
            if line.startswith("[REFLECTION]"):
                if current_entry:
                    relevant_memories.append(current_entry) # Adds current dictionary (complete memory episode) to dictionary

        return relevant_memories

    def reset_memories(self):
        """Clear episodic memory."""
        self.memory_block.reset_memories()

#-------------------------------
#   SMOKE TESTS
#-------------------------------

# Helper functions
def print_result(test_name, passed):
    print(f"\n=== {test_name} ===")
    print(f"Result: {'PASSED' if passed else 'FAILED'}\n")

def print_memories(memories):
    if not memories:
        print("No memories stored.")
        return
    for i, mem in enumerate(memories, 1):
        print(f"Memory #{i}:\n{mem}\n{'-'*40}")

def contains_keywords(text, keywords):
    return any(kw.lower() in text.lower() for kw in keywords)

# Tests
async def test_memory_storage(agent):
    """Single episode is stored correctly."""
    agent.reset_memories()
    await agent.achat("Where is the event happening?")
    memories = agent.get_all_memories()
    passed = len(memories) == 1 and contains_keywords(memories[0], ["event", "convention"])
    print_result("Memory storage", passed)
    print_memories(memories)

async def test_memory_deduplication(agent):
    """Duplicate inputs should not create new episodes."""
    agent.reset_memories()
    await agent.achat("Where is the event happening?")
    await agent.achat("Where is the event happening?")  # duplicate
    memories = agent.get_all_memories()
    passed = len(memories) == 1
    print_result("Memory deduplication", passed)
    print_memories(memories)

async def test_multiple_episodes(agent):
    """Multiple distinct inputs create separate episodes."""
    agent.reset_memories()

    await agent.achat("Where is the event happening?")
    await agent.achat("What are the top restaurants in New York?")

    # Use _aget to retrieve the most relevant entries for each query
    event_memories = await agent.memory_block._aget([ChatMessage(role = "user", content = "event")])
    restaurant_memories = await agent.memory_block._aget([ChatMessage(role = "user", content = "restaurants new york")])

    passed = (
        contains_keywords(event_memories, ["event"]) and
        contains_keywords(restaurant_memories, ["restaurants", "new york"])
    )

    print_result("Multiple episode storage", passed)
    print("Event memory:\n", event_memories)
    print("Restaurant memory:\n", restaurant_memories)

async def test_memory_reset(agent):
    """Memory reset clears all stored episodes."""
    agent.reset_memories()
    await agent.achat("Where is the event happening?")
    agent.reset_memories()
    memories = agent.get_all_memories()
    passed = len(memories) == 0
    print_result("Memory reset", passed)

async def test_recall(agent):
    """Agent can recall previously stored information."""
    agent.reset_memories()
    await agent.achat("Tell me about the best restaurants in New York City.")
    reply = await agent.achat("What were the restaurants again?")
    passed = contains_keywords(reply.response_str, ["restaurants", "new york"])
    print_result("Recall query", passed)
    print(f"Agent reply:\n{reply.response_str}\n")

async def test_empty_input(agent):
    """Empty user input should not crash agent."""
    agent.reset_memories()
    reply = await agent.achat("")
    passed = isinstance(reply.response_str, str) and len(reply.response_str) > 0
    print_result("Empty input handling", passed)
    print(f"Agent reply:\n{reply.response_str}\n")

async def test_whitespace_input(agent):
    """Whitespace-only input is handled gracefully."""
    agent.reset_memories()
    reply = await agent.achat("   ")
    passed = isinstance(reply.response_str, str) and len(reply.response_str) > 0
    print_result("Whitespace input handling", passed)
    print(f"Agent reply:\n{reply.response_str}\n")

async def test_long_input(agent):
    """Very long inputs are processed without crashing."""
    agent.reset_memories()
    long_input = "Tell me about " + "restaurants " * 100
    reply = await agent.achat(long_input)
    passed = isinstance(reply.response_str, str) and len(reply.response_str) > 0
    print_result("Long input handling", passed)

async def test_numeric_input(agent):
    """Numeric input is processed correctly."""
    agent.reset_memories()
    reply = await agent.achat("1234567890")
    passed = isinstance(reply.response_str, str) and len(reply.response_str) > 0
    print_result("Numeric input handling", passed)

async def test_special_char_input(agent):
    """Special characters in input are handled correctly."""
    agent.reset_memories()
    reply = await agent.achat("!@#$%^&*()_+")
    passed = isinstance(reply.response_str, str) and len(reply.response_str) > 0
    print_result("Special character input", passed)

async def test_multiple_duplicate_inputs(agent):
    """Multiple repeated inputs only create one memory per unique query."""
    agent.reset_memories()
    for _ in range(5):
        await agent.achat("Where is the event happening?")
    memories = agent.get_all_memories()
    passed = len(memories) == 1
    print_result("Multiple duplicates handling", passed)

async def test_sequence_of_varied_inputs(agent):
    """Sequence of different inputs stored correctly."""
    agent.reset_memories()
    inputs = [
        "Where is the event happening?",
        "What time is the meeting?",
        "Who is attending?",
        "Remind me about the event location",
        "List the top restaurants in NYC"
    ]
    for inp in inputs:
        await agent.achat(inp)
    memories = agent.get_all_memories()
    passed = len(memories) == len(inputs)
    print_result("Sequence of varied inputs", passed)

async def test_reset_after_multiple(agent):
    """Reset works after multiple stored episodes."""
    agent.reset_memories()
    await agent.achat("Where is the event happening?")
    await agent.achat("What are the top restaurants in New York?")
    agent.reset_memories()
    memories = agent.get_all_memories()
    passed = len(memories) == 0
    print_result("Reset after multiple episodes", passed)

async def test_recall_after_reset(agent):
    """Recall returns empty or default after memory reset."""
    agent.reset_memories()
    await agent.achat("Tell me about NYC restaurants")
    agent.reset_memories()
    reply = await agent.achat("What were the restaurants again?")
    print("REPLY: " + reply.response_str)
    passed = contains_keywords(reply.response_str, ["sorry", "cannot recall", "could", "please", "clarify", "no information"])
    print_result("Recall after reset", passed)

async def test_edge_case_special_keywords(agent):
    """Input with keywords only is processed and stored."""
    agent.reset_memories()
    await agent.achat("event restaurant NYC")
    memories = agent.get_all_memories()
    passed = len(memories) == 1 and contains_keywords(memories[0], ["event", "restaurant", "nyc"])
    print_result("Special keywords input", passed)

async def test_memory_content_structure(agent):
    """Memory entries contain required sections (USER INPUT, AGENT OUTPUT, OUTCOME, LOCATION, REFLECTION)."""
    agent.reset_memories()
    await agent.achat("Where is the event happening?")
    memories = agent.get_all_memories()
    mem = memories[0]
    required_keys = ["USER INPUT", "AGENT OUTPUT", "OUTCOME", "LOCATION", "REFLECTION"]
    passed = all(key in mem for key in required_keys)
    print_result("Memory content structure", passed)

async def main():
    """Runs all test cases."""
    agent = EpisodicAgent()

    await test_memory_storage(EpisodicAgent())
    await test_memory_deduplication(EpisodicAgent())
    await test_multiple_episodes(EpisodicAgent())
    await test_memory_reset(EpisodicAgent())
    await test_recall(EpisodicAgent())
    await test_empty_input(EpisodicAgent())
    await test_whitespace_input(EpisodicAgent())
    await test_long_input(EpisodicAgent())
    await test_numeric_input(EpisodicAgent())
    await test_special_char_input(EpisodicAgent())
    await test_multiple_duplicate_inputs(EpisodicAgent())
    await test_sequence_of_varied_inputs(EpisodicAgent())
    await test_reset_after_multiple(EpisodicAgent())
    await test_recall_after_reset(EpisodicAgent())
    await test_edge_case_special_keywords(EpisodicAgent())
    await test_memory_content_structure(EpisodicAgent())

    print("\nAll episodic agent smoke tests completed.")
    agent.reset_memories()

if __name__ == "__main__":

    #   For running test cases, use this:
    # asyncio.run(main())

    #   For running the agent with human input, use this:
    agent = EpisodicAgent()

    user_input = input("Enter your input: ")
    while user_input.strip() != "":
        reply = asyncio.run(agent.achat(user_input))
        print(f"Agent Response: {reply.response_str}")
        user_input = input("Enter your input: ")

    print("Thank you for chatting with me!")
    agent.reset_memories()