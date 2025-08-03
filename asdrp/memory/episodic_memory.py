#############################################################################
#   File:   episodic_memory.py
#   @author Eric Vincent Fernandes
#   @email  evfdes@gmail.com
#   @date   Aug 2, 2025
#
#   Extracts episodic data [timestamp, user input, agent output, location,
#       outcome (was agent response accepted), reflection (learnings from interaction), and categorical tags]
#############################################################################

from datetime import datetime, timezone
from typing import List, Optional, Any
from uuid import uuid4
import asyncio
import pprint
import json

from pydantic import Field

from llama_index.llms.openai import OpenAI
from llama_index.core import VectorStoreIndex, Document, Settings
from llama_index.core.base.llms.types import ChatMessage, ChatResponse
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.prompts import (RichPromptTemplate)

DEFAULT_EXTRACT_PROMPT = RichPromptTemplate(
    """
    You are a comprehensive information extraction system designed to identify key information from conversations.

    Conversation:
    {chat_history}

    INSTRUCTIONS:
    - Thoroughly identify and summarize aspects of the user's prompts and inputs that triggered the memory and continued the conversation in great detail. Include specific phrasing or context that heavily influenced the response path.
    - Thoroughly identify and summarize the agent's reasoning process, decisions, and outputs in response to the user's prompts throughout the conversation, to record the agent's behavior in context and the path to the final outcome.
    - Summarize the satisfaction level (outcome) of the user with the agent's final response (was it a positive, negative, or mixed reaction and why (clarity, usefulness, unresolved question, etc)).
    - Identify the specific location where the event in question took place solely based on the user's input, or null if not mentioned.
    - Provide a thoughtful reflection: What inferred insights or learned lessons did the agent capture through the conversation.
    - Generate 2 to 4 concise categorical tags (all lowercase) that capture the essence of the conversation and topics discussed within the conversation for easier search and retrieval.

    Respond ONLY in JSON format exactly as follows, with no additional commentary or formatting:
    {
        "user_input": "...",
        "agent_output": "...",
        "outcome": "...",
        "location": "...",
        "reflection": "...",
        "categorical_tags": ["...", "...", "...", "..."]
    }
    """
)

def get_default_llm() -> LLM:
    return Settings.llm

class EpisodicMemoryBlock(BaseMemoryBlock[str]):

    memory_episodes: List[Document] = Field(default_factory = list)
    index: Optional[VectorStoreIndex] = None
    llm: Optional[Any] = None

    def __init__(self, name: str = "episodic_memory_block"):
        super().__init__(name = name)
        self.llm = OpenAI()

    async def _aput(self, messages: List[ChatMessage]) -> None:

        if not messages:
            return

        msg_history = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in messages])     #   Combines the provided chat history into a single, formatted string
        prompt_text = DEFAULT_EXTRACT_PROMPT.format_messages(chat_history = msg_history)        #   Adds the provided chat history to the llm prompt

        response = await self.llm.achat(messages = prompt_text)                                 #   Runs llm model based on the prompt + chat history

        json_response = response.message.content.__str__()

        extracted_info = json.loads(json_response)

        self.add_memory_episode(
            user_input = extracted_info["user_input"],
            agent_output = extracted_info["agent_output"],
            outcome = extracted_info["outcome"],
            location = extracted_info["location"],
            reflection = extracted_info["reflection"],
            categorical_tags = extracted_info["categorical_tags"]
        )

    def add_memory_episode(
            self,
            user_input: str,
            agent_output: str,
            outcome: Optional[str] = None,                  #   Success of the agent's output
            location: Optional[str] = None,                 #   Location specified in interaction
            reflection: Optional[str] = None,               #   Key learnings from interaction
            categorical_tags: Optional[List[str]] = None,   #   Key tags to categorize the interaction
    ) -> None:
        """Adds a new entry (episode/interaction/memory) into memory."""

        for doc in self.memory_episodes:       #   Checks new entry with existing ones to prevent duplicates
            if doc.metadata.get("user_input", "") == user_input and doc.metadata.get("agent_output", "") == agent_output:
                return

        memory_episode_id = str(uuid4())                                    #   Generates a unique identifier per episode
        memory_episode_timestamp = datetime.now(timezone.utc).isoformat()   #   Records timestamp with timezone
        categorical_tags = categorical_tags or []                           #   Creates an empty list (separated from initialization to prevent mutability issues)

        memory_episode_text = f"""
            [USER INPUT]: {user_input}
            [AGENT OUTPUT]: {agent_output}
            [OUTCOME]: {outcome}
            [LOCATION]: {location}
            [REFLECTION]: {reflection}
        """         #   Saves key parts of interaction into memory

        metadata = {
            "id": memory_episode_id,
            "timestamp": memory_episode_timestamp,
            "user_input": user_input,
            "agent_output": agent_output,
            "location": location,
            "outcome": outcome,
            "reflection": reflection,
            "categorical_tags": categorical_tags,
        }           #   For use when searching through different memories

        memory_episode = Document(text = memory_episode_text.strip(), metadata = metadata)
        self.memory_episodes.append(memory_episode)

        if self.index is None:
            parser = SimpleNodeParser()                                     #   Breaks down large memories (Documents) into smaller chunks (nodes)
            nodes = parser.get_nodes_from_documents(self.memory_episodes)   #   Converts document objects into nodes
            self.index = VectorStoreIndex(nodes)                            #   Vectorizes nodes and stores in VectorStore (LlamaIndex)
        else:
            self.index.insert(memory_episode)                               #   Converts, vectorizes, and adds a new document into the VectorStore

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **kwargs: Any) -> str:
        """Returns a string of the 5 most relevant entries (episodes/interactions/memories) based on a given query."""

        if not messages:
            return "ERROR: No Query Provided"
        elif self.index is None:
            return "ERROR: No Memory Entries Stored"

        query = messages[-1].content                                                #   Identifies the query (most recent message)
        query_engine = self.index.as_query_engine(max_num_relevant = 5)             #   Configures response synthesizer for top X entries
        response = query_engine.query(query)                                        #   Finds the 5 most relevant entries

        return "\n".join([node.get_content() for node in response.source_nodes])    #   Extracts and returns content in relevant memory entries

    def get_all_memories(self) -> List[str]:
        """Returns a list of all entries (episodes/interactions/memories) stored in memory."""
        return [doc.text for doc in self.memory_episodes]

    def reset_memories(self) -> None:
        """Removes all entries (episodes/interactions/memories) stored in memory."""
        self.memory_episodes = []
        self.index = None

#-------------------------------------
# Main: smoke tests
#-------------------------------------

async def smoke_test():
    print("\n--- EpisodicMemoryBlock Smoke Tests ---\n")
    mem = EpisodicMemoryBlock()

    # Test 1: Add simple user-assistant exchange
    msg1 = ChatMessage(
        role="user",
        content="How do I boil an egg?",
        additional_kwargs={}
    )
    msg2 = ChatMessage(
        role="assistant",
        content="Place eggs in boiling water for 9 minutes.",
        additional_kwargs={}
    )
    await mem._aput([msg1, msg2])
    print("Test 1 - Basic user-assistant interaction added:")
    pprint.pprint(await mem._aget([msg1]))

    # Test 2: Duplicate message pair (should not be added)
    await mem._aput([msg1, msg2])
    all_memories = mem.get_all_memories()
    print("Test 2 - Duplicate interaction not added again (should match count=1):")
    pprint.pprint(len(all_memories))

    # Test 3: Add different interaction
    msg3 = ChatMessage(role="user", content="What's the capital of France?", additional_kwargs={})
    msg4 = ChatMessage(role="assistant", content="Paris is the capital of France.", additional_kwargs={})
    await mem._aput([msg3, msg4])
    print("Test 3 - Second distinct memory added:")
    pprint.pprint(await mem._aget([msg3]))

    # Test 4: Retrieve all memories (get_all_memories)
    print("Test 4 - All stored memory episodes:")
    pprint.pprint(mem.get_all_memories())

    # Test 5: Query irrelevant info (should still return closest)
    query_msg = ChatMessage(role="user", content="What is your favorite color?", additional_kwargs={})
    print("Test 5 - Query with no strong match:")
    pprint.pprint(await mem._aget([query_msg]))

    # Test 6: Reset memory
    mem.reset_memories()
    print("Test 6 - After reset (should be empty):")
    pprint.pprint(mem.get_all_memories())

    # Test 7: Add message with no assistant response (edge case)
    single_msg = ChatMessage(role="user", content="Tell me about string theory.", additional_kwargs={})
    await mem._aput([single_msg])
    print("Test 7 - Only user message (no agent):")
    pprint.pprint(await mem._aget([single_msg]))

    # Test 8: Long conversation chain (multi-message)
    long_convo = [
        ChatMessage(role="user", content="Plan a 3-day trip to Tokyo.", additional_kwargs={}),
        ChatMessage(role="assistant", content="Here's an itinerary...", additional_kwargs={}),
        ChatMessage(role="user", content="Make it more food-focused.", additional_kwargs={}),
        ChatMessage(role="assistant", content="Here’s an updated food tour version...", additional_kwargs={}),
    ]
    await mem._aput(long_convo)
    print("Test 8 - Multi-turn conversation stored:")
    pprint.pprint(await mem._aget([long_convo[-1]]))

if __name__ == "__main__":
    asyncio.run(smoke_test())