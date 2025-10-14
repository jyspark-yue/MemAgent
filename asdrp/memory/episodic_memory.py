#############################################################################
# File: episodic_agent.py
#
# Description:
#   This memory block extracts episodic data: message timestamps, user input, agent output, location,
#   outcome (was the agent response accepted), reflection (learnings from interaction), and categorical tags.
#
# Authors:
#   @author     Eric Vincent Fernandes
#               - Created episodic_memory.py
#
# Date:
#   Created:    August 24, 2025 (Eric Vincent Fernandes)
#   Modified:   October 6, 2025 (Eric Vincent Fernandes)
#############################################################################

import re
import time
from datetime import datetime, timezone
from typing import List, Optional, Any
from uuid import uuid4

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.prompts import (RichPromptTemplate)
from llama_index.core.utils import count_tokens
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types
from pydantic import Field

_DEFAULT_EXTRACT_PROMPT = RichPromptTemplate(
    """
    You are a comprehensive information extraction system designed to identify key information from conversations.

    Conversation:
    {{chat_history}}

    INSTRUCTIONS:
    1. Summarize the user's overall intent and key messages in this conversation.
    2. Summarize the agent's reasoning process (assistant), decisions, and outputs in response to the user's prompts throughout the conversation, to record the agent's behavior in context and the path to the final outcome.
    3. Identify the outcome: was the user satisfied (positive, negative, or mixed-reaction) or was the goal accomplished?
    4. Extract location/context if mentioned, else null if not mentioned.
    5. Provide a thoughtful reflection: What inferred insights or learned lessons did the agent capture through the conversation.
    6. Generate 2 to 4 concise categorical tags (all lowercase) that capture the essence of the conversation and topics discussed within the conversation for easier search and retrieval.
    
    Return ONLY the requested information in this exact format, with no additional commentary or modifications:
    <results>
        <result>user summary</result>
        <result>agent summary</result>
        <result>outcome</result>
        <result>location</result>
        <result>reflection</result>
        <result>categorical tag 1</result>
        <result>categorical tag 2</result>
        <!-- More categorical tags as needed -->
    </results>
    
    If new results are present, but one specific result is missing, return the following for that specific result: <result>null</result>
    If no new results are present, return: <results></results>
    """
)

def _parse_memory_details_xml(xml_text: str) -> dict[str, str | list[str]]:
    """
    Parse memory details from XML-style text returned by the LLM.
    Expected format:
        <results>
            <result>user summary</result>
            <result>agent summary</result>
            <result>outcome</result>
            <result>location</result>
            <result>reflection</result>
            <result>tag 1</result>
            <result>tag 2</result>
            ...
        </results>
    """
    # Extract text inside each <result>...</result>
    matches = re.findall(r"<result>(.*?)</result>", xml_text, flags=re.DOTALL)

    # Clean up whitespace/newlines
    results = [m.strip() for m in matches if m.strip()]

    # Initialize base dict
    parsed = {
        "user_summary": None,
        "agent_summary": None,
        "outcome": None,
        "location": None,
        "reflection": None,
        "categorical_tags": []
    }

    # Assign values in order if present
    if len(results) >= 1:
        parsed["user_summary"] = results[0]
    if len(results) >= 2:
        parsed["agent_summary"] = results[1]
    if len(results) >= 3:
        parsed["outcome"] = results[2]
    if len(results) >= 4:
        parsed["location"] = results[3]
    if len(results) >= 5:
        parsed["reflection"] = results[4]

    # The remaining items (6+) are tags
    if len(results) > 5:
        parsed["categorical_tags"] = results[5:]

    return parsed

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

class EpisodicMemoryBlock(BaseMemoryBlock[str]):

    memory_episodes: List[Document] = Field(
        default_factory = list
    )
    index: Optional[VectorStoreIndex] = Field(
        default_factory = lambda: None,
    )
    llm: LLM = Field(
        default_factory = lambda: _get_default_llm(),
        description="The LLM to use for proposition extraction.",
    )
    input_tokens: int = Field(
        default=0, description="The number of tokens passed into the LLM when loading the chat history."
    )
    output_tokens: int = Field(
        default=0, description="The number of tokens returned by the LLM when loading the chat history."
    )
    load_chat_history_time: float = Field(
        default=0.0, description="The duration of time it took to load the chat history."
    )

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Processes a full session into a single summarized episodic memory (robust JSON handling).

        Minimal and robust: calls the summarization LLM once per session, then stores the session summary
        locally without triggering any embedding API calls during add.
        """
        if not messages:
            return

        start_time = time.time()

        # Build chat history (string) for prompt & counting
        raw_history = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in messages])
        formatted_prompt = _DEFAULT_EXTRACT_PROMPT.format(chat_history=raw_history)  # plain string

        self.input_tokens = count_tokens(raw_history + "\n" + formatted_prompt)

        try:
            raw_memory_details = await self.llm.acomplete(formatted_prompt)
            self.output_tokens = count_tokens(str(raw_memory_details))  # count output tokens from LLM
            memory_details = _parse_memory_details_xml(str(raw_memory_details))
        except Exception as e:
            print(f"Error parsing content from LLM: {e}")
            return

        # Extract fields safely
        user_summary = memory_details.get("user_summary", "").strip()
        agent_summary = memory_details.get("agent_summary", "").strip()

        if not user_summary or not agent_summary:
            print("ERROR: No user_summary or agent_summary found in LLM output: ", memory_details)
            return

        outcome = memory_details.get("outcome")
        location = memory_details.get("location")
        reflection = memory_details.get("reflection")
        categorical_tags = memory_details.get("categorical_tags", [])

        # Store without embeddings
        self.add_memory_episode(
            user_input=user_summary,
            agent_output=agent_summary,
            outcome=outcome,
            location=location,
            reflection=reflection,
            categorical_tags=categorical_tags,
        )

        self.load_chat_history_time = time.time() - start_time

    def add_memory_episode(
            self,
            user_input: str,
            agent_output: str,
            outcome: Optional[str] = None,
            location: Optional[str] = None,
            reflection: Optional[str] = None,
            categorical_tags: Optional[List[str]] = None,
    ) -> None:
        """Adds a new entry (episode/interaction/memory) into memory.

        NOTE: This implementation does NOT build an embedding index immediately.
        This avoids making embedding API calls per-add (prevents quota errors / long retries).
        Retrieval is done via a lightweight local matching function in _aget.
        """

        # Checks new entry with existing ones to prevent duplicates
        normalized_input = (user_input or "").strip().lower()
        for doc in self.memory_episodes:
            existing_input = doc.metadata.get("user_input", "").strip().lower()
            if existing_input == normalized_input and normalized_input != "":
                return

        memory_episode_id = str(uuid4())
        memory_episode_timestamp = datetime.now(timezone.utc).isoformat()
        categorical_tags = categorical_tags or []

        memory_episode_text = "\n".join([
            f"[USER INPUT]: {user_input}",
            f"[AGENT OUTPUT]: {agent_output}",
            f"[OUTCOME]: {outcome}",
            f"[LOCATION]: {location}",
            f"[REFLECTION]: {reflection}",
        ])

        metadata = {
            "id": memory_episode_id,
            "timestamp": memory_episode_timestamp,
            "user_input": user_input[:300],
            "agent_output": agent_output[:300],
            "location": location,
            "outcome": outcome,
            "reflection": reflection,
            "categorical_tags": categorical_tags,
        }

        memory_episode = Document(text=memory_episode_text.strip(), metadata=metadata)
        self.memory_episodes.append(memory_episode)

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **kwargs: Any) -> str:
        """
        Lightweight local retrieval (no embeddings): return up to 5 most relevant episodes by simple token overlap.
        Returns a concatenated string (same format your parse_memories expects).
        """
        if not messages:
            return "ERROR: No Query Provided"
        if not self.memory_episodes or not isinstance(self.memory_episodes, list):
            return "ERROR: No Memory Entries Stored"

        query = messages[-1].content.lower().strip()
        query_tokens = [t for t in re.split(r"\W+", query) if t]
        query_set = set(query_tokens)

        # Score each episode by token overlap across metadata fields and tags
        scored = []
        for doc in self.memory_episodes:
            u = (doc.metadata.get("user_input") or "").lower()
            a = (doc.metadata.get("agent_output") or "").lower()
            tags = doc.metadata.get("categorical_tags") or []
            combined_tokens = set(re.split(r"\W+", f"{u} {a} {' '.join(tags)}"))

            overlap = len(query_set & combined_tokens)
            tag_bonus = sum(any(q in tag.lower() for q in query_tokens) for tag in tags)
            score = overlap + tag_bonus

            scored.append((score, doc))

        # sort descending by score, if ties fall back to timestamp (most recent)
        scored.sort(key=lambda x: (x[0], x[1].metadata.get("timestamp", "")), reverse=True)

        # take top 5 non-zero score; if none have score>0, fallback to last 5 entries
        top_docs = [doc for (s, doc) in scored if s > 0][:5] or self.memory_episodes[-5:]

        return "\n".join([d.text for d in top_docs])

    def reset_memories(self) -> None:
        """Removes all entries (episodes/interactions/memories) stored in memory."""
        self.memory_episodes = []
        self.index = None