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
#   Created:    August 5, 2025 (Eric Vincent Fernandes)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#############################################################################

import re
import json
import time
from datetime import datetime, timezone
from typing import List, Optional, Any
from uuid import uuid4

from llama_index.core import VectorStoreIndex, Document
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemoryBlock
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.prompts import (RichPromptTemplate)
from llama_index.core.utils import count_tokens
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types
from pydantic import Field

DEFAULT_EXTRACT_PROMPT = RichPromptTemplate(
    """
    You are a comprehensive information extraction system designed to identify key information from conversations.

    Conversation:
    {{chat_history}}

    INSTRUCTIONS:
    - Summarize the user's overall intent and key messages in this conversation.
    - Summarize the agent's reasoning process (assistant), decisions, and outputs in response to the user's prompts throughout the conversation, to record the agent's behavior in context and the path to the final outcome.
    - Identify the outcome: was the user satisfied (positive, negative, or mixed-reaction) or was the goal accomplished?
    - Extract location/context if mentioned, else null if not mentioned.
    - Provide a thoughtful reflection: What inferred insights or learned lessons did the agent capture through the conversation.
    - Generate 2 to 4 concise categorical tags (all lowercase) that capture the essence of the conversation and topics discussed within the conversation for easier search and retrieval.

    Respond ONLY in JSON format exactly as follows, with no additional commentary or formatting:
    {
        "user_summary": "...",
        "agent_summary": "...",
        "outcome": "...",
        "location": "...",
        "reflection": "...",
        "categorical_tags": ["...", "...", "...", "..."]
    }
    """
)

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
        temperature=0.2,
        max_retries=100,
        callback_manager=callback_manager,
        generation_config=gen_cfg,
    )

class EpisodicMemoryBlock(BaseMemoryBlock[str]):

    memory_episodes: List[Document] = Field(
        default_factory = list
    )
    index: Optional[VectorStoreIndex] = Field(
        default_factory = lambda: None,
    )
    llm: LLM = Field(
        default_factory = lambda: get_default_llm(),
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
        msg_history = "\n".join([f"{msg.role.upper()}: {msg.content}" for msg in messages])
        prompt_str = DEFAULT_EXTRACT_PROMPT.format(chat_history=msg_history)  # plain string

        # Safe token counting
        try:
            self.input_tokens = count_tokens(msg_history + "\n" + prompt_str)
        except Exception:
            self.input_tokens = 0

        # Compose llm messages: pass original messages (if useful) and then the instruction
        llm_messages = []
        if isinstance(messages, list) and all(hasattr(m, "role") and hasattr(m, "content") for m in messages):
            llm_messages.extend(messages)
        llm_messages.append(ChatMessage(role="user", content=prompt_str))

        response = None
        try:
            response = await self.llm.achat(messages=llm_messages)
            raw = (response.message.content or "").strip()
            if not raw:
                print("LLM returned empty content for episodic extraction.")
                return

            # strip code fences/backticks often inserted by LLMs
            if raw.startswith("```") and raw.endswith("```"):
                lines = raw.splitlines()
                if len(lines) >= 3:
                    raw = "\n".join(lines[1:-1]).strip()
                else:
                    raw = raw.strip("`").strip()

            raw = re.sub(r"^`+|`+$", "", raw).strip()

            # parse JSON robustly: try direct parse then fallback to first {...} substring
            # try:
            #     extracted_info = json.loads(raw)
            # except json.JSONDecodeError:
            #     # non-greedy match for the first JSON object
            #     m = re.search(r"\{.*?\}", raw, flags=re.S)
            #     if not m:
            #         print("LLM output is not valid JSON. Raw output:\n", raw)
            #         return
            #     json_str = m.group(0)
            #     try:
            #         extracted_info = json.loads(json_str)
            #     except json.JSONDecodeError as e:
            #         print("Failed to parse JSON substring. Raw output:\n", raw)
            #         print("JSON error:", e)
            #         return

            # Try direct parse
            try:
                extracted_info = json.loads(raw)  # try raw (if raw already contains a JSON object)
            except json.JSONDecodeError:
                # If raw might contain additional text, try to extract JSON substring first
                m = re.search(r"\{.*?\}", raw, flags=re.S)
                if m:
                    json_str = m.group(0)
                else:
                    print("LLM output is not valid JSON and no JSON substring found. Raw output:\n", raw)
                    return

                # First try the substring directly
                try:
                    extracted_info = json.loads(json_str)
                except json.JSONDecodeError as e1:
                    # Common cause: unescaped backslashes inside code (LaTeX, regex, etc.)
                    # Escape backslashes that are NOT followed by valid JSON escape characters:
                    # valid JSON escapes after backslash are: " \ / b f n r t u
                    fixed_json_str = re.sub(r'\\(?!["\\/bfnrtu])', r'\\\\', json_str)

                    try:
                        extracted_info = json.loads(fixed_json_str)
                        # OK: we succeeded after escaping invalid backslashes
                    except json.JSONDecodeError as e2:
                        print("Failed to parse JSON substring. Raw output:\n", raw)
                        print("First JSON error:", e1)
                        print("Second JSON error after escaping backslashes:", e2)
                        return

        except Exception as e:
            print(f"LLM call / parsing exception: {e}")
            if response is not None:
                print("LLM raw output:", getattr(response.message, "content", None))
            return

        # count output tokens
        try:
            self.output_tokens = count_tokens(json.dumps(extracted_info))
        except Exception:
            self.output_tokens = 0

        # Fetch fields using the names your prompt uses
        user_summary = extracted_info.get("user_summary", "").strip()
        agent_summary = extracted_info.get("agent_summary", "").strip()
        if not user_summary and not agent_summary:
            print("No user_summary or agent_summary found in LLM output:", extracted_info)
            return

        outcome = extracted_info.get("outcome")
        location = extracted_info.get("location")
        reflection = extracted_info.get("reflection")
        categorical_tags = extracted_info.get("categorical_tags", [])

        # Store summary WITHOUT building embeddings or an index right now
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
        """Lightweight local retrieval (no embeddings): return up to 5 most relevant episodes by simple token overlap.

        Returns a concatenated string (same format your parse_memories expects).
        """
        if not messages:
            return "ERROR: No Query Provided"
        if not self.memory_episodes:
            return "ERROR: No Memory Entries Stored"

        query = messages[-1].content.lower().strip()
        query_tokens = [t for t in re.split(r"\W+", query) if t]

        # Score each episode by token overlap across metadata fields and tags
        scored = []
        for doc in self.memory_episodes:
            score = 0
            # check metadata fields (user_input, agent_output, tags)
            u = (doc.metadata.get("user_input") or "").lower()
            a = (doc.metadata.get("agent_output") or "").lower()
            tags = doc.metadata.get("categorical_tags") or []
            combined = " ".join([u, a, " ".join(tags)]).lower()
            for tk in query_tokens:
                if tk and tk in combined:
                    score += 1
            # small boost for tag matches
            for tag in tags:
                if any(qt in tag.lower() for qt in query_tokens):
                    score += 1
            scored.append((score, doc))

        # sort descending by score, if ties fall back to timestamp (most recent)
        scored.sort(key=lambda x: (x[0], doc.metadata.get("timestamp", "")), reverse=True)

        # take top 5 non-zero score; if none have score>0, fallback to last 5 entries
        top_docs = [doc for (s, doc) in scored if s > 0][:5]
        if not top_docs:
            top_docs = self.memory_episodes[-5:]

        return "\n".join([d.text for d in top_docs])

    def get_all_memories(self) -> List[str]:
        """Returns a list of all entries (episodes/interactions/memories) stored in memory."""
        return [doc.text for doc in self.memory_episodes]

    def reset_memories(self) -> None:
        """Removes all entries (episodes/interactions/memories) stored in memory."""
        self.memory_episodes = []
        self.index = None