#############################################################################
# File: proposition_extraction.py
#
# Description:
#   Class for proposition (facts, opinions, preferences, beliefs, experiences, and goals) extraction from conversations
#
# Authors:
#   @author     Theodore Mui (theodoremui@gmail.com)
#               - Created proposition_extraction.py
#   @author     Eric Vincent Fernandes
#               - Implemented tracking for token/cost metrics
#
# Date:
#   Created:    July 4, 2025  (Theodore Mui)
#   Modified:   September 20, 2025 (Eric Vincent Fernandes)
#############################################################################
import json
import re
import time
from typing import Any, List, Optional, Union

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.llms import LLM
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.prompts import (BasePromptTemplate, PromptTemplate, RichPromptTemplate)
from llama_index.llms.ollama import Ollama
from llama_index.core.utils import count_tokens

try:
    from transformers import AutoTokenizer
    _hf_tokenizer = AutoTokenizer.from_pretrained("gpt2", use_fast=True)
except Exception:
    _hf_tokenizer = None

try:
    from llama_index.core.utils import count_tokens as _llama_count_tokens
except Exception:
    _llama_count_tokens = None

DEFAULT_EXTRACT_PROMPT = RichPromptTemplate("""You are a comprehensive information extraction system designed to identify key propositions from conversations.

INSTRUCTIONS:
1. Review the conversation segment and existing propositions provided prior to this message
2. Extract specific, concrete propositions (facts, opinions, preferences, beliefs, experiences, and goals) the user has disclosed or important information discovered
3. Focus on all types of information including:
   - Factual information (preferences, personal details, requirements, constraints, context)
   - Opinions and beliefs (what the user thinks, feels, or believes about topics)
   - Preferences and choices (what the user likes, dislikes, or prefers)
   - Experiences and anecdotes (what the user has done or experienced)
   - Goals and intentions (what the user wants to achieve or plans to do)
4. Format each proposition as a separate <proposition> XML tag
5. Include both objective facts and subjective opinions - capture the full range of user-disclosed information
6. Do not duplicate information that are already in the existing propositions list

<existing_propositions>
{{ existing_propositions }}
</existing_propositions>

Return ONLY the extracted propositions in this exact format:
<propositions>
  <proposition>Specific proposition 1</proposition>
  <proposition>Specific proposition 2</proposition>
  <!-- More propositions as needed -->
</propositions>

If no new propositions are present, return: <propositions></propositions>""")

DEFAULT_CONDENSE_PROMPT = RichPromptTemplate("""You are a comprehensive proposition condensing system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the current list of existing propositions
2. Condense the propositions into a more concise list, less than {{ max_propositions }} propositions
3. Focus on all types of information including:
   - Factual information (preferences, personal details, requirements, constraints, context)
   - Opinions and beliefs (what the user thinks, feels, or believes about topics)
   - Preferences and choices (what the user likes, dislikes, or prefers)
   - Experiences and anecdotes (what the user has done or experienced)
   - Goals and intentions (what the user wants to achieve or plans to do)
4. Format each proposition as a separate <proposition> XML tag
5. Include both objective facts and subjective opinions - preserve the full range of user information
6. Do not duplicate propositions that are already in the existing propositions list

<existing_propositions>
{{ existing_propositions }}
</existing_propositions>

Return ONLY the condensed propositions in this exact format:
<propositions>
  <proposition>Specific proposition 1</proposition>
  <proposition>Specific proposition 2</proposition>
  <!-- More propositions as needed -->
</propositions>

If no new propositions are present, return: <propositions></propositions>""")


def get_default_llm() -> LLM:
    return Ollama(
        model="qwen3:8b",
        request_timeout=180.0,
        thinking=True,
        json_mode=True,
    )

class PropositionExtractionMemoryBlock(BaseMemoryBlock[str]):
    """
    A memory block that extracts key propositions from conversation history using an LLM.

    This block identifies and stores discrete propositions disclosed during the conversation,
    including facts, opinions, preferences, beliefs, experiences, and goals,
    structuring them in XML format for easy parsing and retrieval.
    """

    name: str = Field(
        default="ExtractedPropositions", description="The name of the memory block."
    )
    llm: LLM = Field(
        default_factory=lambda: get_default_llm(),
        description="The LLM to use for proposition extraction.",
    )
    propositions: List[str] = Field(
        default_factory=list,
        description="List of extracted propositions from the conversation.",
    )
    max_propositions: int = Field(
        default=50, description="The maximum number of propositions to store."
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
    proposition_extraction_prompt_template: BasePromptTemplate = Field(
        default=DEFAULT_EXTRACT_PROMPT,
        description="Template for the proposition extraction prompt.",
    )
    proposition_condense_prompt_template: BasePromptTemplate = Field(
        default=DEFAULT_CONDENSE_PROMPT,
        description="Template for the proposition condense prompt.",
    )

    @field_validator("proposition_extraction_prompt_template", mode="before")
    @classmethod
    def validate_proposition_extraction_prompt_template(
        cls, v: Union[str, BasePromptTemplate]
    ) -> BasePromptTemplate:
        if isinstance(v, str):
            if "{{" in v and "}}" in v:
                v = RichPromptTemplate(v)
            else:
                v = PromptTemplate(v)
        return v

    async def _aget(
        self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any
    ) -> str:
        """Return the current propositions as formatted text."""
        if not self.propositions:
            return ""

        return "\n".join([f"<proposition>{proposition}</proposition>" for proposition in self.propositions])

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Extract propositions from new messages and add them to the propositions list."""

        # Skip if no messages
        if not messages:
            return

        # Track before call
        start_time_ns = time.time_ns()

        # Format existing propositions for the prompt
        existing_propositions_text = ""
        if self.propositions:
            existing_propositions_text = "\n".join(
                [f"<proposition>{proposition}</proposition>" for proposition in self.propositions]
            )

        # Build prompt messages
        prompt_messages = self.proposition_extraction_prompt_template.format_messages(
            existing_propositions=existing_propositions_text,
        )

        # Combine message text for fallback token counting
        total_input_text = "".join([m.content for m in messages]) + existing_propositions_text

        # Send the request and capture wall-clock time
        call_start_wall = time.time()
        response_raw = await self.llm.achat(messages=[*messages, *prompt_messages])
        call_end_wall = time.time()

        # response_raw.message.content could be: JSON string, plain string, or already a dict.
        content_obj = None
        try:
            # Many LLM wrappers put the model output in response_raw.message.content
            content_obj = getattr(response_raw, "message", None)
            if content_obj is not None:
                # content_obj may be a structure with `.content`, or itself a dict/string
                if hasattr(content_obj, "content"):
                    content = getattr(content_obj, "content")
                else:
                    content = content_obj
            else:
                # fallback to the raw object
                content = response_raw
        except Exception:
            content = response_raw

        # Normalize into parsed dict if possible
        parsed = None
        response_text = ""
        # content might be dict already
        if isinstance(content, dict):
            parsed = content
        elif isinstance(content, str):
            # Try to parse JSON string; if it fails, keep as plain text
            try:
                parsed = json.loads(content)
            except Exception:
                parsed = None
                response_text = content
        else:
            # try to coerce to str
            try:
                content_str = str(content)
                try:
                    parsed = json.loads(content_str)
                except Exception:
                    parsed = None
                    response_text = content_str
            except Exception:
                parsed = None
                response_text = ""

        # If parsed is dict, try to extract a textual response in several common keys
        # Ollama style: keys such as "response", "message", "propositions", "candidates", etc.
        condense_extra_time = 0.0
        condense_prompt_eval = 0
        condense_eval = 0

        if isinstance(parsed, dict):
            # If Ollama returned structured 'propositions' list, convert to XML-like text to re-use XML parser
            if "propositions" in parsed and isinstance(parsed["propositions"], list):
                # Build an XML-like string so _parse_propositions_xml still works
                list_propositions = parsed["propositions"]
                # If items are dicts with 'content', prefer that
                xml_parts = []
                for item in list_propositions:
                    if isinstance(item, dict):
                        content_piece = item.get("content") or item.get("text") or str(item)
                    else:
                        content_piece = str(item)
                    xml_parts.append(f"<proposition>{content_piece}</proposition>")
                response_text = "<propositions>\n" + "\n".join(xml_parts) + "\n</propositions>"
            else:
                # Try common keys
                response_text = parsed.get("response") or parsed.get("message") or parsed.get("text") or ""
                # If response_text is list/dict again, stringify
                if isinstance(response_text, (dict, list)):
                    try:
                        response_text = json.dumps(response_text)
                    except Exception:
                        response_text = str(response_text)
        else:
            # parsed is None — response_text was set earlier (plain string) or empty
            response_text = response_text or ""

        # Tokens/time bookkeeping: prefer returned metrics if present, else fallback to tokenizer+wall-time
        # Ollama uses keys: prompt_eval_count, eval_count, total_duration (nanoseconds)
        if isinstance(parsed, dict) and ("prompt_eval_count" in parsed or "eval_count" in parsed or "total_duration" in parsed):
            try:
                self.input_tokens = int(parsed.get("prompt_eval_count", count_tokens(total_input_text)))
            except Exception:
                self.input_tokens = count_tokens(total_input_text)
            try:
                self.output_tokens = int(parsed.get("eval_count", count_tokens(str(response_text))))
            except Exception:
                self.output_tokens = count_tokens(str(response_text))
            # duration: nanoseconds in Ollama -> convert to seconds
            if "total_duration" in parsed:
                try:
                    self.load_chat_history_time = float(parsed.get("total_duration", 0)) / 1_000_000_000.0
                except Exception:
                    self.load_chat_history_time = call_end_wall - call_start_wall
            else:
                self.load_chat_history_time = call_end_wall - call_start_wall
        else:
            # fallback counting
            self.input_tokens = count_tokens(total_input_text)
            self.output_tokens = count_tokens(str(response_text))
            self.load_chat_history_time = call_end_wall - call_start_wall

        # Parse the extracted propositions (XML or structured converted to XML-like above)
        new_propositions = []
        # If parsed had explicit 'propositions' list, prefer to use that directly (avoid re-parsing XML)
        if isinstance(parsed, dict) and "propositions" in parsed and isinstance(parsed["propositions"], list):
            # Extract content fields from structured list
            for item in parsed["propositions"]:
                if isinstance(item, dict):
                    content_piece = item.get("content") or item.get("text") or None
                    if content_piece:
                        new_propositions.append(content_piece.strip())
                else:
                    new_propositions.append(str(item).strip())
        else:
            # Use the XML parser expecting <proposition> tags
            propositions_text = response_text or ""
            new_propositions = self._parse_propositions_xml(str(propositions_text))

        # Add new propositions to the list, avoiding exact-match duplicates
        for proposition in new_propositions:
            if proposition not in self.propositions:
                self.propositions.append(proposition)

        # Condense the propositions if they exceed the max_propositions
        if len(self.propositions) >= self.max_propositions:
            existing_propositions_text = "\n".join(
                [f"<proposition>{proposition}</proposition>" for proposition in self.propositions]
            )

            prompt_messages = self.proposition_condense_prompt_template.format_messages(
                existing_propositions=existing_propositions_text,
                max_propositions=self.max_propositions,
            )

            condense_start = time.time()
            condense_raw = await self.llm.achat(messages=[*messages, *prompt_messages])
            condense_end = time.time()

            # Normalize condense response similarly
            condense_content = None
            try:
                condense_content = getattr(condense_raw, "message", None)
                if condense_content is not None and hasattr(condense_content, "content"):
                    condense_content = getattr(condense_content, "content")
            except Exception:
                condense_content = condense_raw

            condense_parsed = None
            condense_text = ""
            if isinstance(condense_content, dict):
                condense_parsed = condense_content
            elif isinstance(condense_content, str):
                try:
                    condense_parsed = json.loads(condense_content)
                except Exception:
                    condense_parsed = None
                    condense_text = condense_content
            else:
                condense_text = str(condense_content or "")

            # extract tokens/time for condense
            if isinstance(condense_parsed, dict) and ("prompt_eval_count" in condense_parsed or "eval_count" in condense_parsed or "total_duration" in condense_parsed):
                try:
                    condense_prompt_eval = int(condense_parsed.get("prompt_eval_count", 0))
                except Exception:
                    condense_prompt_eval = count_tokens(existing_propositions_text)
                try:
                    condense_eval = int(condense_parsed.get("eval_count", 0))
                except Exception:
                    condense_eval = count_tokens(condense_text)
                if "total_duration" in condense_parsed:
                    try:
                        condense_extra_time = float(condense_parsed.get("total_duration", 0)) / 1_000_000_000.0
                    except Exception:
                        condense_extra_time = condense_end - condense_start
                else:
                    condense_extra_time = condense_end - condense_start
            else:
                # fallback
                condense_prompt_eval = count_tokens(existing_propositions_text)
                condense_eval = count_tokens(condense_text)
                condense_extra_time = condense_end - condense_start

            # update totals
            self.input_tokens += condense_prompt_eval
            self.output_tokens += condense_eval
            # Add the condense time to the load_chat_history_time
            self.load_chat_history_time += condense_extra_time

            # parse condensed propositions
            condensed_new_props = []
            if isinstance(condense_parsed, dict) and "propositions" in condense_parsed and isinstance(condense_parsed["propositions"], list):
                for item in condense_parsed["propositions"]:
                    if isinstance(item, dict):
                        c = item.get("content") or item.get("text") or None
                        if c:
                            condensed_new_props.append(c.strip())
                    else:
                        condensed_new_props.append(str(item).strip())
            else:
                # condense_text may be empty string if condense_parsed provided 'response' key
                if isinstance(condense_parsed, dict):
                    condense_text = condense_parsed.get("response") or condense_parsed.get("message") or condense_text
                    if isinstance(condense_text, (dict, list)):
                        condense_text = json.dumps(condense_text)
                condensed_new_props = self._parse_propositions_xml(condense_text or "")

            if condensed_new_props:
                self.propositions = condensed_new_props

        # Ensure load_chat_history_time has at least the measured wall time as a fallback
        total_wall = (time.time() - (start_time_ns / 1_000_000_000.0)) if start_time_ns else 0.0
        if self.load_chat_history_time <= 0:
            self.load_chat_history_time = total_wall

    def _parse_propositions_xml(self, xml_text: str) -> List[str]:
        """Parse propositions from XML format."""
        propositions = []

        # Extract content between <proposition> tags
        pattern = r"<proposition>(.*?)</proposition>"
        matches = re.findall(pattern, xml_text, re.DOTALL)

        # Clean up extracted propositions
        for match in matches:
            proposition = match.strip()
            if proposition:
                propositions.append(proposition)

        return propositions
