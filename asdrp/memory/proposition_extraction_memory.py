#############################################################################
# File: proposition_extraction_memory.py
#
# Description:
#   Class for proposition (facts, opinions, preferences, beliefs, experiences, and goals) extraction from conversations
#
# Authors:
#   @author     Theodore Mui (theodoremui@gmail.com)
#               - Modified FactExtractionMemoryBlock (LlamaIndex) to create proposition_extraction_memory.py
#   @author     Eric Vincent Fernandes
#               - Implemented tracking for token/cost metrics
#               - Modified code to be compatible with Gemini (GenAI)
#
# Date:
#   Created:    July 4, 2025  (Theodore Mui)
#   Modified:   October 5, 2025 (Eric Vincent Fernandes)
#############################################################################

import re
import time
from typing import Any, List, Optional, Union

from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.callbacks import CallbackManager, TokenCountingHandler
from llama_index.core.llms import LLM
from llama_index.core.memory.memory import BaseMemoryBlock
from llama_index.core.prompts import (BasePromptTemplate, PromptTemplate, RichPromptTemplate)
from llama_index.core.utils import count_tokens
from llama_index.llms.google_genai import GoogleGenAI
from google.genai import types

DEFAULT_EXTRACT_PROMPT = RichPromptTemplate("""You are a precise proposition extraction system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the conversation segment and existing propositions provided prior to this message.
2. Extract specific, concrete propositions the user has disclosed or important information discovered
3. Focus on all types of information like preferences, personal details, requirements, constraints, or context
4. Format each proposition as a separate <proposition> XML tag.
5. Include both objective facts and subjective opinions - capture the full range of user-disclosed information.
6. Do not duplicate propositions that are already in the existing propositions list.
7. KEEP TOTAL OUTPUT UNDER 60000 TOKENS.

AFTER you draft your output:
- CHECK: Ensure the output is under 60000 tokens or 240,000 characters and contains no duplicate propositions.
- IF it fails length, rewrite or condense until it passes.

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

DEFAULT_CONDENSE_PROMPT = RichPromptTemplate("""You are a precise proposition condensing system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the current list of existing propositions.
2. Condense the propositions into a more concise list, less than {{ max_propositions }} propositions.
3. Focus on all types of information like preferences, personal details, requirements, constraints, or context
4. Format each proposition as a separate <proposition> XML tag.
5. Include both objective facts and subjective opinions - preserve the full range of user information.
6. Do not duplicate propositions that are already in the existing propositions list.
7. KEEP TOTAL OUTPUT UNDER 60000 TOKENS.

AFTER you draft your output:
- CHECK: Ensure the output is under 60000 tokens or 240,000 characters and contains no duplicate propositions.
- IF it fails length, rewrite or condense until it passes.

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
        start_time = time.time()

        # Format existing propositions for the prompt
        existing_propositions_text = ""
        if self.propositions:
            existing_propositions_text = "\n".join(
                [f"<proposition>{proposition}</proposition>" for proposition in self.propositions]
            )

        # Create the prompt
        prompt_messages = self.proposition_extraction_prompt_template.format_messages(existing_propositions=existing_propositions_text)

        # Get the propositions extraction
        response = await self.llm.achat(messages=[*messages, *prompt_messages])

        # Count input tokens using count_tokens
        total_input_text = "".join([m.content for m in messages]) + existing_propositions_text
        self.input_tokens = count_tokens(total_input_text)

        # Parse the XML response to extract propositions
        propositions_text = response.message.content or ""
        self.output_tokens = count_tokens(propositions_text)  # count output tokens

        new_propositions = self._parse_propositions_xml(propositions_text)

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
            response = await self.llm.achat(messages=[*messages, *prompt_messages])

            # count tokens for condense output
            self.input_tokens += count_tokens(existing_propositions_text)
            self.output_tokens += count_tokens(response.message.content or "")

            # Replace the propositions with the condensed list
            new_propositions = self._parse_propositions_xml(response.message.content or "")
            if new_propositions:
                self.propositions = new_propositions

        self.load_chat_history_time = time.time() - start_time

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
