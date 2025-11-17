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
#   Modified:   October 16, 2025 (Eric Vincent Fernandes)
#############################################################################

import time
from typing import Any, List, Optional, Union
from llama_index.core.base.llms.types import ChatMessage
from llama_index.core.bridge.pydantic import Field, field_validator
from llama_index.core.prompts import (BasePromptTemplate, PromptTemplate, RichPromptTemplate)
from asdrp.memory.BaseMemBlock import BaseMemBlock

DEFAULT_EXTRACT_PROMPT = RichPromptTemplate("""
You are a precise proposition extraction system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the conversation segment and existing propositions provided prior to this message
2. Extract specific, concrete propositions (facts, opinions, preferences, beliefs, experiences, and goals) the user has disclosed or important information discovered
3. Focus on all types of information including:
    - Factual information (preferences, personal details, requirements, constraints, context)
    - Opinions and beliefs (what the user thinks, feels, or believes about topics)
    - Preferences and choices (what the user likes, dislikes, or prefers)
    - Experiences and anecdotes (what the user has done or experienced)
    - Goals and intentions (what the user wants to achieve or plans to do)
4. Write each proposition on a new line.
5. Include both objective facts and subjective opinions: capture the full range of user-disclosed information.
6. Do not duplicate propositions that are already in the existing propositions list.
7. KEEP TOTAL OUTPUT UNDER 60000 TOKENS or 240,000 CHARACTERS.

AFTER DRAFTING:
- CHECK: Ensure the output is under 60000 tokens or 240,000 characters and contains no duplicate propositions.
- Remove duplicates.
- IF it is too long, rewrite until it passes.

Conversation segment:
---
{{ conversation_segment }}
---

Existing propositions:
---
{{ existing_propositions }}
---

Return ONLY the **new propositions**, one proposition per line.
If no new propositions are present, return "NO NEW PROPOSITIONS".""")

DEFAULT_CONDENSE_PROMPT = RichPromptTemplate("""
You are a comprehensive proposition condensing system designed to identify key information from conversations.

INSTRUCTIONS:
1. Review the current list of existing propositions
2. Condense the propositions into a more concise list, less than {{ max_propositions }} propositions
3. Focus on all types of information including:
    - Factual information (preferences, personal details, requirements, constraints, context)
    - Opinions and beliefs (what the user thinks, feels, or believes about topics)
    - Preferences and choices (what the user likes, dislikes, or prefers)
    - Experiences and anecdotes (what the user has done or experienced)
    - Goals and intentions (what the user wants to achieve or plans to do)
4. Write each proposition on a new line.
5. Include both objective facts and subjective opinions: preserve the full range of user information.
6. Do not duplicate propositions that are already in the existing propositions list.
7. KEEP TOTAL OUTPUT UNDER 60000 TOKENS or 240,000 CHARACTERS.

AFTER DRAFTING:
- CHECK: Ensure the output is under 60000 tokens or 240,000 characters and contains no duplicate propositions.
- Remove duplicates.
- IF it is too long, rewrite until it passes.

Existing propositions:
---
{{ existing_propositions }}
---

Return ONLY the **condensed propositions**, one per line.
If no propositions remain after condensing, return "NO CONDENSED PROPOSITIONS".""")

class PropositionExtractionMemoryBlock(BaseMemBlock):
    """
    A memory block that extracts key propositions from conversation history using an LLM.

    This block identifies and stores discrete propositions disclosed during the conversation,
    including facts, opinions, preferences, beliefs, experiences, and goals,
    structuring them in XML format for easy parsing and retrieval.
    """

    _propositions: List[str] = Field(
        default_factory = list,
        description = "List of extracted propositions from the conversation."
    )
    max_propositions: int = Field(
        default = 50, description = "The maximum number of propositions to store."
    )
    _proposition_extraction_prompt_template: BasePromptTemplate = Field(
        default = DEFAULT_EXTRACT_PROMPT,
        description = "Template for the proposition extraction prompt."
    )
    _proposition_condense_prompt_template: BasePromptTemplate = Field(
        default = DEFAULT_CONDENSE_PROMPT,
        description = "Template for the proposition condense prompt."
    )

    @field_validator("proposition_extraction_prompt_template", mode = "before")
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

    async def _aget(self, messages: Optional[List[ChatMessage]] = None, **block_kwargs: Any) -> str:
        """Return the current propositions as formatted text."""
        if not self._propositions:
            return ""

        return "\n".join([f"{proposition}" for proposition in self._propositions])

    async def _aput(self, messages: List[ChatMessage]) -> None:
        """Extract propositions from new messages and add them to the propositions list."""

        # Skip if no messages
        if not messages:
            return

        # Track before call
        start_time = time.time()

        # Format existing propositions for the prompt
        existing_propositions_text = ""
        if self._propositions:
            existing_propositions_text = "\n".join([f"{proposition}" for proposition in self._propositions])

        conversation_text = "\n".join([m.content for m in messages])

        # Create the prompt
        prompt_messages = self._proposition_extraction_prompt_template.format(max_propositions = self.max_propositions, conversation_segment = conversation_text, existing_propositions = existing_propositions_text)

        # Get the propositions extraction
        response = await self.llm.acomplete(prompt_messages)
        propositions_text = "" if response.text == "NO NEW PROPOSITIONS" else response.text
        new_propositions = [line.strip() for line in propositions_text.strip().splitlines() if line.strip()]

        # Count input tokens
        self.input_tokens += int(len(prompt_messages) / 4)
        self.output_tokens += int(len(propositions_text) / 4)

        # Add new propositions to the list, avoiding exact-match duplicates
        for proposition in new_propositions:
            if proposition not in self._propositions:
                self._propositions.append(proposition)

        # Condense the propositions if they exceed the max_propositions
        if len(self._propositions) >= self.max_propositions:
            existing_propositions_text = "\n".join([f"{proposition}" for proposition in self._propositions])

            prompt_messages = self._proposition_condense_prompt_template.format(
                max_propositions = self.max_propositions,
                existing_propositions = existing_propositions_text
            )
            response = await self.llm.acomplete(prompt_messages)
            propositions_text = "" if response.text == "NO CONDENSED PROPOSITIONS" else response.text
            new_propositions = [line.strip() for line in propositions_text.strip().splitlines() if line.strip()]

            # count tokens for condense output
            self.input_tokens += int(len(existing_propositions_text) / 4)
            self.output_tokens += int(len(propositions_text) / 4)

            # Replace the propositions with the condensed list
            if new_propositions:
                self._propositions = new_propositions

        self.load_chat_history_time = time.time() - start_time