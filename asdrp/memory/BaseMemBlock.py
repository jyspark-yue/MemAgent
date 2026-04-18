#############################################################################
# File: BaseMemBlock.py
#
# Description:
#   Wrapper class that extends BaseMemoryBlock
#
# Authors:
#   @author         Judy Yu
#                   - Created BaseMemBlock.py
#
# Contributors:
#   @contributor    Eric Vincent Fernandes
#                   - Modified code to be compatible with Gemini (GenAI)
#
# Date:
#   Created:    October 14, 2025  (Judy Yu)
#   Modified:   October 16, 2025 (Eric Vincent Fernandes)
#############################################################################

from google.genai import types
from llama_index.core.callbacks import TokenCountingHandler, CallbackManager
from llama_index.core.llms import LLM
from llama_index.core.memory import BaseMemoryBlock
from llama_index.llms.google_genai import GoogleGenAI
from pydantic import Field

_SAFETY_SETTINGS = [
    types.SafetySetting(
        category = types.HarmCategory.HARM_CATEGORY_HARASSMENT,
        # threshold = types.HarmBlockThreshold.OFF,
        threshold = types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category = types.HarmCategory.HARM_CATEGORY_HATE_SPEECH,
        # threshold = types.HarmBlockThreshold.OFF,
        threshold = types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category = types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
        # threshold = types.HarmBlockThreshold.OFF,
        threshold = types.HarmBlockThreshold.BLOCK_NONE,
    ),
    types.SafetySetting(
        category = types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
        # threshold = types.HarmBlockThreshold.OFF,
        threshold = types.HarmBlockThreshold.BLOCK_NONE,
    )
]

_GEN_CONFIG = types.GenerateContentConfig(
    safety_settings = _SAFETY_SETTINGS,
    temperature = 0.0,
    thinking_config = types.ThinkingConfig(thinking_budget = -1))

def _get_default_llm(callback_manager=CallbackManager(handlers = [TokenCountingHandler()])) -> LLM:
    return GoogleGenAI(
        model = "gemini-2.5-flash-lite",
        temperature = 0.0,
        max_retries = 5,
        callback_manager = callback_manager,
        generation_config = _GEN_CONFIG,
    )

class BaseMemBlock(BaseMemoryBlock[str]):
    input_tokens: int = Field(
        default = 0, description = "The number of tokens passed into the LLM when loading the chat history."
    )
    output_tokens: int = Field(
        default = 0, description = "The number of tokens returned by the LLM when loading the chat history."
    )
    load_chat_history_time: float = Field(
        default = 0.0, description = "The duration of time the memory block took to load the chat history."
    )
    llm: LLM = Field(
        default_factory = _get_default_llm
    )