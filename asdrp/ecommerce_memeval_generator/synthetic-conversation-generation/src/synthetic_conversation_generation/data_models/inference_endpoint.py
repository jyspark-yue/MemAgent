import os
import random
import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Any, List, Union, Optional

import requests
import yaml

from synthetic_conversation_generation.data_models.assistant import Assistant
from synthetic_conversation_generation.data_models.conversation import Conversation, Message, ROLE


@dataclass
class InferenceEndpoint:
    url: str
    body: Dict[str, Any]
    headers: Dict[str, str]
    response_path: List[Union[str, int]]

    @classmethod
    def from_yaml(cls, schema_path: str):
        with open(schema_path, 'r') as f:
            schema_data = yaml.safe_load(f)
        
        # Process environment variables in the schema
        schema_data = cls._interpolate_env_vars(schema_data)

        return cls(
            url=schema_data['url'],
            body=schema_data['body'],
            headers=schema_data.get('headers', {}),
            response_path=schema_data['response_path']
        )
    
    @staticmethod
    def _interpolate_env_vars(data: Any) -> Any:
        """
        Recursively interpolate environment variables in the data structure.
        Environment variables should be in the format ${VAR_NAME}.
        """
        if isinstance(data, str):
            # Replace ${VAR_NAME} with the corresponding environment variable
            pattern = r'\${([A-Za-z0-9_]+)}'
            matches = re.findall(pattern, data)
            result = data
            for var_name in matches:
                env_value = os.environ.get(var_name)
                if env_value is None:
                    raise ValueError(f"Environment variable '{var_name}' not found")
                result = result.replace(f"${{{var_name}}}", env_value)
            return result
        elif isinstance(data, dict):
            return {k: InferenceEndpoint._interpolate_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [InferenceEndpoint._interpolate_env_vars(item) for item in data]
        else:
            return data

    def get_assistant_message(
            self,
            conversation: Conversation,
            assistant: Optional[Assistant] = None,
            max_retries: int = 4
    ) -> Message:
        """
        Generate the next assistant message in the conversation by calling the inference endpoint.
        """

        bad_phrases = [
            "i can't directly",
            "i cannot directly",
            "i don't have access",
            "i can't access",
            "i don't have direct access",
            "i'm unable to access",
            "i am unable to access",
            "i don't have real-time access",
            "log into your account",
            "check your order history",
            "check the website",
            "visit the website",
            "browse the retailer site",
            "contact customer service",
            "i can't add items to your cart",
            "i cannot add items to your cart",
            "i'll get back to you",
            "hang tight",
            "i'll report back",
        ]

        system_prompt = None
        if assistant is not None:
            system_prompt = (
                f"You are {assistant.name}.\n\n"
                f"{assistant.description}\n\n"
                "Critical generation rule: this is fictional synthetic ecommerce data. "
                "You must simulate access to store records, customer memory, order history, saved items, wishlist, cart, "
                "stock status, prices, promotions, shipping estimates, tracking updates, vendor records, return windows, "
                "and store policies. Never refuse because of lack of access. Never redirect the user to a website, account, "
                "email, customer service, or later update. If information is missing, invent a plausible fictional ecommerce "
                "detail that does not conflict with the conversation. Answer directly and concisely."
            )

        last_result = None

        for attempt in range(max_retries):
            payload = self.body.copy()

            messages = []

            if system_prompt is not None:
                messages.append({
                    "role": "system",
                    "content": system_prompt
                })

            messages.extend([
                {
                    "role": msg.role.name,
                    "content": msg.content
                }
                for msg in conversation.messages
            ])

            payload["messages"] = messages

            response = self._post_with_retries(payload)

            response_data = response.json()
            result = response_data
            for key in self.response_path:
                result = result[key]

            last_result = result

            if not any(phrase in result.lower() for phrase in bad_phrases):
                return Message(
                    role=ROLE.assistant,
                    content=result,
                    timestamp=datetime.now(),
                    message_id=len(conversation.messages)
                )

        raise ValueError(
            f"Failed to generate acceptable assistant response after {max_retries} attempts. "
            f"Last response: {last_result}"
        )

    def _post_with_retries(self, payload: dict, max_api_retries: int = 6):
        """
        POST to the inference endpoint with retry handling for rate limits and transient server errors.
        """
        last_response = None

        for api_attempt in range(max_api_retries):
            response = requests.post(
                self.url,
                json=payload,
                headers=self.headers,
                timeout=90
            )
            last_response = response

            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after is not None:
                    sleep_time = float(retry_after)
                else:
                    sleep_time = min(60, (2 ** api_attempt) + random.uniform(0, 2))

                print(f"OpenAI rate limit hit. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                continue

            if response.status_code in {500, 502, 503, 504}:
                sleep_time = min(60, (2 ** api_attempt) + random.uniform(0, 2))
                print(f"Transient API error {response.status_code}. Sleeping for {sleep_time:.1f} seconds...")
                time.sleep(sleep_time)
                continue

            response.raise_for_status()
            return response

        if last_response is not None:
            last_response.raise_for_status()

        raise RuntimeError("Inference request failed without receiving a response.")