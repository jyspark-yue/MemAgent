#############################################################################
# File: models.py
#
# Description:
#   Defines the typed schemas used throughout EcommerceMemEval generation.
#
#   - Enumerates supported ecommerce memory question types.
#   - Models raw messages/conversations, generated QA/evidence, and final dataset rows.
#   - Exposes the generation-result JSON schema sent to the LLM.
#############################################################################

from __future__ import annotations

from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv())

from enum import StrEnum
from typing import Any
from pydantic import BaseModel, Field, ConfigDict, field_validator, model_validator


class QuestionType(StrEnum):
    preference_recall = "preference_recall"
    constraint_recall = "constraint_recall"
    purchase_history_recall = "purchase_history_recall"
    cart_or_wishlist_recall = "cart_or_wishlist_recall"
    return_support_recall = "return_support_recall"
    temporal_update = "temporal_update"
    conflict_resolution = "conflict_resolution"
    multi_session_synthesis = "multi_session_synthesis"
    recommendation_from_memory = "recommendation_from_memory"
    abstention = "abstention"


class Message(BaseModel):
    role: str
    content: str

    @field_validator("role")
    @classmethod
    def role_must_be_valid(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"user", "assistant", "system"}:
            raise ValueError(f"Invalid role: {value}")
        return value

    @field_validator("content")
    @classmethod
    def content_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Message content is empty")
        return value


class RawConversation(BaseModel):
    messages: list[Message]

    @model_validator(mode="after")
    def must_have_messages(self) -> "RawConversation":
        if not self.messages:
            raise ValueError("Conversation has no messages")
        return self


class EvidenceItem(BaseModel):
    qa_pair_id: str = ""
    session_ids: list[str] = Field(default_factory=list)
    facts: list[str] = Field(default_factory=list)

    @field_validator("facts")
    @classmethod
    def facts_must_not_be_empty(cls, value: list[str]) -> list[str]:
        cleaned = [v.strip() for v in value if v and v.strip()]
        if not cleaned:
            raise ValueError("Evidence facts cannot be empty")
        return cleaned

    @field_validator("session_ids")
    @classmethod
    def session_ids_must_be_clean(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for v in value:
            if not v or not v.strip():
                continue
            item = v.strip()
            if item not in seen:
                cleaned.append(item)
                seen.add(item)

        return cleaned


class GeneratedQA(BaseModel):
    question: str
    answers: list[str]
    question_type: QuestionType
    evidence: EvidenceItem

    @field_validator("question")
    @classmethod
    def question_must_not_be_empty(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question is empty")
        return value

    @field_validator("answers")
    @classmethod
    def answers_must_not_be_empty(cls, value: list[str]) -> list[str]:
        cleaned: list[str] = []
        seen: set[str] = set()

        for v in value:
            if not v or not v.strip():
                continue
            item = v.strip()
            key = item.lower()
            if key not in seen:
                cleaned.append(item)
                seen.add(key)

        if not cleaned:
            raise ValueError("Answers cannot be empty")
        return cleaned


class GenerationResult(BaseModel):
    qas: list[GeneratedQA]


class DatasetMetadata(BaseModel):
    customer_id: str
    source: str
    qa_pair_ids: list[str]
    question_types: list[QuestionType]
    evidence: list[EvidenceItem]


class DatasetRow(BaseModel):
    model_config = ConfigDict(use_enum_values=True)

    context: str
    questions: list[str]
    answers: list[list[str]]
    metadata: DatasetMetadata

    @model_validator(mode="after")
    def aligned_lengths(self) -> "DatasetRow":
        n = len(self.questions)
        if n == 0:
            raise ValueError("Dataset row must have at least one question")
        if len(self.answers) != n:
            raise ValueError("answers length must match questions length")
        if len(self.metadata.qa_pair_ids) != n:
            raise ValueError("qa_pair_ids length must match questions length")
        if len(self.metadata.question_types) != n:
            raise ValueError("question_types length must match questions length")
        if len(self.metadata.evidence) != n:
            raise ValueError("evidence length must match questions length")
        for qa_id, ev in zip(self.metadata.qa_pair_ids, self.metadata.evidence):
            if ev.qa_pair_id != qa_id:
                raise ValueError(
                    f"Evidence qa_pair_id {ev.qa_pair_id} does not match {qa_id}"
                )
        return self


def model_json_schema_for_llm() -> dict[str, Any]:
    """Schema for the LLM output only, not the final dataset row."""
    return GenerationResult.model_json_schema()
