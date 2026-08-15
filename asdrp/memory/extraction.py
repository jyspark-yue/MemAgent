#############################################################################
# File: extraction.py
#
# Description:
#   Provides the small records and text cleanup shared by proposition and
#   graph extraction. It also parses common explicit benchmark fact forms
#   without an LLM call.
#
#   - Defines proposition, entity, and directed relation records.
#   - Normalizes Unicode, case, spacing, and punctuation for stable
#     keys.
#   - Recognizes common MemoryAgentBench fact sentence patterns.
#   - Parses broad subject-verb-object statements when no specific
#     pattern matches.
#   - Recognizes common conversational preference, intent, ownership,
#     recommendation, and location relations without an LLM call.
#   - Keeps unknown sentences as standalone assertions instead of
#     discarding them.
#############################################################################

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class PropositionRecord:
    # One standalone fact and the source it came from.

    statement: str  # Complete fact text.
    subject: str  # Main entity described by the fact.
    relation: str  # Link between the subject and object.
    object: str  # Value or second entity.
    temporal: str  # Optional date, time, or order note.
    source_ordinal: int  # Position of the source entry.
    source_id: str  # ID of the source entry.
    metadata: dict[str, Any] = field(default_factory=dict)  # Extra saved fields.

    @property
    def fact_key(self) -> str:
        # Facts with this same key can be unnecessary and new versions of one value.
        return f"{normalize_key(self.subject)}|{normalize_key(self.relation)}"


@dataclass(slots=True)
class EntityRecord:
    # One named node in an extracted graph.

    name: str  # Name used in the source.
    entity_type: str = "unknown"  # Person, place, product, or another type.
    description: str = ""  # Short source-based description.


@dataclass(slots=True)
class RelationRecord:
    # One directed edge and the statement that supports it.

    source: str  # Entity where the edge starts.
    relation: str  # Edge label.
    target: str  # Entity or value where the edge ends.
    statement: str  # Full source-based statement.
    temporal: str = ""  # Optional date, time, or order note.
    source_ordinal: int = 0  # Position of the supporting entry.
    source_id: str = ""  # ID of the supporting entry.


# Known fact sentence forms and the relation name saved for each one.
_RELATION_PATTERNS: tuple[tuple[str, str], ...] = (
    # Check dataset-specific forms before the broad subject/relation/object rule.
    # This keeps entity names correct for graph paths and updated facts.
    (
        r"^The (?:univeristy|university) where (?P<s>.+?) was educated is (?P<o>.+?)\.?$",
        "educated_at",
    ),
    (
        r"^The headquarters of (?P<s>.+?) is located in the city of (?P<o>.+?)\.?$",
        "headquarters_in",
    ),
    (r"^The type of music that (?P<s>.+?) plays is (?P<o>.+?)\.?$", "music_genre"),
    (r"^The company that produced (?P<s>.+?) is (?P<o>.+?)\.?$", "produced_by"),
    (
        r"^The name of the current head of state in (?P<s>.+?) is (?P<o>.+?)\.?$",
        "head_of_state",
    ),
    (r"^(?P<s>.+?) worked in the city of (?P<o>.+?)\.?$", "worked_in"),
    (r"^(?P<s>.+?) works in the field of (?P<o>.+?)\.?$", "occupation"),
    (r"^(?P<s>.+?) was created in the country of (?P<o>.+?)\.?$", "created_in_country"),
    (r"^(?P<s>.+?) was created in the city of (?P<o>.+?)\.?$", "created_in_city"),
    (r"^(?P<s>.+?) was born in the city of (?P<o>.+?)\.?$", "born_in"),
    (r"^(?P<s>.+?) died in the city of (?P<o>.+?)\.?$", "died_in"),
    (r"^(?P<s>.+?) is a citizen of (?P<o>.+?)\.?$", "citizen_of"),
    (r"^(?P<s>.+?) is married to (?P<o>.+?)\.?$", "married_to"),
    (r"^(?P<s>.+?) is employed by (?P<o>.+?)\.?$", "employed_by"),
    (r"^(?P<s>.+?) was created by (?P<o>.+?)\.?$", "created_by"),
    (r"^(?P<s>.+?) was founded by (?P<o>.+?)\.?$", "founded_by"),
    (r"^(?P<s>.+?) was founded in the city of (?P<o>.+?)\.?$", "founded_in"),
    (r"^(?P<s>.+?) was performed by (?P<o>.+?)\.?$", "performed_by"),
    (r"^(?P<s>.+?) was developed by (?P<o>.+?)\.?$", "developed_by"),
    (
        r"^(?P<s>.+?) is located in the continent of (?P<o>.+?)\.?$",
        "located_in_continent",
    ),
    (r"^(?P<s>.+?) is located in the country of (?P<o>.+?)\.?$", "located_in_country"),
    (r"^(?P<s>.+?) speaks the language of (?P<o>.+?)\.?$", "speaks_language"),
    (r"^(?P<s>.+?) is affiliated with the religion of (?P<o>.+?)\.?$", "religion"),
    (r"^(?P<s>.+?) is associated with the sport of (?P<o>.+?)\.?$", "sport"),
    (r"^(?P<s>.+?) plays the position of (?P<o>.+?)\.?$", "position"),
    (r"^(?P<s>.+?)'s child is (?P<o>.+?)\.?$", "child"),
    (r"^(?P<s>.+?) is famous for (?P<o>.+?)\.?$", "famous_for"),
    (r"^The (?P<r>.+?) of (?P<s>.+?) is (?P<o>.+?)\.?$", "{r}"),
    # Common conversational forms also appear in LongMemEval-style memory.
    (r"^(?P<s>.+?) prefers (?P<o>.+?)\.?$", "prefers"),
    (r"^(?P<s>.+?) likes (?P<o>.+?)\.?$", "likes"),
    (r"^(?P<s>.+?) dislikes (?P<o>.+?)\.?$", "dislikes"),
    (r"^(?P<s>.+?) wants (?P<o>.+?)\.?$", "wants"),
    (r"^(?P<s>.+?) needs (?P<o>.+?)\.?$", "needs"),
    (r"^(?P<s>.+?) plans to (?P<o>.+?)\.?$", "plans_to"),
    (r"^(?P<s>.+?) intends to (?P<o>.+?)\.?$", "intends_to"),
    (r"^(?P<s>.+?) owns (?P<o>.+?)\.?$", "owns"),
    (r"^(?P<s>.+?) bought (?P<o>.+?)\.?$", "purchased"),
    (r"^(?P<s>.+?) purchased (?P<o>.+?)\.?$", "purchased"),
    (r"^(?P<s>.+?) uses (?P<o>.+?)\.?$", "uses"),
    (r"^(?P<s>.+?) lives in (?P<o>.+?)\.?$", "lives_in"),
    (r"^(?P<s>.+?) works at (?P<o>.+?)\.?$", "works_at"),
    (r"^(?P<s>.+?) recommended (?P<o>.+?)\.?$", "recommended"),
    (r"^(?P<s>.+?) chose (?P<o>.+?)\.?$", "chose"),
    (r"^(?P<s>.+?) selected (?P<o>.+?)\.?$", "selected"),
)


def normalize_key(value: str) -> str:
    # Normalize labels for entity identity and relation-key comparisons.

    value = unicodedata.normalize("NFKC", value).casefold()  # Unify text forms.
    value = re.sub(r"[^\w]+", " ", value, flags=re.UNICODE)  # Drop punctuation.
    return " ".join(value.split())


def parse_explicit_fact(
    statement: str, *, ordinal: int, source_id: str
) -> PropositionRecord:
    # Parse common FactConsolidation statements without spending an LLM call.
    #
    # Unknown sentence forms remain valid propositions. They use the full
    # statement as the subject and an ``asserts`` relation, so no information is
    # discarded just because the local parser does not know its sentence form.

    cleaned = re.sub(r"^\s*\d+\.\s*", "", statement.strip())  # Remove list number.
    for pattern, relation_template in _RELATION_PATTERNS:
        match = re.match(pattern, cleaned, flags=re.IGNORECASE)  # Try this form.
        if not match:
            continue
        groups = match.groupdict()  # Named subject, object, and optional relation.
        relation = relation_template  # Normalized relation for this sentence form.
        if relation_template == "{r}":
            relation = normalize_key(groups.get("r", "related_to")).replace(" ", "_")
        return PropositionRecord(
            statement=cleaned,
            subject=(groups.get("s") or cleaned).strip(),
            relation=relation,
            object=(groups.get("o") or "").strip().rstrip("."),
            temporal="",
            source_ordinal=ordinal,
            source_id=source_id,
        )

    # Broad fallback for statements of the form ``X is/was Y``.
    generic = re.match(
        r"^(?P<s>.+?)\s+(?P<r>is|was|are|were|has|had)\s+(?P<o>.+?)\.?$", cleaned
    )
    if generic:
        return PropositionRecord(
            statement=cleaned,
            subject=generic.group("s").strip(),
            relation=generic.group("r").strip(),
            object=generic.group("o").strip(),
            temporal="",
            source_ordinal=ordinal,
            source_id=source_id,
        )

    return PropositionRecord(
        statement=cleaned,
        subject=cleaned,
        relation="asserts",
        object="",
        temporal="",
        source_ordinal=ordinal,
        source_id=source_id,
    )
