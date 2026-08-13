#############################################################################
# File: test_architecture_packers.py
#
# Description:
#   Checks the token packing and query wording shared by the
#   extraction-based memory architectures. It also verifies that model
#   output cannot invent source provenance.
#
#   - Confirms episodic and proposition extraction packers preserve
#     every entry once and in order.
#   - Confirms graph extraction packing preserves every entry.
#   - Checks episodic temporal wording and proposition historical
#     wording.
#   - Checks episodic, proposition, and graph extraction source IDs and
#     ordinals.
#   - Runs without Qdrant or external API calls.
#############################################################################

from __future__ import annotations

import pytest

from asdrp.memory.episodic_memory import EpisodicMemoryBlock
from asdrp.memory.graph_extraction import GraphExtractor
from asdrp.memory.proposition_extraction_memory import PropositionMemoryBlock

pytestmark = pytest.mark.unit  # Every test in this file is local and API-free.


@pytest.mark.parametrize(
    "cls,method,limit_kw",
    [
        (EpisodicMemoryBlock, "_pack_entries", "extraction_input_tokens"),
        (PropositionMemoryBlock, "_pack_entries", "extraction_input_tokens"),
    ],
)
def test_llm_extraction_packers_split_large_entry_sequences(
    cls, method, limit_kw, fake_runtime, memory_entry_factory
):
    # Use the same check for both memory classes that pack extraction prompts.
    kwargs = {
        "task_id": "t",
        "runtime": fake_runtime,
        "qdrant_url": "http://127.0.0.1:6333",
    }
    kwargs[limit_kw] = 500  # Force several packs without splitting entries.
    block = cls(**kwargs)  # Memory class selected by the parameter row.
    entries = [memory_entry_factory(i, "token " * 180) for i in range(6)]
    packs = getattr(block, method)(entries)  # Run its private packing helper.
    assert len(packs) >= 3
    assert [e.entry_id for pack in packs for e in pack] == [e.entry_id for e in entries]
    assert all(pack for pack in packs)


def test_graph_extractor_pack_preserves_all_entries(fake_runtime, memory_entry_factory):
    extractor = GraphExtractor(  # Uses the same small prompt limit.
        runtime=fake_runtime, input_token_limit=520, concurrency=2
    )
    entries = [memory_entry_factory(i, "token " * 170) for i in range(7)]
    packs = extractor._pack(entries)  # No entry should be lost or repeated.
    assert len(packs) >= 3
    assert [e.entry_id for p in packs for e in p] == [e.entry_id for e in entries]


def test_episodic_temporal_query_detection():
    # Time wording should expand neighbors, while a direct fact should not.
    assert EpisodicMemoryBlock._is_temporal_query(
        "What did I prefer before the update?"
    )
    assert not EpisodicMemoryBlock._is_temporal_query("What color do I like?")


def test_proposition_historical_query_detection():
    # Past-state wording should keep older fact versions available.
    assert PropositionMemoryBlock._is_historical_query("What was the previous address?")
    assert not PropositionMemoryBlock._is_historical_query("What is the address?")


@pytest.mark.asyncio
async def test_llm_extraction_rejects_unknown_source_provenance(
    fake_runtime, memory_entry_factory
):
    # A model cannot move extracted content outside the source pack by inventing an ID.
    entry = memory_entry_factory(3, "Alice moved to Rome.")

    fake_runtime.responses.append(
        [
            {
                "event": "Alice moved to Rome.",
                "participants": ["Alice"],
                "source_id": "not-in-this-pack",
                "source_ordinal": "not-an-integer",
            }
        ]
    )
    episodic = EpisodicMemoryBlock(
        task_id="e", runtime=fake_runtime, qdrant_url="http://127.0.0.1:6333"
    )
    episodes = await episodic._extract_pack([entry])
    assert episodes[0].source_id == entry.entry_id
    assert episodes[0].source_ordinal == entry.ordinal

    fake_runtime.responses.append(
        [
            {
                "statement": "Alice moved to Rome.",
                "subject": "Alice",
                "relation": "moved_to",
                "object": "Rome",
                "source_id": "not-in-this-pack",
                "source_ordinal": "not-an-integer",
            }
        ]
    )
    proposition = PropositionMemoryBlock(
        task_id="p", runtime=fake_runtime, qdrant_url="http://127.0.0.1:6333"
    )
    facts = await proposition._extract_pack([entry])
    assert facts[0].source_id == entry.entry_id
    assert facts[0].source_ordinal == entry.ordinal

    fake_runtime.responses.append(
        {
            "entities": [{"name": "Alice"}, {"name": "Rome"}],
            "relations": [
                {
                    "source": "Alice",
                    "relation": "moved_to",
                    "target": "Rome",
                    "statement": "Alice moved to Rome.",
                    "source_id": "not-in-this-pack",
                    "source_ordinal": "not-an-integer",
                }
            ],
        }
    )
    extractor = GraphExtractor(runtime=fake_runtime)
    _, relations = await extractor._extract_pack([entry])
    assert relations[0].source_id == entry.entry_id
    assert relations[0].source_ordinal == entry.ordinal
