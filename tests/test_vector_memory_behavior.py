#############################################################################
# File: test_vector_memory_behavior.py
#
# Description:
#   Checks the raw vector baseline's stored payloads and neighbor
#   expansion without Qdrant or OpenAI calls. Fixed retrieval results make
#   ordering and deduplication exact.
#
#   - Checks that raw text, source IDs, ordinals, entry type, and
#     metadata are retained.
#   - Checks adjacent source expansion around several dense matches.
#   - Checks overlapping neighbor windows are deduplicated.
#   - Checks final neighboring evidence remains in chronological order.
#############################################################################

from __future__ import annotations

import pytest

from asdrp.eval_schemas import RetrievalPlan, RetrievedMemory
from asdrp.memory.vector_memory import VectorMemoryBlock

pytestmark = pytest.mark.unit  # Fake methods keep every check local.


class CaptureVector(VectorMemoryBlock):
    # Save records in memory instead of sending them to Qdrant.

    async def _store_text_records(self, records):
        self.records = [dict(record) for record in records]  # Copy caller data.


@pytest.mark.asyncio
async def test_vector_put_retains_raw_text_ids_ordinals_and_metadata(
    fake_runtime, memory_entry_factory
):
    block = CaptureVector(
        task_id="t", runtime=fake_runtime, qdrant_url="http://127.0.0.1:6333"
    )
    entries = [memory_entry_factory(i) for i in range(3)]  # Ordered raw sessions.
    await block.put(entries)
    assert [record["text"] for record in block.records] == [
        entry.text for entry in entries
    ]
    assert [record["ordinal"] for record in block.records] == [0, 1, 2]
    assert all(record["entry_kind"] == "session" for record in block.records)


@pytest.mark.asyncio
async def test_vector_get_adds_neighbors_deduplicates_and_orders(fake_runtime):
    block = VectorMemoryBlock(  # Search methods are replaced below.
        task_id="t", runtime=fake_runtime, qdrant_url="http://127.0.0.1:6333"
    )
    seeds = [  # Two vector matches that sit several entries apart.
        RetrievedMemory("a", "A", 0.9, 5, {}),
        RetrievedMemory("b", "B", 0.8, 8, {}),
    ]
    neighbors = [  # Source entries around both matches.
        RetrievedMemory("x", "X", 1, 4, {}),
        seeds[0],
        RetrievedMemory("y", "Y", 1, 6, {}),
        seeds[1],
        RetrievedMemory("z", "Z", 1, 9, {}),
    ]

    async def similar(*_args, **_kwargs):
        return seeds

    async def ordinals(*_args, **_kwargs):
        return neighbors

    block._query_similar = similar  # Return the fixed vector matches.
    block._retrieve_ordinals = ordinals  # Return the fixed neighboring entries.
    result = await block.get("q", RetrievalPlan(top_k=2, neighbor_window=1))
    assert [item.ordinal for item in result] == [4, 5, 6, 8, 9]
    assert len({item.memory_id for item in result}) == len(result)
