#############################################################################
# File: test_graph_memories.py
#
# Description:
#   Checks graph memory behavior with fixed local entities and relations.
#   No extraction model or external service is used.
#
#   - Checks that updated edges retain history and mark only the newest
#     value active.
#   - Checks that current questions hide superseded edges and historical
#     questions retain them.
#   - Checks that graph traversal does not exceed the requested hop count.
#############################################################################


from __future__ import annotations

import pytest

from asdrp.eval_schemas import RetrievalPlan
from asdrp.memory.extraction import EntityRecord, RelationRecord
from asdrp.memory.graph_memory import GraphMemoryBlock

pytestmark = pytest.mark.unit  # Fake extraction and vectors keep these tests local.

# Two versions of one fact plus a second graph hop.
RELATIONS = [
    RelationRecord(
        source="Alice",
        relation="lives_in",
        target="Paris",
        statement="Alice lives in Paris.",
        temporal="unnecessary",
        source_ordinal=0,
        source_id="e0",
    ),
    RelationRecord(
        source="Alice",
        relation="lives_in",
        target="Rome",
        statement="Alice lives in Rome.",
        temporal="new",
        source_ordinal=1,
        source_id="e1",
    ),
    RelationRecord(
        source="Rome",
        relation="located_in",
        target="Italy",
        statement="Rome is in Italy.",
        temporal="",
        source_ordinal=2,
        source_id="e2",
    ),
]
ENTITIES = [EntityRecord(name=name) for name in ("Alice", "Paris", "Rome", "Italy")]


@pytest.mark.asyncio
async def test_graph_marks_superseded_edge_and_filters_latest(
    fake_runtime, memory_entry_factory
):
    block = GraphMemoryBlock(task_id="g", runtime=fake_runtime)

    async def extract(entries):
        # Return the fixed graph instead of calling an extraction model.
        del entries
        return ENTITIES, RELATIONS

    block._extractor.extract = extract
    await block.initialize()
    await block.put([memory_entry_factory(0)])

    edges = list(block._graph.edges(keys=True, data=True))
    assert sum(bool(data["active"]) for *_, data in edges) == 2
    assert sum(not bool(data["active"]) for *_, data in edges) == 1

    current = await block.get(
        "Where does Alice live?",
        RetrievalPlan(top_k=5, graph_hops=2, prefer_latest=True),
    )
    assert "Alice lives in Rome." in [item.text for item in current]
    assert "Alice lives in Paris." not in [item.text for item in current]

    history = await block.get(
        "Where did Alice live before?",
        RetrievalPlan(top_k=5, graph_hops=2, prefer_latest=True),
    )
    assert "Alice lives in Paris." in [item.text for item in history]


@pytest.mark.asyncio
async def test_graph_does_not_walk_past_requested_hops(
    fake_runtime, memory_entry_factory
):
    # A two-hop walk from A may reach A-B and B-C, but not the third edge C-D.
    relations = [
        RelationRecord("A", "links_to", "B", "A links to B.", source_ordinal=0),
        RelationRecord("B", "links_to", "C", "B links to C.", source_ordinal=1),
        RelationRecord("C", "links_to", "D", "C links to D.", source_ordinal=2),
    ]
    entities = [EntityRecord(name=name) for name in ("A", "B", "C", "D")]
    block = GraphMemoryBlock(task_id="hops", runtime=fake_runtime)

    async def extract(entries):
        del entries
        return entities, relations

    block._extractor.extract = extract
    await block.initialize()
    await block.put([memory_entry_factory(0)])
    block._rank_seed_nodes = lambda query, query_vector: [("a", 1.0)]

    result = await block.get("start at A", RetrievalPlan(top_k=10, graph_hops=2))
    statements = {item.text for item in result}

    assert "A links to B." in statements
    assert "B links to C." in statements
    assert "C links to D." not in statements
