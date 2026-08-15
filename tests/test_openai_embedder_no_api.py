#############################################################################
# File: test_openai_embedder_no_api.py
#
# Description:
#   Checks embedding configuration and batch packing without creating a
#   live request. These tests isolate the deterministic local parts of the
#   shared embedding service.
#
#   - Builds a minimal test object without opening an OpenAI client.
#   - Checks batch count and aggregate token limits while preserving
#     input order.
#   - Checks invalid batch, concurrency, pacing, timeout, and retry
#     settings.
#############################################################################
from __future__ import annotations

import pytest

from asdrp.openai_embedder import OpenAIEmbedder

pytestmark = pytest.mark.unit  # These checks do not call OpenAI.


def make_embedder(**kwargs):
    # Skip the network client setup because this test only needs batch settings.
    embedder = OpenAIEmbedder.__new__(OpenAIEmbedder)
    embedder.batch_size = kwargs.get("batch_size", 3)  # String-count limit.
    embedder.batch_token_limit = kwargs.get("batch_token_limit", 10)  # Token limit.
    return embedder


def test_batch_packing_obeys_count_and_token_limits():
    embedder = make_embedder(batch_size=3, batch_token_limit=10)  # Test-only object.
    texts = ["a", "b", "c", "d", "e"]  # Five inputs in source order.
    counts = [4, 4, 4, 2, 8]  # Matching token counts.
    batches = embedder._pack_batches(texts, counts)  # Packed in original order.
    assert batches == [["a", "b"], ["c", "d"], ["e"]]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"batch_size": 0},
        {"batch_token_limit": 0},
        {"max_parallel_batches": 0},
        {"tokens_per_minute": 0},
        {"rate_limit_buffer_seconds": -1},
        {"request_timeout": 0},
        {"retry_attempts": 0},
        {"retry_base_delay": 2, "retry_max_delay": 1},
    ],
)
def test_constructor_rejects_invalid_limits(kwargs):
    # Each parameter row makes one required limit invalid.
    with pytest.raises(ValueError):
        OpenAIEmbedder(**kwargs)
