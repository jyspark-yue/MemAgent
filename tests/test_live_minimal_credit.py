#############################################################################
# File: test_live_minimal_credit.py
#
# Description:
#   Provides one opt-in live embedding smoke test. It stays skipped by
#   default and uses only two short strings when explicitly enabled.
#
#   - Loads the local environment file when present.
#   - Requires RUN_LIVE_API_TESTS=1 before spending API credit.
#   - Checks result count, vector width, and model batch count.
#   - Closes the embedding client after success or failure.
#############################################################################

from __future__ import annotations

import os

import pytest
from dotenv import load_dotenv

load_dotenv()  # Allow an API key from the project's local environment file.

pytestmark = pytest.mark.live_api  # This file is skipped unless enabled by hand.


@pytest.mark.asyncio
async def test_single_embedding_request_shape_only():
    # Keep live credit use opt-in and limited to two tiny strings.
    if os.environ.get("RUN_LIVE_API_TESTS") != "1":
        pytest.skip("Set RUN_LIVE_API_TESTS=1 to spend a tiny amount of API credit")
    from asdrp.openai_embedder import OpenAIEmbedder

    embedder = OpenAIEmbedder(  # Smallest useful live embedding setup.
        batch_size=4,
        batch_token_limit=100,
        max_parallel_batches=1,
        tokens_per_minute=10000,
    )
    try:
        result = await embedder.embed_documents(["alpha", "beta"])  # One batch.
        assert len(result.vectors) == 2
        assert all(len(vector) == 1536 for vector in result.vectors)
        assert result.model_batches == 1
    finally:
        await embedder.close()
