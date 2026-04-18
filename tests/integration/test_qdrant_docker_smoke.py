import uuid

import pytest
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException
from qdrant_client.models import Distance, PointStruct, VectorParams


@pytest.mark.integration
def test_qdrant_docker_roundtrip():
    """Smoke test against a live Qdrant service (typically Docker on localhost:6333)."""
    client = QdrantClient(host="localhost", port=6333, timeout=3.0)

    try:
        client.get_collections()
    except Exception as exc:
        pytest.skip(
            "Qdrant is not reachable on localhost:6333. "
            "Start it with: docker compose up -d qdrant"
        )

    collection = f"smoke_{uuid.uuid4().hex[:8]}"
    try:
        client.create_collection(
            collection_name=collection,
            vectors_config=VectorParams(size=4, distance=Distance.COSINE),
        )

        client.upsert(
            collection_name=collection,
            points=[
                PointStruct(id=1, vector=[0.1, 0.2, 0.3, 0.4], payload={"text": "kyoto"}),
                PointStruct(id=2, vector=[0.9, 0.9, 0.9, 0.9], payload={"text": "paris"}),
            ],
        )

        hits = client.search(
            collection_name=collection,
            query_vector=[0.1, 0.2, 0.31, 0.39],
            limit=1,
        )

        assert hits, "Expected at least one Qdrant hit."
        assert hits[0].payload is not None
    finally:
        try:
            client.delete_collection(collection_name=collection)
        except (ResponseHandlingException, Exception):
            pass
