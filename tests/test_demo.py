import numpy as np
from qdrant_client import QdrantClient

from medrag.demo import bootstrap


class TinyEmbedder:
    def encode(self, texts, **kwargs):
        dense = np.zeros((len(texts), 1024), dtype=np.float32)
        dense[:, 0] = 1
        return {"dense": dense, "sparse": [{"1": 1.0} for _ in texts]}


def test_bootstrap_is_idempotent_and_does_not_touch_research_collection():
    client = QdrantClient(":memory:")
    try:
        from medrag.index.qdrant_setup import create_collection
        create_collection(client, "medrag_text")
        first = bootstrap(client, TinyEmbedder())
        second = bootstrap(client, TinyEmbedder())
        assert first == second == 3
        assert client.count("medrag_demo").count == 3
        assert client.count("medrag_text").count == 0
        rows, _ = client.scroll("medrag_demo", with_payload=True)
        assert all(row.payload["content_kind"] == "authored_demo_summary" for row in rows)

    finally:
        client.close()
