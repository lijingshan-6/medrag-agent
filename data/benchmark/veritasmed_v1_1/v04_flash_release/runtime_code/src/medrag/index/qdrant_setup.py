"""Qdrant collection creation helpers."""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)


def create_collection(
    client: QdrantClient,
    name: str = "medrag_text",
    recreate: bool = False,
) -> None:
    if client.collection_exists(name):
        if not recreate:
            print(f"[qdrant] collection '{name}' already exists, skipping creation")
            return
        client.delete_collection(name)
    client.create_collection(
        collection_name=name,
        vectors_config={
            "dense": VectorParams(size=1024, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(
                index=SparseIndexParams(on_disk=False),
            ),
        },
    )
    print(f"[qdrant] created collection '{name}' (dense + sparse)")


__all__ = ["create_collection"]
