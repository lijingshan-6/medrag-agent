"""Upsert chunks with dense (+ optional sparse) vectors into Qdrant."""

from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, SparseVector

from medrag.ingest.chunker import Chunk


def _to_sparse_vector(weights: dict) -> SparseVector:
    """Convert BGE-M3 lexical_weights dict → Qdrant SparseVector.

    lexical_weights keys are string token IDs; values are float weights.
    Empty dicts get a single zero entry to satisfy Qdrant's non-empty requirement.
    """
    if not weights:
        return SparseVector(indices=[0], values=[0.0])
    return SparseVector(
        indices=[int(k) for k in weights],
        values=[float(v) for v in weights.values()],
    )


def index_chunks(
    client: QdrantClient,
    chunks: list[Chunk],
    dense_vecs: np.ndarray,
    sparse_weights: list[dict] | None = None,
    collection: str = "medrag_text",
    batch: int = 256,
    on_batch: callable | None = None,
) -> None:
    total = len(chunks)
    uploaded = 0
    points: list[PointStruct] = []
    for i, (c, vec) in enumerate(zip(chunks, dense_vecs)):
        vector_payload: dict = {"dense": vec.tolist()}
        if sparse_weights is not None:
            vector_payload["sparse"] = _to_sparse_vector(sparse_weights[i])

        points.append(PointStruct(
            id=str(uuid.uuid5(uuid.NAMESPACE_URL, c.chunk_id)),
            vector=vector_payload,
            payload={
                "chunk_id": c.chunk_id,
                "source": c.source,
                "doc_id": c.doc_id,
                "text": c.text,
                **c.metadata,
            },
        ))
        if len(points) >= batch:
            client.upsert(collection_name=collection, points=points)
            uploaded += len(points)
            if on_batch:
                on_batch(uploaded, total)
            points = []
    if points:
        client.upsert(collection_name=collection, points=points)
        uploaded += len(points)
        if on_batch:
            on_batch(uploaded, total)


__all__ = ["index_chunks"]
