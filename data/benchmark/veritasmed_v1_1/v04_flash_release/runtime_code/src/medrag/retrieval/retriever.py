"""Dense retrieval from Qdrant using BGE-M3 embeddings."""

from __future__ import annotations

from dataclasses import dataclass

from typing import TYPE_CHECKING

from qdrant_client import QdrantClient

if TYPE_CHECKING:
    from medrag.index.embedder import BGEM3Embedder


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    score: float
    payload: dict

    @property
    def citation(self) -> str:
        src = self.payload.get("source", "")
        doc = self.payload.get("doc_id", "?")
        if src == "pubmed":
            return f"PMID:{doc}"
        if src == "pmc":
            return f"PMC:{doc}"
        return doc


class DenseRetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: BGEM3Embedder,
        collection: str = "medrag_text",
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        vec = self.embedder.encode([query])["dense"][0].tolist()
        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            using="dense",
            limit=k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk_id=h.payload["chunk_id"],
                text=h.payload["text"],
                score=h.score,
                payload=h.payload,
            )
            for h in result.points
        ]


__all__ = ["RetrievedChunk", "DenseRetriever"]
