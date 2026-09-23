"""Multi-Query RRF retriever — Pipeline P5.

Core idea:
  A single phrasing of a medical question may not match the vocabulary used
  in the corpus.  Generate N alternative reformulations with an LLM, retrieve
  top-k hits for each, then fuse with RRF.  Covers more terminological
  variation (e.g. "heart attack" vs "myocardial infarction") and improves
  recall without sacrificing precision (RRF downweights duplicates).

Flow:
  query
    └─ LLM → [original, rewrite1, rewrite2, rewrite3]
                  └─ for each: BGE-M3 dense → Qdrant ANN → ranked list
                                    └─ RRF fusion → top-k merged hits
"""

from __future__ import annotations

import re
from collections import defaultdict

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient

from medrag.agent.utils import strip_thinking
from medrag.config import DEFAULT_OLLAMA_MODEL, ollama_base_url
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import RetrievedChunk

MQ_SYSTEM = (
    "You are a medical information retrieval expert. Given a medical question, "
    "generate 3 alternative phrasings that express the same information need "
    "but use different vocabulary, perspective, or level of technicality. "
    "Output ONLY a numbered list (1. ... 2. ... 3. ...). No explanations."
)

MQ_USER_TEMPLATE = "Medical question: {query}\n\nGenerate 3 alternative phrasings:"


def _rrf(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = defaultdict(float)
    for ranking in rankings:
        for rank, chunk_id in enumerate(ranking):
            scores[chunk_id] += 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda x: -x[1])


class MultiQueryRetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: BGEM3Embedder,
        collection: str = "medrag_text",
        llm_model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.5,
        candidate_k: int = 10,
        rrf_k: int = 60,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection
        self.candidate_k = candidate_k
        self.rrf_k = rrf_k
        self.llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url(),
            reasoning=False,
            temperature=temperature,
            num_ctx=512,
        )

    def _rewrite_query(self, query: str) -> list[str]:
        """Return [original_query, rewrite1, rewrite2, rewrite3]."""
        resp = self.llm.invoke([
            SystemMessage(content=MQ_SYSTEM),
            HumanMessage(content=MQ_USER_TEMPLATE.format(query=query)),
        ])
        raw = strip_thinking(resp.content).strip()

        # Parse numbered list: "1. ...\n2. ...\n3. ..."
        rewrites = re.findall(r"^\d+\.\s*(.+)$", raw, re.MULTILINE)
        rewrites = [r.strip() for r in rewrites if r.strip()][:3]

        # Always include the original query first
        return [query] + rewrites

    def _dense_retrieve(self, query: str) -> tuple[list[str], dict]:
        """Return (ranked chunk_ids, id_to_point mapping)."""
        enc = self.embedder.encode([query])
        vec: list[float] = enc["dense"][0].tolist()
        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=vec,
            using="dense",
            limit=self.candidate_k,
            with_payload=True,
        )
        ranking = []
        id_to_point = {}
        for p in result.points:
            cid = p.payload["chunk_id"]
            ranking.append(cid)
            id_to_point[cid] = p
        return ranking, id_to_point

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        queries = self._rewrite_query(query)
        self._last_queries = queries  # expose for debugging

        all_rankings: list[list[str]] = []
        id_to_point: dict[str, object] = {}

        for q in queries:
            ranking, points = self._dense_retrieve(q)
            all_rankings.append(ranking)
            id_to_point.update(points)  # later queries don't overwrite earlier (first-seen wins)

        fused = _rrf(all_rankings, k=self.rrf_k)

        results: list[RetrievedChunk] = []
        for chunk_id, rrf_score in fused[:k]:
            p = id_to_point.get(chunk_id)
            if p is None:
                continue
            results.append(RetrievedChunk(
                chunk_id=chunk_id,
                text=p.payload["text"],
                score=rrf_score,
                payload=p.payload,
            ))
        return results

    @property
    def last_queries(self) -> list[str]:
        """Expose the last generated query set for debugging/display."""
        return getattr(self, "_last_queries", [])


__all__ = ["MultiQueryRetriever"]
