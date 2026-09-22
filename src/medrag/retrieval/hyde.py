"""HyDE (Hypothetical Document Embeddings) retriever — Pipeline P4.

Core idea (Gao et al., 2022):
  Instead of encoding the *question* directly, ask an LLM to write a short
  hypothetical passage that *would answer* the question, then encode that
  passage as the query vector.  Because the hypothetical passage lives in
  "answer space" rather than "question space", its embedding is closer to
  real answer documents in the corpus, improving recall for complex queries.

Flow:
  query (question)
      └─ LLM → hypothetical_passage (2-3 sentences from a medical paper)
                    └─ BGE-M3 dense encode → query vector
                                └─ Qdrant ANN search → top-k hits
"""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from qdrant_client import QdrantClient

from medrag.agent.utils import strip_thinking
from medrag.config import DEFAULT_OLLAMA_MODEL, ollama_base_url
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import RetrievedChunk

HYDE_SYSTEM = (
    "You are a medical research assistant. Your task is to write a short, "
    "factual passage (2-3 sentences) as if it were excerpted from a real "
    "PubMed abstract or clinical guideline that directly answers the given "
    "medical question. Focus on technical accuracy. Do NOT include citations, "
    "author names, or journal names. Do NOT answer the question conversationally "
    "— write in the style of a paper excerpt."
)

HYDE_USER_TEMPLATE = (
    "Medical question: {query}\n\n"
    "Write a short 2-3 sentence passage from a medical paper that answers this question:"
)


class HyDERetriever:
    def __init__(
        self,
        qdrant: QdrantClient,
        embedder: BGEM3Embedder,
        collection: str = "medrag_text",
        llm_model: str = DEFAULT_OLLAMA_MODEL,
        temperature: float = 0.3,
    ):
        self.qdrant = qdrant
        self.embedder = embedder
        self.collection = collection
        self.llm = ChatOllama(
            model=llm_model,
            base_url=ollama_base_url(),
            reasoning=False,
            temperature=temperature,
            num_ctx=1024,
        )

    def _generate_hypothesis(self, query: str) -> str:
        resp = self.llm.invoke([
            SystemMessage(content=HYDE_SYSTEM),
            HumanMessage(content=HYDE_USER_TEMPLATE.format(query=query)),
        ])
        hypothesis = strip_thinking(resp.content).strip()
        self._last_hypothesis = hypothesis
        return hypothesis

    def retrieve(self, query: str, k: int = 5) -> list[RetrievedChunk]:
        hypothesis = self._generate_hypothesis(query)

        # Encode the hypothetical passage (not the original question)
        enc = self.embedder.encode([hypothesis])
        dense_vec: list[float] = enc["dense"][0].tolist()

        result = self.qdrant.query_points(
            collection_name=self.collection,
            query=dense_vec,
            using="dense",
            limit=k,
            with_payload=True,
        )
        return [
            RetrievedChunk(
                chunk_id=p.payload["chunk_id"],
                text=p.payload["text"],
                score=p.score,
                payload=p.payload,
            )
            for p in result.points
        ]

    @property
    def last_hypothesis(self) -> str | None:
        """Expose the last generated hypothesis for debugging/display."""
        return getattr(self, "_last_hypothesis", None)


__all__ = ["HyDERetriever"]
