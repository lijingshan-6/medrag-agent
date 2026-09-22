from __future__ import annotations

from medrag.retrieval.reranker import _select_coverage_chunks
from medrag.retrieval.retriever import RetrievedChunk


def _chunk(chunk_id: str, doc_id: str, score: float) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=chunk_id,
        score=score,
        payload={"source": "pubmed", "doc_id": doc_id},
    )


def test_select_coverage_chunks_does_not_force_a_different_source_for_shared_evidence() -> None:
    prostate = [
        _chunk("pubmed:1:0", "1", 0.99),
        _chunk("pubmed:2:0", "2", 0.80),
    ]
    breast = [
        _chunk("pubmed:1:0", "1", 0.98),
        _chunk("pubmed:3:0", "3", 0.79),
    ]

    selected = _select_coverage_chunks([prostate, breast], top_k=3)

    assert selected[0].chunk_id == "pubmed:1:0"
    assert selected[1].chunk_id == "pubmed:2:0"
    assert len(selected) == 3
    assert len({chunk.chunk_id for chunk in selected}) == 3


def test_select_coverage_chunks_fills_remaining_slots_by_score() -> None:
    first = [_chunk("pubmed:1:0", "1", 0.90), _chunk("pubmed:2:0", "2", 0.85)]
    second = [_chunk("pubmed:3:0", "3", 0.80), _chunk("pubmed:4:0", "4", 0.75)]

    selected = _select_coverage_chunks([first, second], top_k=3)

    assert [chunk.chunk_id for chunk in selected] == [
        "pubmed:1:0",
        "pubmed:3:0",
        "pubmed:2:0",
    ]
