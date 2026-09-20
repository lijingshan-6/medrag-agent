"""Transparent deterministic metrics for the VeritasMed benchmark."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import fmean
from typing import Any

from medrag.benchmark.schema import Answerability, BenchmarkQuestion, ClaimImportance


@dataclass(frozen=True)
class RetrievalScore:
    required_claim_recall: float
    supporting_chunk_recall: float
    all_required_found: bool
    ndcg: float
    reciprocal_rank: float
    hard_negative_hit: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AnswerScore:
    claim_completeness: float
    claim_support_precision: float
    citation_coverage: float
    answerability_score: float
    strict_pass: bool

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _entailing_chunks(question: BenchmarkQuestion) -> dict[str, set[str]]:
    return {
        claim.claim_id: {
            evidence.chunk_id
            for evidence in claim.evidence
            if evidence.support.value in {"direct", "derived"}
        }
        for claim in question.gold_claims
        if claim.importance is ClaimImportance.REQUIRED
    }


def _dcg(relevances: list[int]) -> float:
    return sum(
        relevance / math.log2(rank + 1)
        for rank, relevance in enumerate(relevances, start=1)
    )


def score_retrieval(
    question: BenchmarkQuestion,
    retrieved_chunk_ids: list[str],
    *,
    k: int = 5,
) -> RetrievalScore:
    """Score one ranking using claim coverage and exact frozen chunk IDs."""

    ranking = retrieved_chunk_ids[:k]
    ranking_set = set(ranking)
    required = _entailing_chunks(question)
    found_claims = sum(bool(chunks & ranking_set) for chunks in required.values())
    required_recall = found_claims / len(required) if required else 1.0

    supporting = set(question.supporting_chunk_ids)
    supporting_recall = (
        len(supporting & ranking_set) / len(supporting) if supporting else 1.0
    )
    relevant_chunks = set().union(*required.values()) if required else set()
    relevances = [int(chunk_id in relevant_chunks) for chunk_id in ranking]
    ideal_count = min(len(relevant_chunks), k)
    ideal_dcg = _dcg([1] * ideal_count)
    ndcg = _dcg(relevances) / ideal_dcg if ideal_dcg else 0.0
    first_relevant = next(
        (rank for rank, value in enumerate(relevances, start=1) if value),
        None,
    )
    reciprocal_rank = 1 / first_relevant if first_relevant else 0.0
    return RetrievalScore(
        required_claim_recall=required_recall,
        supporting_chunk_recall=supporting_recall,
        all_required_found=required_recall == 1.0,
        ndcg=ndcg,
        reciprocal_rank=reciprocal_rank,
        hard_negative_hit=bool(set(question.hard_negative_chunk_ids) & ranking_set),
    )


def score_answer(
    question: BenchmarkQuestion,
    *,
    claim_mappings: dict[str, list[str]],
    abstained: bool,
    boundary_acknowledged: bool,
    forbidden_claims_present: bool,
) -> AnswerScore:
    """Score curator-approved claim-to-citation mappings for one answer.

    This function deliberately does not infer semantic equivalence.  A local model
    may propose mappings, but a visible mapping artifact must be accepted or
    corrected before these deterministic metrics are final.
    """

    required = _entailing_chunks(question)
    if required:
        supported_claims = {
            claim_id
            for claim_id, allowed_chunks in required.items()
            if allowed_chunks & set(claim_mappings.get(claim_id, []))
        }
        claim_completeness = len(supported_claims) / len(required)
        citation_coverage = claim_completeness
    else:
        claim_completeness = 1.0 if not claim_mappings else 0.0
        citation_coverage = claim_completeness

    cited_pairs = [
        (claim_id, chunk_id)
        for claim_id, chunk_ids in claim_mappings.items()
        for chunk_id in chunk_ids
    ]
    correct_pairs = sum(
        claim_id in required and chunk_id in required[claim_id]
        for claim_id, chunk_id in cited_pairs
    )
    support_precision = correct_pairs / len(cited_pairs) if cited_pairs else 1.0

    if question.answerability is Answerability.UNANSWERABLE:
        answerability = float(
            abstained
            and boundary_acknowledged
            and not claim_mappings
            and not forbidden_claims_present
        )
    elif question.answerability is Answerability.PARTIAL:
        answerability = float(not abstained and boundary_acknowledged)
    else:
        answerability = float(not abstained)

    strict_pass = (
        claim_completeness == 1.0
        and support_precision == 1.0
        and citation_coverage == 1.0
        and answerability == 1.0
        and not forbidden_claims_present
    )
    return AnswerScore(
        claim_completeness=claim_completeness,
        claim_support_precision=support_precision,
        citation_coverage=citation_coverage,
        answerability_score=answerability,
        strict_pass=strict_pass,
    )


def aggregate_scores(rows: list[dict[str, Any]], metric_names: list[str]) -> dict[str, float]:
    """Return macro averages over visible per-question metric dictionaries."""

    if not rows:
        return {name: 0.0 for name in metric_names}
    return {
        name: fmean(float(row[name]) for row in rows)
        for name in metric_names
    }


__all__ = [
    "AnswerScore",
    "RetrievalScore",
    "aggregate_scores",
    "score_answer",
    "score_retrieval",
]
