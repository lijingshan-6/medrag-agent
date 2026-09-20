from __future__ import annotations

import pytest

from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.scoring import score_answer, score_retrieval


def _question(answerability: str = "complete") -> BenchmarkQuestion:
    claims = []
    missing = []
    if answerability != "unanswerable":
        claims = [
            {
                "claim_id": "C1",
                "text": "First supported claim.",
                "importance": "required",
                "evidence": [
                    {"chunk_id": "a", "quote": "A", "support": "direct"},
                    {"chunk_id": "b", "quote": "B", "support": "context"},
                ],
            },
            {
                "claim_id": "C2",
                "text": "Second supported claim.",
                "importance": "required",
                "evidence": [
                    {"chunk_id": "c", "quote": "C", "support": "derived", "derivation": "C supports the bounded inference."}
                ],
            },
        ]
    if answerability != "complete":
        missing = ["The requested comparison is absent."]
    return BenchmarkQuestion.model_validate(
        {
            "id": "VMG-001",
            "version": "1.0.0",
            "split": "development",
            "task_type": "insufficient_evidence" if answerability == "unanswerable" else "within_document_synthesis",
            "answerability": answerability,
            "category": "cardiology",
            "difficulty_reason": "Scoring fixture.",
            "question": "What does the evidence show?",
            "gold_claims": claims,
            "sources": [],
            "supporting_chunk_ids": [] if answerability == "unanswerable" else ["a", "b", "c"],
            "hard_negative_chunk_ids": ["x"],
            "forbidden_claims": ["Unsupported recommendation"],
            "missing_evidence": missing,
            "rubric": {
                "complete_if": ["All evidence boundaries are preserved."],
                "partial_if": ["One qualifier is absent."],
                "fail_if": ["Unsupported claim."],
            },
            "review": {"status": "frozen", "evidence_checked": True, "ambiguity_checked": True, "answer_checked": True},
        }
    )


def test_retrieval_metrics_match_hand_calculation() -> None:
    result = score_retrieval(_question(), ["a", "x", "c", "z"], k=4)

    assert result.required_claim_recall == 1.0
    assert result.supporting_chunk_recall == pytest.approx(2 / 3)
    assert result.all_required_found is True
    assert result.hard_negative_hit is True
    assert result.reciprocal_rank == 1.0
    assert result.ndcg == pytest.approx((1 + 1 / 2) / (1 + 1 / 1.5849625))


def test_answer_scoring_rewards_supported_mappings_and_boundary() -> None:
    result = score_answer(
        _question(answerability="partial"),
        claim_mappings={"C1": ["a"], "C2": ["c", "z"]},
        abstained=False,
        boundary_acknowledged=True,
        forbidden_claims_present=False,
    )

    assert result.claim_completeness == 1.0
    assert result.claim_support_precision == pytest.approx(2 / 3)
    assert result.citation_coverage == 1.0
    assert result.answerability_score == 1.0
    assert result.strict_pass is False


def test_unanswerable_question_passes_only_with_clean_abstention() -> None:
    question = _question(answerability="unanswerable")

    passing = score_answer(
        question,
        claim_mappings={},
        abstained=True,
        boundary_acknowledged=True,
        forbidden_claims_present=False,
    )
    failing = score_answer(
        question,
        claim_mappings={"invented": ["x"]},
        abstained=False,
        boundary_acknowledged=False,
        forbidden_claims_present=True,
    )

    assert passing.strict_pass is True
    assert passing.answerability_score == 1.0
    assert failing.strict_pass is False
    assert failing.answerability_score == 0.0
