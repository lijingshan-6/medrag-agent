from __future__ import annotations

import copy

import pytest
from pydantic import ValidationError

from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.validation import validate_questions


def _valid_question() -> dict:
    return {
        "id": "VMG-001",
        "version": "1.0.0",
        "split": "development",
        "task_type": "single_evidence",
        "answerability": "complete",
        "category": "cardiology",
        "difficulty_reason": "Requires extracting a qualified outcome.",
        "question": "What outcome was reported for the study population?",
        "gold_claims": [
            {
                "claim_id": "C1",
                "text": "The intervention reduced the measured outcome at 12 weeks.",
                "importance": "required",
                "evidence": [
                    {
                        "chunk_id": "pmc:doc1:0",
                        "quote": "The intervention reduced the measured outcome at 12 weeks.",
                        "support": "direct",
                    }
                ],
            }
        ],
        "sources": [
            {
                "source_id": "PMC:doc1",
                "pmid": "12345678",
                "pmcid": "PMC1234567",
                "title": "Example study",
                "year": 2024,
                "publication_type": "Journal Article",
                "study_design": "randomized trial",
                "role": "primary_study",
                "limitations": ["Single-centre study"],
            }
        ],
        "supporting_chunk_ids": [],
        "hard_negative_chunk_ids": ["pmc:doc1:1"],
        "forbidden_claims": ["The result applies to every patient."],
        "missing_evidence": [],
        "rubric": {
            "complete_if": ["C1 is present with the 12-week qualifier."],
            "partial_if": ["The direction is present without the time point."],
            "fail_if": ["The answer generalises beyond the study population."],
        },
        "review": {
            "evidence_checked": True,
            "ambiguity_checked": True,
            "answer_checked": True,
            "status": "adjudicated",
        },
    }


def test_valid_complete_question_parses() -> None:
    question = BenchmarkQuestion.model_validate(_valid_question())

    assert question.id == "VMG-001"
    assert question.gold_claims[0].evidence[0].chunk_id == "pmc:doc1:0"


@pytest.mark.parametrize("answerability", ["partial", "unanswerable"])
def test_incomplete_answerability_requires_missing_evidence(answerability: str) -> None:
    row = _valid_question()
    row["answerability"] = answerability

    with pytest.raises(ValidationError, match="missing_evidence"):
        BenchmarkQuestion.model_validate(row)


def test_required_claim_requires_evidence() -> None:
    row = _valid_question()
    row["gold_claims"][0]["evidence"] = []

    with pytest.raises(ValidationError, match="Required claims need evidence"):
        BenchmarkQuestion.model_validate(row)


def test_duplicate_claim_ids_are_rejected() -> None:
    row = _valid_question()
    row["gold_claims"].append(copy.deepcopy(row["gold_claims"][0]))

    with pytest.raises(ValidationError, match="Duplicate claim_id"):
        BenchmarkQuestion.model_validate(row)


def test_quote_must_occur_in_referenced_chunk() -> None:
    question = BenchmarkQuestion.model_validate(_valid_question())

    errors = validate_questions(
        [question],
        {
            "pmc:doc1:0": "The control group showed no meaningful change.",
            "pmc:doc1:1": "A plausible but non-answering neighbouring passage.",
        },
    )

    assert errors == [
        "VMG-001/C1/pmc:doc1:0: evidence quote does not occur in referenced chunk"
    ]


def test_quote_matching_normalizes_whitespace_only() -> None:
    question = BenchmarkQuestion.model_validate(_valid_question())

    errors = validate_questions(
        [question],
        {
            "pmc:doc1:0": (
                "The intervention reduced   the measured outcome\n"
                "at 12 weeks."
            ),
            "pmc:doc1:1": "A plausible but non-answering neighbouring passage.",
        },
    )

    assert errors == []


def test_validation_reports_duplicate_question_ids_and_missing_chunks() -> None:
    first = BenchmarkQuestion.model_validate(_valid_question())
    second = BenchmarkQuestion.model_validate(_valid_question())

    errors = validate_questions([first, second], {})

    assert "VMG-001: duplicate question id" in errors
    assert (
        "VMG-001/C1/pmc:doc1:0: referenced chunk is missing from snapshot"
        in errors
    )
