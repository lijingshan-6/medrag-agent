from __future__ import annotations

from collections import Counter

import pytest

from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.selection import freeze_questions, validate_composition


def _question(
    number: int,
    *,
    task_type: str,
    answerability: str,
    category: str,
    frozen: bool = True,
) -> BenchmarkQuestion:
    required_claims = []
    supporting = []
    if answerability != "unanswerable":
        supporting = [f"pubmed:{number}:0"]
        required_claims = [
            {
                "claim_id": "C1",
                "text": f"Supported claim {number}",
                "importance": "required",
                "evidence": [
                    {
                        "chunk_id": supporting[0],
                        "quote": f"Evidence {number}",
                        "support": "direct",
                    }
                ],
            }
        ]
    return BenchmarkQuestion.model_validate(
        {
            "id": f"VMG-{number:03d}",
            "version": "1.0.0",
            "split": "test",
            "task_type": task_type,
            "answerability": answerability,
            "category": category,
            "difficulty_reason": "Fixture for selection constraints.",
            "question": f"Question {number}?",
            "gold_claims": required_claims,
            "sources": [],
            "supporting_chunk_ids": supporting,
            "hard_negative_chunk_ids": [f"pubmed:negative-{number}:0"],
            "forbidden_claims": [],
            "missing_evidence": (
                [] if answerability == "complete" else ["Requested evidence is absent."]
            ),
            "rubric": {
                "complete_if": ["The supported answer or evidence boundary is stated."],
                "partial_if": ["A required qualification is omitted."],
                "fail_if": ["The answer invents evidence."],
            },
            "review": {
                "evidence_checked": frozen,
                "ambiguity_checked": frozen,
                "answer_checked": frozen,
                "status": "frozen" if frozen else "adjudicated",
            },
        }
    )


def _valid_candidates() -> list[BenchmarkQuestion]:
    specifications = (
        [("single_evidence", "complete")] * 10
        + [("within_document_synthesis", "complete")] * 15
        + [("cross_document_comparison", "complete")] * 10
        + [("limitations_and_safety", "complete")] * 5
        + [("limitations_and_safety", "partial")] * 5
        + [("insufficient_evidence", "unanswerable")] * 5
    )
    return [
        _question(
            number,
            task_type=task,
            answerability=answerability,
            category=f"domain-{(number - 1) % 10}",
        )
        for number, (task, answerability) in enumerate(specifications, start=1)
    ]


def test_freeze_questions_is_deterministic_and_assigns_exact_splits() -> None:
    candidates = list(reversed(_valid_candidates()))
    development_ids = {f"VMG-{number:03d}" for number in range(1, 16)}

    first, first_map = freeze_questions(candidates, development_ids=development_ids)
    second, second_map = freeze_questions(candidates, development_ids=development_ids)

    assert [row.model_dump() for row in first] == [row.model_dump() for row in second]
    assert first_map == second_map
    assert [row.id for row in first] == [f"VMG-{number:03d}" for number in range(1, 51)]
    assert Counter(row.split.value for row in first) == {"development": 15, "test": 35}
    assert validate_composition(first) == []


def test_composition_checks_task_answerability_and_domain_targets() -> None:
    rows = _valid_candidates()
    rows[0] = rows[0].model_copy(update={"category": "domain-1"})
    errors = validate_composition(rows)

    assert any("at least 8 domains" in error or "maximum 7" in error for error in errors) is False
    assert Counter(row.task_type.value for row in rows) == {
        "single_evidence": 10,
        "within_document_synthesis": 15,
        "cross_document_comparison": 10,
        "limitations_and_safety": 10,
        "insufficient_evidence": 5,
    }
    assert Counter(row.answerability.value for row in rows) == {
        "complete": 40,
        "partial": 5,
        "unanswerable": 5,
    }


def test_non_frozen_candidate_is_ineligible() -> None:
    rows = _valid_candidates()
    rows[7] = rows[7].model_copy(
        update={
            "review": rows[7].review.model_copy(update={"status": "adjudicated"})
        }
    )

    with pytest.raises(ValueError, match="not frozen"):
        freeze_questions(
            rows,
            development_ids={f"VMG-{number:03d}" for number in range(1, 16)},
        )
