from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from medrag.benchmark.review import (
    append_review_event,
    current_status,
    load_review_events,
)
from medrag.benchmark.schema import ReviewEvent, ReviewStatus


def _event(
    from_status: ReviewStatus,
    to_status: ReviewStatus,
    *,
    actor: str = "curator",
    model_tag: str | None = None,
    findings: list[str] | None = None,
) -> ReviewEvent:
    return ReviewEvent(
        question_id="VMG-001",
        from_status=from_status,
        to_status=to_status,
        actor=actor,
        created_at=datetime.now(timezone.utc),
        model_tag=model_tag,
        prompt_version="review-v1" if model_tag else None,
        findings=findings or [],
        decision="reviewed" if model_tag else "accepted",
        resolution="Checked against the quoted source passage.",
        raw_output_sha256="a" * 64 if model_tag else None,
    )


def test_review_states_advance_in_order(tmp_path: Path) -> None:
    log = tmp_path / "review.jsonl"
    append_review_event(
        log, _event(ReviewStatus.DRAFT, ReviewStatus.EVIDENCE_CHECKED)
    )
    append_review_event(
        log,
        _event(ReviewStatus.EVIDENCE_CHECKED, ReviewStatus.AMBIGUITY_CHECKED),
    )

    assert current_status(load_review_events(log), "VMG-001") is ReviewStatus.AMBIGUITY_CHECKED


def test_skipping_a_review_state_is_rejected(tmp_path: Path) -> None:
    log = tmp_path / "review.jsonl"

    with pytest.raises(ValueError, match="invalid review transition"):
        append_review_event(
            log, _event(ReviewStatus.DRAFT, ReviewStatus.AMBIGUITY_CHECKED)
        )


def test_model_disagreement_is_preserved_before_adversarial_transition(
    tmp_path: Path,
) -> None:
    log = tmp_path / "review.jsonl"
    append_review_event(
        log, _event(ReviewStatus.DRAFT, ReviewStatus.EVIDENCE_CHECKED)
    )
    append_review_event(
        log,
        _event(ReviewStatus.EVIDENCE_CHECKED, ReviewStatus.AMBIGUITY_CHECKED),
    )
    append_review_event(
        log,
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.AMBIGUITY_CHECKED,
            actor="model",
            model_tag="qwen3:8b",
            findings=["answer is supported"],
        ),
    )
    append_review_event(
        log,
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.AMBIGUITY_CHECKED,
            actor="model",
            model_tag="medgemma1.5:4b",
            findings=["population qualifier is unclear"],
        ),
    )
    append_review_event(
        log,
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.ADVERSARIAL_CHECKED,
        ),
    )

    events = load_review_events(log)
    assert events[2].findings != events[3].findings
    assert [events[2].model_tag, events[3].model_tag] == [
        "qwen3:8b",
        "medgemma1.5:4b",
    ]


def test_adjudication_requires_a_curator_resolution(tmp_path: Path) -> None:
    log = tmp_path / "review.jsonl"
    prior = [
        _event(ReviewStatus.DRAFT, ReviewStatus.EVIDENCE_CHECKED),
        _event(ReviewStatus.EVIDENCE_CHECKED, ReviewStatus.AMBIGUITY_CHECKED),
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.AMBIGUITY_CHECKED,
            actor="model",
            model_tag="qwen3:8b",
        ),
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.AMBIGUITY_CHECKED,
            actor="model",
            model_tag="medgemma1.5:4b",
        ),
        _event(ReviewStatus.AMBIGUITY_CHECKED, ReviewStatus.ADVERSARIAL_CHECKED),
    ]
    for event in prior:
        append_review_event(log, event)

    with pytest.raises(ValueError, match="curator"):
        append_review_event(
            log,
            _event(
                ReviewStatus.ADVERSARIAL_CHECKED,
                ReviewStatus.ADJUDICATED,
                actor="model",
                model_tag="qwen3:8b",
            ),
        )
