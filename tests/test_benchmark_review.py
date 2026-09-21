from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
import sys

import pytest
from pydantic import BaseModel

from medrag.benchmark import review as review_module
from medrag.benchmark.review import (
    append_review_event,
    current_status,
    load_review_events,
)
from medrag.benchmark.schema import ReviewEvent, ReviewStatus


def _load_review_candidates_module():
    path = Path(__file__).parents[1] / "scripts" / "benchmark" / "review_candidates.py"
    spec = importlib.util.spec_from_file_location("review_candidates_test", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ReviewBatch.model_rebuild()
    return module


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


def test_qwen35_can_fill_the_primary_review_role(tmp_path: Path) -> None:
    log = tmp_path / "review.jsonl"
    prior = [
        _event(ReviewStatus.DRAFT, ReviewStatus.EVIDENCE_CHECKED),
        _event(ReviewStatus.EVIDENCE_CHECKED, ReviewStatus.AMBIGUITY_CHECKED),
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.AMBIGUITY_CHECKED,
            actor="model",
            model_tag="qwen3.5:9b",
        ),
        _event(
            ReviewStatus.AMBIGUITY_CHECKED,
            ReviewStatus.AMBIGUITY_CHECKED,
            actor="model",
            model_tag="medgemma1.5:4b",
        ),
    ]
    for event in prior:
        append_review_event(log, event)

    append_review_event(
        log,
        _event(ReviewStatus.AMBIGUITY_CHECKED, ReviewStatus.ADVERSARIAL_CHECKED),
    )

    assert current_status(load_review_events(log), "VMG-001") is ReviewStatus.ADVERSARIAL_CHECKED


def test_model_review_schema_bounds_each_generated_text() -> None:
    module = _load_review_candidates_module()
    schema = module.ReviewBatch.model_json_schema()
    properties = schema["$defs"]["ModelReview"]["properties"]

    assert properties["findings"]["items"]["maxLength"] == 240
    assert properties["reconstructed_claims"]["items"]["maxLength"] == 500


def test_revise_without_findings_uses_the_evidence_boundary() -> None:
    module = _load_review_candidates_module()
    review = module.ModelReview(
        question_id="VMG-001",
        reconstructed_claims=[],
        findings=[],
        decision="revise",
        evidence_boundary="The passage does not support the requested treatment comparison.",
    )

    assert module._effective_findings(review) == [
        "The passage does not support the requested treatment comparison."
    ]


def test_generate_json_uses_ollama_generate_endpoint(monkeypatch) -> None:
    class Result(BaseModel):
        decision: str

    calls = []

    class Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict:
            return {"response": '{"decision":"pass"}'}

    class Client:
        def __init__(self, *, timeout: float):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return None

        def post(self, url: str, *, json: dict):
            calls.append((url, json))
            return Response()

    monkeypatch.setattr(review_module.httpx, "Client", Client)

    result = review_module.ollama_generate_json(
        model="llama3.1:8b",
        prompt="Review one item.",
        response_model=Result,
    )

    assert result.value == {"decision": "pass"}
    assert calls[0][0] == "http://127.0.0.1:11434/api/generate"
    assert calls[0][1]["format"] == Result.model_json_schema()
    assert calls[0][1]["think"] is False


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
