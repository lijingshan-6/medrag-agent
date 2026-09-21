"""Append-only review ledger and schema-constrained local model calls."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
from pydantic import BaseModel

from medrag.benchmark.schema import ReviewEvent, ReviewStatus

_NEXT_STATUS = {
    ReviewStatus.DRAFT: ReviewStatus.EVIDENCE_CHECKED,
    ReviewStatus.EVIDENCE_CHECKED: ReviewStatus.AMBIGUITY_CHECKED,
    ReviewStatus.AMBIGUITY_CHECKED: ReviewStatus.ADVERSARIAL_CHECKED,
    ReviewStatus.ADVERSARIAL_CHECKED: ReviewStatus.ADJUDICATED,
    ReviewStatus.ADJUDICATED: ReviewStatus.FROZEN,
}
_PRIMARY_REVIEW_MODELS = {"qwen3:8b", "qwen3.5:9b"}
_MEDICAL_REVIEW_MODEL = "medgemma1.5:4b"


@dataclass(frozen=True)
class OllamaJSONResult:
    value: dict[str, Any]
    raw_output_sha256: str


def load_review_events(path: Path) -> list[ReviewEvent]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [ReviewEvent.model_validate_json(line) for line in handle if line.strip()]


def events_for_question(
    events: list[ReviewEvent], question_id: str
) -> list[ReviewEvent]:
    return [event for event in events if event.question_id == question_id]


def current_status(events: list[ReviewEvent], question_id: str) -> ReviewStatus:
    relevant = events_for_question(events, question_id)
    return relevant[-1].to_status if relevant else ReviewStatus.DRAFT


def _validate_event(event: ReviewEvent, events: list[ReviewEvent]) -> None:
    current = current_status(events, event.question_id)
    if event.from_status is not current:
        raise ValueError(
            f"{event.question_id}: event starts at {event.from_status.value}, "
            f"current status is {current.value}"
        )

    if event.from_status is event.to_status:
        if not event.model_tag:
            raise ValueError("same-state observations require a model tag")
        return

    expected = _NEXT_STATUS.get(current)
    is_reset = (
        current is ReviewStatus.ADJUDICATED
        and event.to_status is ReviewStatus.EVIDENCE_CHECKED
    )
    if event.to_status is not expected and not is_reset:
        raise ValueError(
            f"{event.question_id}: invalid review transition "
            f"{current.value} -> {event.to_status.value}"
        )

    if event.to_status is ReviewStatus.ADVERSARIAL_CHECKED:
        observed_models = {
            item.model_tag
            for item in events_for_question(events, event.question_id)
            if item.from_status is ReviewStatus.AMBIGUITY_CHECKED
            and item.to_status is ReviewStatus.AMBIGUITY_CHECKED
            and item.model_tag
        }
        missing = []
        if not observed_models.intersection(_PRIMARY_REVIEW_MODELS):
            missing.append("a Qwen primary reviewer")
        if _MEDICAL_REVIEW_MODEL not in observed_models:
            missing.append(_MEDICAL_REVIEW_MODEL)
        if missing:
            raise ValueError(
                f"{event.question_id}: adversarial review is missing models "
                f"{', '.join(missing)}"
            )

    if event.to_status is ReviewStatus.ADJUDICATED:
        if event.actor.casefold() != "curator":
            raise ValueError("adjudication requires a curator")
        if not event.resolution.strip():
            raise ValueError("adjudication requires a source-based curator resolution")


def append_review_event(path: Path, event: ReviewEvent) -> None:
    """Validate and append one event without rewriting prior model opinions."""

    events = load_review_events(path)
    _validate_event(event, events)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(event.model_dump_json() + "\n")


def ollama_json(
    *,
    model: str,
    prompt: str,
    response_model: type[BaseModel],
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 180.0,
) -> OllamaJSONResult:
    """Call Ollama deterministically and validate its visible JSON response."""

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "think": False,
        "format": response_model.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
    content = response.json()["message"]["content"]
    parsed = response_model.model_validate_json(content)
    return OllamaJSONResult(
        value=parsed.model_dump(mode="json"),
        raw_output_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


def ollama_generate_json(
    *,
    model: str,
    prompt: str,
    response_model: type[BaseModel],
    base_url: str = "http://127.0.0.1:11434",
    timeout_seconds: float = 180.0,
) -> OllamaJSONResult:
    """Use Ollama's generate endpoint for models with chat-schema incompatibilities."""

    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "think": False,
        "format": response_model.model_json_schema(),
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{base_url.rstrip('/')}/api/generate", json=payload)
        response.raise_for_status()
    content = response.json()["response"]
    parsed = response_model.model_validate_json(content)
    return OllamaJSONResult(
        value=parsed.model_dump(mode="json"),
        raw_output_sha256=hashlib.sha256(content.encode("utf-8")).hexdigest(),
    )


__all__ = [
    "OllamaJSONResult",
    "append_review_event",
    "current_status",
    "events_for_question",
    "load_review_events",
    "ollama_generate_json",
    "ollama_json",
]
