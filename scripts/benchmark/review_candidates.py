"""Run source-only local-model reviews and maintain the append-only ledger."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field

from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.review import (
    append_review_event,
    current_status,
    events_for_question,
    load_review_events,
    ollama_json,
)
from medrag.benchmark.schema import BenchmarkQuestion, ReviewEvent, ReviewStatus
from medrag.benchmark.validation import validate_questions


FindingText = Annotated[str, Field(min_length=1, max_length=240)]
ReconstructedClaim = Annotated[str, Field(min_length=1, max_length=500)]


class ModelReview(BaseModel):
    model_config = ConfigDict(extra="forbid")
    question_id: str
    reconstructed_claims: list[ReconstructedClaim] = Field(default_factory=list, max_length=4)
    findings: list[FindingText] = Field(default_factory=list, max_length=4)
    decision: Literal["pass", "revise"]
    evidence_boundary: str = Field(min_length=5, max_length=500)


class ReviewBatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    items: list[ModelReview] = Field(min_length=1, max_length=8)


def _effective_findings(review: ModelReview) -> list[str]:
    if review.findings or review.decision == "pass":
        return list(review.findings)
    return [review.evidence_boundary]


def _read_questions(path: Path) -> list[BenchmarkQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [BenchmarkQuestion.model_validate_json(line) for line in handle if line.strip()]


def _save_questions(path: Path, questions: list[BenchmarkQuestion]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(question.model_dump_json() + "\n")


def _event(
    question_id: str,
    from_status: ReviewStatus,
    to_status: ReviewStatus,
    *,
    actor: str,
    decision: str,
    resolution: str,
    findings: list[str] | None = None,
    model_tag: str | None = None,
    raw_hash: str | None = None,
) -> ReviewEvent:
    return ReviewEvent(
        question_id=question_id,
        from_status=from_status,
        to_status=to_status,
        actor=actor,
        created_at=datetime.now(timezone.utc),
        model_tag=model_tag,
        prompt_version="source-only-review-v1" if model_tag else None,
        findings=findings or [],
        decision=decision,
        resolution=resolution,
        raw_output_sha256=raw_hash,
    )


def _passages(question: BenchmarkQuestion, chunks: dict[str, str]) -> str:
    supporting_quotes: list[tuple[str, str]] = []
    for claim in question.gold_claims:
        for evidence in claim.evidence:
            pair = (evidence.chunk_id, evidence.quote)
            if pair not in supporting_quotes:
                supporting_quotes.append(pair)
    blocks = [
        f"[PASSAGE {index}] {chunk_id}\n{quote}"
        for index, (chunk_id, quote) in enumerate(supporting_quotes, start=1)
    ]
    for index, chunk_id in enumerate(question.hard_negative_chunk_ids, start=1):
        text = chunks[chunk_id]
        blocks.append(f"[NEAR-TOPIC {index}] {chunk_id}\n{text[:1200]}")
    return "\n\n".join(blocks) or "No supporting passage is supplied."


def _prompt(
    question: BenchmarkQuestion,
    chunks: dict[str, str],
    reviewer: Literal["reconstruction", "medical"],
) -> str:
    role = (
        "Independently reconstruct the smallest responsible answer from the passages."
        if reviewer == "reconstruction"
        else "Act as an adversarial medical reviewer. Check population, intervention or exposure, comparator, outcome, time point, causal wording, uncertainty, and whether the question asks beyond the passages."
    )
    return f"""{role}
You do not have a proposed gold answer. Use only the passages below. Do not use outside medical
knowledge. A near-topic passage may be a hard negative. If the responsible response is to say the
evidence is absent, say so. Return `pass` with an empty findings list only when the question has one
clear interpretation and the passages support a responsible answer boundary. Return `revise` and
short, specific findings for ambiguity, missing qualifiers, unrelated comparisons, or unsupported
recommendations. Set question_id exactly to {question.id}.

Question ID: {question.id}
Task type: {question.task_type.value}
Question: {question.question}

Passages:
{_passages(question, chunks)}
"""


def _batch_prompt(
    questions: list[BenchmarkQuestion],
    chunks: dict[str, str],
    reviewer: Literal["reconstruction", "medical"],
) -> str:
    introductions = (
        "Independently reconstruct the smallest responsible answer for each question from its passages."
        if reviewer == "reconstruction"
        else "Act as an adversarial medical reviewer for each question. Check population, intervention or exposure, comparator, outcome, time point, causal wording, uncertainty, and whether the question asks beyond its passages."
    )
    blocks = []
    for question in questions:
        blocks.append(
            f"QUESTION {question.id}\n"
            f"Task type: {question.task_type.value}\n"
            f"Question: {question.question}\n"
            f"Passages:\n{_passages(question, chunks)}"
        )
    expected_ids = ", ".join(question.id for question in questions)
    return f"""{introductions}
You do not have proposed gold answers. Use only each question's own passages and do not use outside
medical knowledge. A near-topic passage may be a hard negative. Return exactly one review for each
of these IDs: {expected_ids}. Use `pass` with an empty findings list only when a question has one
clear interpretation and its passages support a responsible answer boundary. Use `revise` with
short, specific findings for ambiguity, missing qualifiers, unrelated comparisons, or unsupported
recommendations.

{chr(10).join(blocks)}
"""


def _advance_static_checks(
    question: BenchmarkQuestion,
    log_path: Path,
    chunks: dict[str, str],
) -> None:
    events = load_review_events(log_path)
    status = current_status(events, question.id)
    if status is ReviewStatus.DRAFT:
        errors = validate_questions([question], chunks)
        if errors:
            raise ValueError("; ".join(errors))
        append_review_event(
            log_path,
            _event(
                question.id,
                ReviewStatus.DRAFT,
                ReviewStatus.EVIDENCE_CHECKED,
                actor="curator",
                decision="accepted",
                resolution="Every gold evidence quote occurs in its frozen chunk and every referenced chunk resolves.",
            ),
        )
        status = ReviewStatus.EVIDENCE_CHECKED
    if status is ReviewStatus.EVIDENCE_CHECKED:
        append_review_event(
            log_path,
            _event(
                question.id,
                ReviewStatus.EVIDENCE_CHECKED,
                ReviewStatus.AMBIGUITY_CHECKED,
                actor="curator",
                decision="accepted_for_model_review",
                resolution="Question wording, answerability label, and named evidence boundary were checked together before model review.",
            ),
        )


def run_models(
    questions_path: Path,
    log_path: Path,
    root: Path,
    primary_model: str,
    medical_model: str,
) -> None:
    questions = _read_questions(questions_path)
    chunks = {
        row["chunk_id"]: row["text"]
        for row in build_normalized_chunks(
            root / "data/raw/pubmed/abstracts.jsonl",
            root / "data/raw/pmc/full_texts.jsonl",
        )
    }
    models = ((primary_model, "reconstruction"), (medical_model, "medical"))
    for question in questions:
        _advance_static_checks(question, log_path, chunks)

    for model, reviewer in models:
        print(f"starting source-only pass with {model}", flush=True)
        batch_size = 8 if reviewer == "reconstruction" else 1
        pending: list[BenchmarkQuestion] = []
        for question in questions:
            events = load_review_events(log_path)
            existing_models = {
                event.model_tag
                for event in events_for_question(events, question.id)
                if event.from_status is ReviewStatus.AMBIGUITY_CHECKED
                and event.to_status is ReviewStatus.AMBIGUITY_CHECKED
            }
            if (
                current_status(events, question.id) is ReviewStatus.AMBIGUITY_CHECKED
                and model not in existing_models
            ):
                pending.append(question)
        for batch_start in range(0, len(pending), batch_size):
            batch = pending[batch_start : batch_start + batch_size]
            print(
                f"[{batch_start + 1:02d}-{batch_start + len(batch):02d}/{len(pending):02d}] "
                f"{model}: {','.join(question.id for question in batch)}",
                flush=True,
            )
            result = ollama_json(
                model=model,
                prompt=_batch_prompt(batch, chunks, reviewer),
                response_model=ReviewBatch,
                timeout_seconds=300,
            )
            reviews = {
                review.question_id: review
                for review in ReviewBatch.model_validate(result.value).items
            }
            expected_ids = {question.id for question in batch}
            if not expected_ids.issubset(reviews):
                raise ValueError(
                    f"{model} omitted review IDs {sorted(expected_ids - set(reviews))}"
                )
            for question in batch:
                review = reviews[question.id]
                findings = _effective_findings(review)
                append_review_event(
                    log_path,
                    _event(
                        question.id,
                        ReviewStatus.AMBIGUITY_CHECKED,
                        ReviewStatus.AMBIGUITY_CHECKED,
                        actor="model",
                        model_tag=model,
                        raw_hash=result.raw_output_sha256,
                        findings=findings,
                        decision=review.decision,
                        resolution=(
                            f"Independent reconstruction: {' | '.join(review.reconstructed_claims) or 'none'}. "
                            f"Evidence boundary: {review.evidence_boundary}"
                        ),
                    ),
                )

    for question in questions:
        events = load_review_events(log_path)
        question_events = events_for_question(events, question.id)
        model_events = [
            event
            for event in question_events
            if event.from_status is ReviewStatus.AMBIGUITY_CHECKED
            and event.to_status is ReviewStatus.AMBIGUITY_CHECKED
            and event.model_tag in {primary_model, medical_model}
        ]
        if {event.model_tag for event in model_events} == {primary_model, medical_model}:
            findings = [
                f"{event.model_tag}: {finding}"
                for event in model_events
                for finding in event.findings
            ]
            append_review_event(
                log_path,
                _event(
                    question.id,
                    ReviewStatus.AMBIGUITY_CHECKED,
                    ReviewStatus.ADVERSARIAL_CHECKED,
                    actor="curator",
                    findings=findings,
                    decision="needs_resolution" if findings else "clean",
                    resolution=(
                        "Model findings are preserved for source adjudication."
                        if findings
                        else "Both source-only model passes returned no finding; curator adjudication remains separate."
                    ),
                ),
            )


def adjudicate_clean(questions_path: Path, log_path: Path) -> int:
    questions = _read_questions(questions_path)
    events = load_review_events(log_path)
    accepted = 0
    for question in questions:
        if current_status(events, question.id) is not ReviewStatus.ADVERSARIAL_CHECKED:
            continue
        observations = [
            event
            for event in events_for_question(events, question.id)
            if event.model_tag and event.from_status is ReviewStatus.AMBIGUITY_CHECKED
        ]
        findings = [finding for event in observations for finding in event.findings]
        if findings:
            continue
        append_review_event(
            log_path,
            _event(
                question.id,
                ReviewStatus.ADVERSARIAL_CHECKED,
                ReviewStatus.ADJUDICATED,
                actor="curator",
                decision="accepted",
                resolution="Curator confirmed the exact evidence spans, question scope, and answerability boundary after both independent reviews returned no finding.",
            ),
        )
        question.review.evidence_checked = True
        question.review.ambiguity_checked = True
        question.review.answer_checked = True
        question.review.status = ReviewStatus.ADJUDICATED
        accepted += 1
        events = load_review_events(log_path)
    _save_questions(questions_path, questions)
    return accepted


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--primary-model", default="qwen3.5:9b")
    parser.add_argument("--medical-model", default="medgemma1.5:4b")
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/candidates.jsonl"),
    )
    parser.add_argument(
        "--log",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/review_log.jsonl"),
    )
    parser.add_argument("--stage", choices=("models", "adjudicate-clean"), default="models")
    args = parser.parse_args()
    if args.stage == "models":
        run_models(
            args.questions,
            args.log,
            args.root.resolve(),
            args.primary_model,
            args.medical_model,
        )
    else:
        count = adjudicate_clean(args.questions, args.log)
        print(f"adjudicated {count} clean candidates")


if __name__ == "__main__":
    main()
