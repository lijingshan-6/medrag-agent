"""Deterministic quality checks for a frozen benchmark question set."""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Annotated, Any, Literal

from pydantic import Field

from medrag.benchmark.schema import (
    Answerability,
    BenchmarkQuestion,
    GoldClaim,
    QuestionRubric,
    SourceRecord,
    StrictModel,
)


class AuditDecision(StrictModel):
    """One source-first curator decision after a blind challenge."""

    question_id: str = Field(pattern=r"^VMG-[0-9]{3}$")
    decision: Literal["keep", "revise", "replace"]
    evidence_entailment: Literal["pass", "fail"]
    qualifier_integrity: Literal["pass", "fail"]
    ambiguity: Literal["pass", "fail"]
    answerability: Literal["pass", "fail"]
    hard_negative: Literal["pass", "fail", "not_applicable"]
    rubric: Literal["pass", "fail"]
    challenger_model: str | None = None
    challenger_decision: Literal["pass", "revise"] | None = None
    challenger_raw_output_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )
    reviewer_findings: list[str] = Field(default_factory=list)
    curator_resolution: str = Field(min_length=10)


AuditFinding = Annotated[str, Field(min_length=1, max_length=240)]


class BlindAuditReview(StrictModel):
    """Schema-constrained challenge from a model outside the drafting families."""

    question_id: str = Field(pattern=r"^VMG-[0-9]{3}$")
    decision: Literal["pass", "revise"]
    evidence_entailment: Literal["pass", "fail"]
    qualifier_integrity: Literal["pass", "fail"]
    ambiguity: Literal["pass", "fail"]
    answerability: Literal["pass", "fail"]
    hard_negative: Literal["pass", "fail", "not_applicable"]
    rubric: Literal["pass", "fail"]
    findings: list[AuditFinding] = Field(default_factory=list, max_length=6)
    rationale: str = Field(min_length=10, max_length=600)


def normalize_blind_review(
    question: BenchmarkQuestion, review: BlindAuditReview
) -> BlindAuditReview:
    """Derive a consistent challenge decision while preserving raw output separately."""

    hard_negative = (
        review.hard_negative
        if question.hard_negative_chunk_ids
        else "not_applicable"
    )
    dimensions = [
        review.evidence_entailment,
        review.qualifier_integrity,
        review.ambiguity,
        review.answerability,
        review.rubric,
    ]
    if hard_negative != "not_applicable":
        dimensions.append(hard_negative)
    decision = (
        "revise"
        if review.decision == "revise"
        or review.findings
        or any(value == "fail" for value in dimensions)
        else "pass"
    )
    findings = list(review.findings)
    if decision == "revise" and not findings:
        findings = [review.rationale[:240].rstrip()]
    return review.model_copy(
        update={
            "decision": decision,
            "hard_negative": hard_negative,
            "findings": findings,
        }
    )


class QuestionRevision(StrictModel):
    """Explicit source-first correction applied when freezing a new version."""

    question_id: str = Field(pattern=r"^VMG-[0-9]{3}$")
    reason: str = Field(min_length=10)
    question: str | None = None
    difficulty_reason: str | None = None
    gold_claims: list[GoldClaim] | None = None
    sources: list[SourceRecord] | None = None
    supporting_chunk_ids: list[str] | None = None
    forbidden_claims: list[str] | None = None
    missing_evidence: list[str] | None = None
    rubric: QuestionRubric | None = None


def _normalized(text: str) -> str:
    return re.sub(r"\W+", " ", text.casefold()).strip()


def _referenced_chunks(question: BenchmarkQuestion) -> set[str]:
    chunks = set(question.supporting_chunk_ids) | set(question.hard_negative_chunk_ids)
    chunks.update(
        evidence.chunk_id
        for claim in question.gold_claims
        for evidence in claim.evidence
    )
    return chunks


def benchmark_profile(questions: list[BenchmarkQuestion]) -> dict[str, Any]:
    """Return high-signal shape, duplication, and split-integrity checks."""

    sources_by_split: dict[str, set[str]] = defaultdict(set)
    chunks_by_split: dict[str, set[str]] = defaultdict(set)
    years: Counter[str] = Counter()
    designs: Counter[str] = Counter()
    questions_normalized: Counter[str] = Counter()
    claims_normalized: Counter[str] = Counter()

    for question in questions:
        split = question.split.value
        questions_normalized[_normalized(question.question)] += 1
        chunks_by_split[split].update(_referenced_chunks(question))
        for claim in question.gold_claims:
            claims_normalized[_normalized(claim.text)] += 1
        for source in question.sources:
            sources_by_split[split].add(source.source_id)
            years[str(source.year) if source.year is not None else "missing"] += 1
            designs[source.study_design] += 1

    development_sources = sources_by_split["development"]
    test_sources = sources_by_split["test"]
    development_chunks = chunks_by_split["development"]
    test_chunks = chunks_by_split["test"]
    return {
        "question_count": len(questions),
        "duplicate_question_count": sum(
            count - 1 for count in questions_normalized.values() if count > 1
        ),
        "duplicate_claim_count": sum(
            count - 1 for count in claims_normalized.values() if count > 1
        ),
        "source_overlap_ids": sorted(development_sources & test_sources),
        "evidence_chunk_overlap_ids": sorted(development_chunks & test_chunks),
        "source_years": dict(sorted(years.items())),
        "study_designs": dict(sorted(designs.items())),
    }


def build_blind_audit_prompt(
    question: BenchmarkQuestion, chunks: dict[str, str]
) -> str:
    """Build a source-only audit packet without previous decisions or outputs."""

    claims: list[str] = []
    for claim in question.gold_claims:
        evidence_rows = []
        for evidence in claim.evidence:
            frozen_text = chunks.get(evidence.chunk_id, "<missing chunk>")
            evidence_rows.append(
                f"- {evidence.chunk_id} ({evidence.support.value})\n"
                f"  Gold quote: {evidence.quote}\n"
                f"  Frozen passage: {frozen_text}"
            )
        claims.append(
            f"{claim.claim_id} [{claim.importance.value}]: {claim.text}\n"
            + "\n".join(evidence_rows)
        )
    negatives = [
        f"- {chunk_id}: {chunks.get(chunk_id, '<missing chunk>')}"
        for chunk_id in question.hard_negative_chunk_ids
    ]
    sources = [
        f"- {source.source_id}; {source.title}; {source.year}; "
        f"design={source.study_design}; limitations={source.limitations}"
        for source in question.sources
    ]
    return f"""Audit one medical-literature benchmark item using only the frozen text below.
Do not use outside medical knowledge. Check whether every gold claim follows from its quoted passage,
including population, intervention or exposure, comparator, outcome, number, unit, time point,
negation, uncertainty, and causal strength. Check that the question has one clear interpretation,
the complete/partial/unanswerable label is correct, the rubric is deterministic, and any near-topic
hard negative truly omits the declared missing evidence. A plausible claim is a failure when the
frozen text does not state it. Set hard_negative to not_applicable when no near-topic negative is
supplied. Return pass only when every applicable dimension passes and findings is empty. Otherwise
return revise with at least one concise finding. Keep the rationale to one or two sentences.

Question ID: {question.id}
Question: {question.question}
Task type: {question.task_type.value}
Answerability: {question.answerability.value}
Missing evidence: {question.missing_evidence}
Forbidden claims: {question.forbidden_claims}
Rubric complete_if: {question.rubric.complete_if}
Rubric partial_if: {question.rubric.partial_if}
Rubric fail_if: {question.rubric.fail_if}

Sources:
{chr(10).join(sources)}

Gold claims and frozen evidence:
{chr(10).join(claims) if claims else '<no answerable gold claim>'}

Near-topic hard negatives:
{chr(10).join(negatives) if negatives else '<not applicable>'}
"""


def validate_audit_decisions(
    questions: list[BenchmarkQuestion], decisions: list[AuditDecision]
) -> list[str]:
    """Validate that every question has one resolved, internally consistent audit."""

    errors: list[str] = []
    expected_ids = {question.id for question in questions}
    counts = Counter(decision.question_id for decision in decisions)
    for question_id in sorted(expected_ids - set(counts)):
        errors.append(f"missing audit decision: {question_id}")
    for question_id in sorted(set(counts) - expected_ids):
        errors.append(f"unknown audit decision: {question_id}")
    for question_id, count in sorted(counts.items()):
        if count > 1:
            errors.append(f"duplicate audit decision: {question_id}")

    for decision in decisions:
        applicable = [
            decision.evidence_entailment,
            decision.qualifier_integrity,
            decision.ambiguity,
            decision.answerability,
            decision.rubric,
        ]
        if decision.hard_negative != "not_applicable":
            applicable.append(decision.hard_negative)
        if decision.decision == "keep" and any(value != "pass" for value in applicable):
            errors.append(
                f"{decision.question_id}: keep requires every applicable dimension to pass"
            )
    return errors


def source_connected_groups(
    questions: list[BenchmarkQuestion],
) -> list[list[str]]:
    """Group questions connected by a source or gold-evidence chunk."""

    parent = {question.id: question.id for question in questions}

    def find(question_id: str) -> str:
        while parent[question_id] != question_id:
            parent[question_id] = parent[parent[question_id]]
            question_id = parent[question_id]
        return question_id

    def union(left: str, right: str) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            earlier, later = sorted((left_root, right_root))
            parent[later] = earlier

    owners: dict[str, str] = {}
    for question in questions:
        keys = {f"source:{source.source_id}" for source in question.sources}
        keys.update(
            f"chunk:{evidence.chunk_id}"
            for claim in question.gold_claims
            for evidence in claim.evidence
        )
        for key in keys:
            if key in owners:
                union(question.id, owners[key])
            else:
                owners[key] = question.id

    grouped: dict[str, list[str]] = defaultdict(list)
    for question in questions:
        grouped[find(question.id)].append(question.id)
    return sorted((sorted(group) for group in grouped.values()), key=lambda row: row[0])


def validate_source_disjoint_split(
    questions: list[BenchmarkQuestion],
) -> list[str]:
    """Reject a split that places a source-connected group on both sides."""

    by_id = {question.id: question for question in questions}
    errors = []
    for group in source_connected_groups(questions):
        splits = {by_id[question_id].split.value for question_id in group}
        if len(splits) > 1:
            errors.append(f"source-connected group crosses splits: {', '.join(group)}")
    return errors


def freeze_audited_questions(
    questions: list[BenchmarkQuestion],
    decisions: list[AuditDecision],
    revisions: list[QuestionRevision],
    *,
    development_ids: set[str],
    development_count: int = 15,
    source_designs: dict[str, str] | None = None,
) -> list[BenchmarkQuestion]:
    """Apply resolved revisions and create a source-disjoint v1.1 question set."""

    audit_errors = validate_audit_decisions(questions, decisions)
    if audit_errors:
        raise ValueError("; ".join(audit_errors))
    known_ids = {question.id for question in questions}
    if not development_ids <= known_ids:
        unknown = ", ".join(sorted(development_ids - known_ids))
        raise ValueError(f"unknown development IDs: {unknown}")
    if len(development_ids) != development_count:
        raise ValueError(
            f"development split must contain {development_count} questions, "
            f"got {len(development_ids)}"
        )

    revision_counts = Counter(revision.question_id for revision in revisions)
    duplicate_revisions = sorted(
        question_id for question_id, count in revision_counts.items() if count > 1
    )
    if duplicate_revisions:
        raise ValueError("duplicate revisions: " + ", ".join(duplicate_revisions))
    revision_by_id = {revision.question_id: revision for revision in revisions}
    decision_by_id = {decision.question_id: decision for decision in decisions}
    for question_id, decision in decision_by_id.items():
        if decision.decision != "keep" and question_id not in revision_by_id:
            raise ValueError(f"{question_id}: {decision.decision} requires a revision")

    frozen: list[BenchmarkQuestion] = []
    revision_fields = (
        "question",
        "difficulty_reason",
        "gold_claims",
        "sources",
        "supporting_chunk_ids",
        "forbidden_claims",
        "missing_evidence",
        "rubric",
    )
    for question in questions:
        row = question.model_dump(mode="json")
        revision = revision_by_id.get(question.id)
        if revision is not None:
            for field in revision_fields:
                value = getattr(revision, field)
                if value is not None:
                    if hasattr(value, "model_dump"):
                        value = value.model_dump(mode="json")
                    elif isinstance(value, list):
                        value = [
                            item.model_dump(mode="json")
                            if hasattr(item, "model_dump")
                            else item
                            for item in value
                        ]
                    row[field] = value
        row["version"] = "1.1.0"
        row["split"] = (
            "development" if question.id in development_ids else "test"
        )
        if source_designs is not None:
            for source in row["sources"]:
                if source["source_id"] in source_designs:
                    source["study_design"] = source_designs[source["source_id"]]
        if question.answerability is not Answerability.UNANSWERABLE:
            row["hard_negative_chunk_ids"] = []
        frozen.append(BenchmarkQuestion.model_validate(row))

    split_errors = validate_source_disjoint_split(frozen)
    if split_errors:
        raise ValueError("; ".join(split_errors))
    return frozen


__all__ = [
    "AuditDecision",
    "BlindAuditReview",
    "QuestionRevision",
    "benchmark_profile",
    "build_blind_audit_prompt",
    "normalize_blind_review",
    "source_connected_groups",
    "freeze_audited_questions",
    "validate_source_disjoint_split",
    "validate_audit_decisions",
]
