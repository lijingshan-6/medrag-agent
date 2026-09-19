"""Pydantic contract for the VeritasMed evidence-grounded benchmark."""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Forbid accidental schema drift in committed benchmark artifacts."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class BenchmarkSplit(str, Enum):
    DEVELOPMENT = "development"
    TEST = "test"


class TaskType(str, Enum):
    SINGLE_EVIDENCE = "single_evidence"
    WITHIN_DOCUMENT_SYNTHESIS = "within_document_synthesis"
    CROSS_DOCUMENT_COMPARISON = "cross_document_comparison"
    LIMITATIONS_AND_SAFETY = "limitations_and_safety"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class Answerability(str, Enum):
    COMPLETE = "complete"
    PARTIAL = "partial"
    UNANSWERABLE = "unanswerable"


class SupportLevel(str, Enum):
    DIRECT = "direct"
    DERIVED = "derived"
    CONTEXT = "context"


class ClaimImportance(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class ReviewStatus(str, Enum):
    DRAFT = "draft"
    EVIDENCE_CHECKED = "evidence_checked"
    AMBIGUITY_CHECKED = "ambiguity_checked"
    ADVERSARIAL_CHECKED = "adversarial_checked"
    ADJUDICATED = "adjudicated"
    FROZEN = "frozen"


class EvidenceSpan(StrictModel):
    chunk_id: str = Field(min_length=1)
    quote: str = Field(min_length=1)
    support: SupportLevel
    derivation: str | None = None

    @model_validator(mode="after")
    def require_derivation_note(self) -> "EvidenceSpan":
        if self.support is SupportLevel.DERIVED and not self.derivation:
            raise ValueError("Derived evidence requires a derivation note")
        return self


class GoldClaim(StrictModel):
    claim_id: str = Field(pattern=r"^C[1-9][0-9]*$")
    text: str = Field(min_length=1)
    importance: ClaimImportance
    evidence: list[EvidenceSpan] = Field(default_factory=list)

    @model_validator(mode="after")
    def required_claim_has_entailing_evidence(self) -> "GoldClaim":
        if self.importance is ClaimImportance.REQUIRED:
            if not self.evidence:
                raise ValueError("Required claims need evidence")
            if not any(
                item.support in {SupportLevel.DIRECT, SupportLevel.DERIVED}
                for item in self.evidence
            ):
                raise ValueError("Required claims need direct or derived evidence")
        return self


class SourceRecord(StrictModel):
    source_id: str = Field(min_length=1)
    pmid: str | None = None
    pmcid: str | None = None
    title: str = Field(min_length=1)
    year: int | None = Field(default=None, ge=1800, le=2200)
    publication_type: str = Field(min_length=1)
    study_design: str = Field(min_length=1)
    role: str = Field(min_length=1)
    limitations: list[str] = Field(default_factory=list)


class QuestionRubric(StrictModel):
    complete_if: list[str] = Field(min_length=1)
    partial_if: list[str] = Field(default_factory=list)
    fail_if: list[str] = Field(min_length=1)


class QuestionReview(StrictModel):
    evidence_checked: bool = False
    ambiguity_checked: bool = False
    answer_checked: bool = False
    status: ReviewStatus = ReviewStatus.DRAFT


class BenchmarkQuestion(StrictModel):
    id: str = Field(pattern=r"^VMG-[0-9]{3}$")
    version: str = Field(pattern=r"^[0-9]+\.[0-9]+\.[0-9]+$")
    split: BenchmarkSplit
    task_type: TaskType
    answerability: Answerability
    category: str = Field(min_length=1)
    difficulty_reason: str = Field(min_length=1)
    question: str = Field(min_length=1)
    gold_claims: list[GoldClaim] = Field(default_factory=list)
    sources: list[SourceRecord] = Field(default_factory=list)
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    hard_negative_chunk_ids: list[str] = Field(default_factory=list)
    forbidden_claims: list[str] = Field(default_factory=list)
    missing_evidence: list[str] = Field(default_factory=list)
    rubric: QuestionRubric
    review: QuestionReview

    @model_validator(mode="after")
    def validate_question_contract(self) -> "BenchmarkQuestion":
        claim_ids = [claim.claim_id for claim in self.gold_claims]
        if len(claim_ids) != len(set(claim_ids)):
            raise ValueError("Duplicate claim_id")

        required = [
            claim for claim in self.gold_claims
            if claim.importance is ClaimImportance.REQUIRED
        ]
        if self.answerability is Answerability.COMPLETE:
            if not required:
                raise ValueError("Complete questions need a required claim")
            if self.missing_evidence:
                raise ValueError("Complete questions cannot declare missing_evidence")
        else:
            if not self.missing_evidence:
                raise ValueError(
                    "partial and unanswerable questions require missing_evidence"
                )
        if self.answerability is Answerability.UNANSWERABLE and required:
            raise ValueError("Unanswerable questions cannot contain required claims")

        repeated_lists = {
            "supporting_chunk_ids": self.supporting_chunk_ids,
            "hard_negative_chunk_ids": self.hard_negative_chunk_ids,
        }
        for name, values in repeated_lists.items():
            if len(values) != len(set(values)):
                raise ValueError(f"Duplicate values in {name}")
        return self


class ReviewEvent(StrictModel):
    question_id: str = Field(pattern=r"^VMG-[0-9]{3}$")
    from_status: ReviewStatus
    to_status: ReviewStatus
    actor: str = Field(min_length=1)
    created_at: datetime
    model_tag: str | None = None
    prompt_version: str | None = None
    findings: list[str] = Field(default_factory=list)
    decision: str = Field(min_length=1)
    resolution: str = Field(min_length=1)
    raw_output_sha256: str | None = Field(
        default=None, pattern=r"^[a-f0-9]{64}$"
    )


__all__ = [
    "Answerability",
    "BenchmarkQuestion",
    "BenchmarkSplit",
    "ClaimImportance",
    "EvidenceSpan",
    "GoldClaim",
    "QuestionReview",
    "QuestionRubric",
    "ReviewEvent",
    "ReviewStatus",
    "SourceRecord",
    "SupportLevel",
    "TaskType",
]
