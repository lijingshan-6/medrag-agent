from __future__ import annotations

from medrag.benchmark.audit import (
    AuditDecision,
    BlindAuditReview,
    QuestionRevision,
    benchmark_profile,
    build_blind_audit_prompt,
    freeze_audited_questions,
    normalize_blind_review,
    source_connected_groups,
    validate_source_disjoint_split,
    validate_audit_decisions,
)
from medrag.benchmark.schema import BenchmarkQuestion


def _question(
    question_id: str,
    *,
    split: str,
    source_id: str,
    chunk_id: str,
    question: str,
    year: int = 2026,
) -> BenchmarkQuestion:
    return BenchmarkQuestion.model_validate(
        {
            "id": question_id,
            "version": "1.0.0",
            "split": split,
            "task_type": "single_evidence",
            "answerability": "complete",
            "category": "fixture",
            "difficulty_reason": "Audit fixture.",
            "question": question,
            "gold_claims": [
                {
                    "claim_id": "C1",
                    "text": f"Claim for {question_id}.",
                    "importance": "required",
                    "evidence": [
                        {"chunk_id": chunk_id, "quote": "Exact quote.", "support": "direct"}
                    ],
                }
            ],
            "sources": [
                {
                    "source_id": source_id,
                    "pmid": source_id.removeprefix("PMID:"),
                    "pmcid": None,
                    "title": "Fixture source",
                    "year": year,
                    "publication_type": "Journal Article",
                    "study_design": "cohort study",
                    "role": "primary_study",
                    "limitations": [],
                }
            ],
            "supporting_chunk_ids": [chunk_id],
            "hard_negative_chunk_ids": [],
            "forbidden_claims": [],
            "missing_evidence": [],
            "rubric": {"complete_if": ["C1"], "partial_if": [], "fail_if": ["Missing C1"]},
            "review": {
                "evidence_checked": True,
                "ambiguity_checked": True,
                "answer_checked": True,
                "status": "frozen",
            },
        }
    )


def test_profile_exposes_split_leakage_and_content_shape() -> None:
    questions = [
        _question("VMG-001", split="development", source_id="PMID:1", chunk_id="pubmed:1:0", question="What was found?"),
        _question("VMG-002", split="test", source_id="PMID:1", chunk_id="pubmed:1:0", question="What was found?"),
    ]

    profile = benchmark_profile(questions)

    assert profile["source_overlap_ids"] == ["PMID:1"]
    assert profile["evidence_chunk_overlap_ids"] == ["pubmed:1:0"]
    assert profile["duplicate_question_count"] == 1
    assert profile["source_years"] == {"2026": 2}


def test_complete_item_audits_are_required_for_every_question() -> None:
    questions = [
        _question("VMG-001", split="development", source_id="PMID:1", chunk_id="pubmed:1:0", question="Question one?"),
        _question("VMG-002", split="test", source_id="PMID:2", chunk_id="pubmed:2:0", question="Question two?"),
    ]
    decisions = [
        AuditDecision(
            question_id="VMG-001",
            decision="keep",
            evidence_entailment="pass",
            qualifier_integrity="pass",
            ambiguity="pass",
            answerability="pass",
            hard_negative="not_applicable",
            rubric="pass",
            reviewer_findings=[],
            curator_resolution="All required checks passed against the source text.",
        )
    ]

    assert validate_audit_decisions(questions, decisions) == [
        "missing audit decision: VMG-002"
    ]


def test_keep_decision_cannot_hide_a_failed_dimension() -> None:
    question = _question("VMG-001", split="development", source_id="PMID:1", chunk_id="pubmed:1:0", question="Question one?")
    decision = AuditDecision(
        question_id="VMG-001",
        decision="keep",
        evidence_entailment="pass",
        qualifier_integrity="fail",
        ambiguity="pass",
        answerability="pass",
        hard_negative="not_applicable",
        rubric="pass",
        reviewer_findings=["A time-point qualifier is missing."],
        curator_resolution="Needs a qualifier revision.",
    )

    assert validate_audit_decisions([question], [decision]) == [
        "VMG-001: keep requires every applicable dimension to pass"
    ]


def test_blind_prompt_contains_gold_contract_but_not_prior_outcomes() -> None:
    question = _question("VMG-001", split="development", source_id="PMID:1", chunk_id="pubmed:1:0", question="What was found?")

    prompt = build_blind_audit_prompt(question, {"pubmed:1:0": "Full frozen passage."})

    assert "What was found?" in prompt
    assert "Claim for VMG-001." in prompt
    assert "Exact quote." in prompt
    assert "Full frozen passage." in prompt
    assert "retrieval rank" not in prompt.casefold()
    assert "agent answer" not in prompt.casefold()


def test_blind_audit_schema_bounds_each_finding() -> None:
    schema = BlindAuditReview.model_json_schema()

    assert schema["properties"]["findings"]["items"]["maxLength"] == 240
    assert schema["properties"]["rationale"]["maxLength"] == 600


def test_blind_audit_normalizes_an_inconsistent_model_decision() -> None:
    question = _question(
        "VMG-001",
        split="development",
        source_id="PMID:1",
        chunk_id="pubmed:1:0",
        question="What was found?",
    )
    raw = BlindAuditReview(
        question_id="VMG-001",
        decision="pass",
        evidence_entailment="pass",
        qualifier_integrity="pass",
        ambiguity="fail",
        answerability="pass",
        hard_negative="fail",
        rubric="pass",
        findings=[],
        rationale="The wording has two plausible interpretations.",
    )

    normalized = normalize_blind_review(question, raw)

    assert normalized.decision == "revise"
    assert normalized.hard_negative == "not_applicable"
    assert normalized.findings == ["The wording has two plausible interpretations."]


def test_source_connected_questions_must_stay_in_one_split() -> None:
    questions = [
        _question("VMG-001", split="development", source_id="PMID:1", chunk_id="pubmed:1:0", question="One?"),
        _question("VMG-002", split="test", source_id="PMID:1", chunk_id="pubmed:1:1", question="Two?"),
        _question("VMG-003", split="test", source_id="PMID:2", chunk_id="pubmed:2:0", question="Three?"),
    ]

    groups = source_connected_groups(questions)

    assert groups == [["VMG-001", "VMG-002"], ["VMG-003"]]
    assert validate_source_disjoint_split(questions) == [
        "source-connected group crosses splits: VMG-001, VMG-002"
    ]


def test_freeze_applies_revision_and_clears_nonanswerable_only_negatives() -> None:
    question = _question("VMG-001", split="test", source_id="PMID:1", chunk_id="pubmed:1:0", question="Original question?")
    question = question.model_copy(update={"hard_negative_chunk_ids": ["pubmed:9:0"]})
    audit = AuditDecision(
        question_id="VMG-001",
        decision="revise",
        evidence_entailment="fail",
        qualifier_integrity="pass",
        ambiguity="pass",
        answerability="pass",
        hard_negative="not_applicable",
        rubric="pass",
        reviewer_findings=["The claim is too broad."],
        curator_resolution="The replacement claim is bounded to the quoted finding.",
    )
    revision = QuestionRevision(
        question_id="VMG-001",
        reason="Bound the claim to the reported measurement.",
        gold_claims=[
            {
                "claim_id": "C1",
                "text": "Revised supported claim.",
                "importance": "required",
                "evidence": [
                    {"chunk_id": "pubmed:1:0", "quote": "Exact quote.", "support": "direct"}
                ],
            }
        ],
    )

    frozen = freeze_audited_questions(
        [question],
        [audit],
        [revision],
        development_ids={"VMG-001"},
        development_count=1,
        source_designs={"PMID:1": "retrospective cohort"},
    )

    assert frozen[0].version == "1.1.0"
    assert frozen[0].split.value == "development"
    assert frozen[0].gold_claims[0].text == "Revised supported claim."
    assert frozen[0].hard_negative_chunk_ids == []
    assert frozen[0].sources[0].study_design == "retrospective cohort"
