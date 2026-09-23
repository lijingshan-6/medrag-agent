"""Recompute saved Agent metrics without a model, index, or raw medical corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from statistics import fmean
from typing import Any

from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.scoring import score_answer, score_retrieval, visible_citation_for_chunk_id


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _unique_by_id(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    result = {row["id"]: row for row in rows}
    if len(result) != len(rows):
        raise ValueError("Duplicate question IDs in saved artifact")
    return result


def recompute_saved_run(
    questions_path: Path, answers_path: Path, assessments_path: Path
) -> dict[str, Any]:
    """Apply the unchanged scorer to answer-hash-bound source-first assessments.

    Recalculation does not re-adjudicate semantic correctness. The assessment file
    records that decision and must match the exact saved answers being scored.
    """

    raw = json.loads(answers_path.read_text(encoding="utf-8"))
    question_hash = sha256_file(questions_path)
    if raw["questions_sha256"] != question_hash:
        raise ValueError("Saved run does not match the frozen question file")
    split = raw["split"]
    questions = [
        BenchmarkQuestion.model_validate_json(line)
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    questions = [q for q in questions if split == "all" or q.split.value == split]
    answers = _unique_by_id(raw["results"])
    assessments = _unique_by_id([
        json.loads(line)
        for line in assessments_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ])
    expected = {q.id for q in questions}
    if set(answers) != expected or set(assessments) != expected:
        raise ValueError("Questions, saved answers and assessments must cover exactly the same split")

    results = []
    for question in questions:
        answer = answers[question.id]
        review = assessments[question.id]
        digest = hashlib.sha256(answer["answer"].encode("utf-8")).hexdigest()
        if answer["answer_sha256"] != digest or review["answer_sha256"] != digest:
            raise ValueError(f"{question.id}: stale assessment or changed answer text")
        if answer["query"] != question.question:
            raise ValueError(f"{question.id}: saved query differs from the frozen question")
        assessment = review["assessment"]
        mappings = {}
        allowed = {c.claim_id: {e.chunk_id for e in c.evidence} for c in question.gold_claims}
        for mapping in assessment["claim_mappings"]:
            claim_id = mapping["claim_id"]
            if claim_id in mappings or claim_id not in allowed:
                raise ValueError(f"{question.id}: duplicate or unknown claim mapping")
            mapped_chunks = mapping["citation_chunk_ids"]
            for chunk_id in mapped_chunks:
                citation = visible_citation_for_chunk_id(chunk_id)
                if (
                    chunk_id not in answer["retrieved_chunk_ids"]
                    or chunk_id not in allowed[claim_id]
                    or f"[{citation}]" not in answer["answer"]
                ):
                    raise ValueError(f"{question.id}: mapped evidence is not retrieved and cited gold evidence")
            mappings[claim_id] = mapped_chunks
        metrics = score_answer(
            question,
            claim_mappings=mappings,
            abstained=assessment["abstained"],
            boundary_acknowledged=assessment["boundary_acknowledged"],
            forbidden_claims_present=assessment["forbidden_claims_present"],
            unsupported_material_claim_count=assessment["unsupported_material_claim_count"],
            missing_required_qualifier_count=assessment["missing_required_qualifier_count"],
        ).as_dict()
        results.append({
            "id": question.id,
            "answerability": question.answerability.value,
            "task_type": question.task_type.value,
            "question": question.question,
            "answer": answer["answer"],
            "answer_sha256": digest,
            "retrieved_chunk_ids": answer["retrieved_chunk_ids"],
            "agent_faithful": answer["faithful"],
            "latency_seconds": answer["latency_seconds"],
            "assessment": assessment,
            "resolution": review["curator_resolution"],
            "metrics": metrics,
            "retrieval_metrics": score_retrieval(question, answer["retrieved_chunk_ids"]).as_dict(),
        })

    answerable = [r for r in results if r["answerability"] != "unanswerable"]
    answer_summary = {
        "questions": len(results),
        **{name: fmean(float(r["metrics"][name]) for r in results) for name in results[0]["metrics"]},
    }
    retrieval_summary = {
        "questions": len(answerable),
        **{
            name: fmean(float(r["retrieval_metrics"][name]) for r in answerable)
            for name in results[0]["retrieval_metrics"]
        },
    }
    return {
        "schema_version": "1.0",
        "split": split,
        "answer_source": raw["answer_source"],
        "answer_model": raw["model"],
        "assessment_method": "Codex source-first adjudication; no independent clinician review",
        "code_commit": raw["code_commit"],
        "working_tree_dirty_at_run_start": raw["working_tree_dirty"],
        "runtime_config": raw["runtime_config"],
        "input_sha256": {
            "questions": question_hash,
            "answers": sha256_file(answers_path),
            "assessments": sha256_file(assessments_path),
        },
        "summary": {
            "answers": answer_summary,
            "retrieval_answerable_at_5": retrieval_summary,
            "runtime": raw["summary"],
        },
        "results": results,
    }


def render_cases(report: dict[str, Any], questions_path: Path, *, version: str = "v0.3") -> str:
    """Render the saved answers beside their frozen evidence and adjudications."""

    questions = {
        row["id"]: row
        for line in questions_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    lines = [
        f"# {version} {report['split']} answers and evidence",
        "",
        f"These are real production-Agent outputs, not guided-demo fixtures. All {len(report['results'])} answers",
        f"come from one {report['split']} run at `{report['code_commit'][:7]}` using `{report['answer_model']}`.",
        "The decisions below are Codex source-first adjudications, not independent clinician reviews.",
        "Recalculation checks the saved mappings and arithmetic; it does not independently verify their semantics.",
        "",
        f"[Results and limitations](agent-{version}-report.md) · [Demo guide](demo.md)",
        "",
        "| Question | Strict pass | Agent self-check | Missing qualifiers | Unsupported claims |",
        "|---|---|---|---:|---:|",
    ]
    for row in report["results"]:
        metrics = row["metrics"]
        lines.append(
            f"| [{row['id']}](#{row['id'].lower()}) | {'Yes' if metrics['strict_pass'] else 'No'} "
            f"| {'Pass' if row['agent_faithful'] else 'Fail'} "
            f"| {metrics['missing_required_qualifier_count']} | {metrics['unsupported_material_claim_count']} |"
        )
    for row in report["results"]:
        question = questions[row["id"]]
        lines.extend(["", f"## {row['id']}", "", row["question"], "", "**Saved Agent answer**", "", row["answer"], ""])
        lines.extend(["**Source-first decision**", "", row["resolution"], ""])
        lines.extend(["**Frozen claims and evidence**", ""])
        for claim in question["gold_claims"]:
            lines.extend([f"- **{claim['claim_id']}**: {claim['text']}", ""])
            for evidence in claim["evidence"]:
                lines.extend([f"  Evidence `{evidence['chunk_id']}`:", "", f"  > {evidence['quote']}", ""])
        if question["missing_evidence"]:
            lines.extend(["Required evidence boundary: " + " ".join(question["missing_evidence"]), ""])
        for source in question["sources"]:
            if source.get("pmid"):
                lines.extend([f"Source: [{source['title']}](https://pubmed.ncbi.nlm.nih.gov/{source['pmid']}/)", ""])
        lines.extend([
            "Retrieved chunks: " + ", ".join(f"`{cid}`" for cid in row["retrieved_chunk_ids"]),
            "", f"Answer SHA-256: `{row['answer_sha256']}`", "",
        ])
    return "\n".join(lines).rstrip() + "\n"
