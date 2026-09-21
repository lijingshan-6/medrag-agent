"""Freeze the audited, source-disjoint VeritasMed benchmark v1.1."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medrag.benchmark.audit import (
    AuditDecision,
    QuestionRevision,
    benchmark_profile,
    freeze_audited_questions,
    validate_audit_decisions,
)
from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.validation import validate_questions


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_questions(path: Path, questions: list[BenchmarkQuestion]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for question in questions:
            handle.write(question.model_dump_json() + "\n")


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    parent_dir = root / args.parent_dir
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    questions = [
        BenchmarkQuestion.model_validate(row)
        for row in _read_jsonl(parent_dir / "questions.jsonl")
    ]
    decisions = [
        AuditDecision.model_validate(row)
        for row in _read_jsonl(output_dir / "quality_audit.jsonl")
    ]
    revisions = [
        QuestionRevision.model_validate(row)
        for row in _read_jsonl(output_dir / "revisions.jsonl")
    ]
    split_manifest = json.loads(
        (output_dir / "split_manifest.json").read_text(encoding="utf-8")
    )
    source_designs = json.loads(
        (output_dir / "source_designs.json").read_text(encoding="utf-8")
    )
    source_ids = {
        source.source_id for question in questions for source in question.sources
    }
    missing_designs = source_ids - set(source_designs)
    extra_designs = set(source_designs) - source_ids
    if missing_designs or extra_designs:
        raise ValueError(
            f"source-design coverage mismatch; missing={sorted(missing_designs)}, "
            f"extra={sorted(extra_designs)}"
        )
    audit_errors = validate_audit_decisions(questions, decisions)
    if audit_errors:
        raise ValueError("; ".join(audit_errors))

    frozen = freeze_audited_questions(
        questions,
        decisions,
        revisions,
        development_ids=set(split_manifest["development_ids"]),
        source_designs=source_designs,
    )
    normalized = build_normalized_chunks(
        root / "data/raw/pubmed/abstracts.jsonl",
        root / "data/raw/pmc/full_texts.jsonl",
    )
    chunk_text = {row["chunk_id"]: row["text"] for row in normalized}
    evidence_errors = validate_questions(frozen, chunk_text)
    if evidence_errors:
        raise ValueError("; ".join(evidence_errors))

    questions_path = output_dir / "questions.jsonl"
    _write_questions(questions_path, frozen)
    profile = benchmark_profile(frozen)
    if profile["source_overlap_ids"] or profile["evidence_chunk_overlap_ids"]:
        raise ValueError("the frozen split is not source and chunk disjoint")

    parent_manifest = json.loads(
        (parent_dir / "dataset_manifest.json").read_text(encoding="utf-8")
    )
    corpus_manifest = json.loads(
        (parent_dir / "manifest.json").read_text(encoding="utf-8")
    )
    task_counts = Counter(question.task_type.value for question in frozen)
    answerability_counts = Counter(question.answerability.value for question in frozen)
    split_counts = Counter(question.split.value for question in frozen)
    domains = Counter(question.category for question in frozen)
    keep_count = sum(item.decision == "keep" for item in decisions)
    revision_count = sum(item.decision == "revise" for item in decisions)
    artifact_paths = {
        "quality_audit": output_dir / "quality_audit.jsonl",
        "revisions": output_dir / "revisions.jsonl",
        "model_adjudications": output_dir / "model_adjudications.jsonl",
        "llama_blind_audit": output_dir / "blind_audit_reviews.jsonl",
        "qwen35_final_audit": (
            output_dir / "qwen35_final_audit" / "blind_audit_reviews.jsonl"
        ),
        "model_manifest": output_dir / "model_manifest.json",
    }
    missing_artifacts = [
        name for name, path in artifact_paths.items() if not path.exists()
    ]
    if missing_artifacts:
        raise ValueError(
            "missing v1.1 provenance artifacts: " + ", ".join(missing_artifacts)
        )
    manifest = {
        "schema_version": "1.1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "parent_questions_sha256": parent_manifest["questions_sha256"],
        "corpus_snapshot_id": corpus_manifest["snapshot_id"],
        "questions_path": str(questions_path.relative_to(root)).replace("\\", "/"),
        "questions_sha256": _sha256(questions_path),
        "question_count": len(frozen),
        "development_count": split_counts["development"],
        "test_count": split_counts["test"],
        "source_count": len(source_ids),
        "normalized_chunk_count": len(normalized),
        "curator_keep_count": keep_count,
        "curator_revision_count": revision_count,
        "source_overlap_count": 0,
        "evidence_chunk_overlap_count": 0,
        "clinician_reviewed": False,
        "artifact_sha256": {
            name: _sha256(path) for name, path in artifact_paths.items()
        },
    }
    (output_dir / "dataset_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    shutil.copyfile(parent_dir / "manifest.json", output_dir / "corpus_manifest.json")

    card = f"""# VeritasMed Evidence-Grounded Benchmark v1.1

This is a source-disjoint engineering benchmark for evidence-grounded medical-literature RAG. It is not clinician-reviewed and must not be used as a clinical-safety claim.

## Why v1.1 exists

The v1.0 audit found source leakage across development and test, generic or contradictory study-design metadata, gold-contract defects, and unrelated hard negatives on answerable items. v1.1 preserves v1.0, repairs those defects, and assigns complete source-connected groups to one split.

## Frozen composition

- Corpus snapshot: `{manifest['corpus_snapshot_id']}`
- Question SHA-256: `{manifest['questions_sha256']}`
- Questions: {len(frozen)} ({split_counts['development']} development, {split_counts['test']} test)
- Unique sources: {len(source_ids)}
- Source overlap between splits: 0
- Evidence-chunk overlap between splits: 0
- Task types: {dict(sorted(task_counts.items()))}
- Answerability: {dict(sorted(answerability_counts.items()))}
- Domains: {dict(sorted(domains.items()))}
- Source years: 2026 only

## Audit result

All 50 items received a source-first curator decision. {keep_count} were retained and {revision_count} gold contracts were revised. Exact evidence validation is run against all {len(normalized):,} normalized chunks. Answerable items no longer carry unrelated cross-domain hard negatives; the five abstention items retain same-topic, non-answering passages.

`qwen3.5:9b` generated a fresh 80-item candidate pool and independently re-audited the final 50. `llama3.1:8b` served only as an independent-family blind challenger. Their raw decisions and hashes are stored in the audit artifacts; source-first curator decisions in `quality_audit.jsonl` remain authoritative. `model_adjudications.jsonl` records how every model challenge was resolved.

## Intended use and limits

v1.1 is suitable for engineering iteration on this frozen corpus: retrieval coverage, evidence-bounded answering, citation support, and calibrated abstention. It is not representative of all medical literature because the local snapshot contains only 2026 records, most evidence is abstract-level, and no clinician independently adjudicated the labels. Public test labels deter casual development leakage but cannot provide a secret leaderboard.
"""
    (output_dir / "dataset_card.md").write_text(card, encoding="utf-8", newline="\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--parent-dir",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1_1"),
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
