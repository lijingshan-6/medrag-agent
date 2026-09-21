"""Profile the benchmark and run a source-only blind challenge for every item."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medrag.benchmark.audit import (
    AuditDecision,
    BlindAuditReview,
    benchmark_profile,
    build_blind_audit_prompt,
    normalize_blind_review,
    validate_audit_decisions,
)
from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.review import ollama_generate_json
from medrag.benchmark.schema import BenchmarkQuestion


def _read_questions(path: Path) -> list[BenchmarkQuestion]:
    with path.open(encoding="utf-8") as handle:
        return [
            BenchmarkQuestion.model_validate_json(line)
            for line in handle
            if line.strip()
        ]


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_summary(
    path: Path,
    *,
    questions: list[BenchmarkQuestion],
    review_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
) -> None:
    decisions = [AuditDecision.model_validate(row) for row in decision_rows]
    review_counts = Counter(row["review"]["decision"] for row in review_rows)
    decision_counts = Counter(row["decision"] for row in decision_rows)
    payload = {
        "schema_version": "1.0",
        "profile": benchmark_profile(questions),
        "blind_reviews": {
            "count": len(review_rows),
            "decisions": dict(sorted(review_counts.items())),
        },
        "curator_decisions": {
            "count": len(decision_rows),
            "decisions": dict(sorted(decision_counts.items())),
            "validation_errors": validate_audit_decisions(questions, decisions),
        },
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    questions = _read_questions(root / args.questions)
    output_dir = root / args.output_dir
    reviews_path = output_dir / "blind_audit_reviews.jsonl"
    decisions_path = output_dir / "quality_audit.jsonl"
    summary_path = output_dir / "quality_summary.json"
    review_rows = _read_jsonl(reviews_path)
    existing = {row["question_id"]: row for row in review_rows}

    if not args.profile_only:
        chunks = {
            row["chunk_id"]: row["text"]
            for row in build_normalized_chunks(
                root / "data/raw/pubmed/abstracts.jsonl",
                root / "data/raw/pmc/full_texts.jsonl",
            )
        }
        selected = questions[: args.limit] if args.limit else questions
        for offset, question in enumerate(selected, start=1):
            if args.resume and question.id in existing:
                print(f"[resume] {offset}/{len(selected)} {question.id}", flush=True)
                continue
            prompt = build_blind_audit_prompt(question, chunks)
            result = ollama_generate_json(
                model=args.model,
                prompt=prompt,
                response_model=BlindAuditReview,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
            )
            raw_review = BlindAuditReview.model_validate(result.value)
            if raw_review.question_id != question.id:
                raise ValueError(
                    f"review ID mismatch: expected {question.id}, got {raw_review.question_id}"
                )
            review = normalize_blind_review(question, raw_review)
            existing[question.id] = {
                "question_id": question.id,
                "model": args.model,
                "prompt_version": "blind-gold-audit-v1",
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
                "raw_output_sha256": result.raw_output_sha256,
                "raw_review": raw_review.model_dump(mode="json"),
                "review": review.model_dump(mode="json"),
            }
            review_rows = [existing[item.id] for item in questions if item.id in existing]
            _write_jsonl(reviews_path, review_rows)
            print(
                f"[audit] {offset}/{len(selected)} {question.id} {review.decision}",
                flush=True,
            )

    review_rows = [existing[item.id] for item in questions if item.id in existing]
    decision_rows = _read_jsonl(decisions_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_summary(
        summary_path,
        questions=questions,
        review_rows=review_rows,
        decision_rows=decision_rows,
    )
    print(json.dumps({"summary": str(summary_path), "reviews": len(review_rows)}))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/questions.jsonl"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1_1"),
    )
    parser.add_argument("--model", default="llama3.1:8b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--profile-only", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
