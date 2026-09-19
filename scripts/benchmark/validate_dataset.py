"""Validate benchmark JSONL rows against their frozen chunk snapshot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import ValidationError

from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.validation import validate_questions


def _read_chunks(path: Path) -> tuple[dict[str, str], list[str]]:
    chunks: dict[str, str] = {}
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            errors.append(f"{path}:{line_number}: invalid JSON: {exc.msg}")
            continue
        chunk_id = str(row.get("chunk_id", "")).strip()
        text = str(row.get("text", ""))
        if not chunk_id:
            errors.append(f"{path}:{line_number}: missing chunk_id")
        elif chunk_id in chunks:
            errors.append(f"{path}:{line_number}: duplicate chunk_id {chunk_id}")
        else:
            chunks[chunk_id] = text
    return chunks, errors


def _read_questions(path: Path) -> tuple[list[BenchmarkQuestion], list[str]]:
    questions: list[BenchmarkQuestion] = []
    errors: list[str] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            questions.append(BenchmarkQuestion.model_validate_json(line))
        except (ValidationError, ValueError) as exc:
            errors.append(f"{path}:{line_number}: {exc}")
    return questions, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--questions", type=Path, required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--chunks",
        type=Path,
        help="Already-normalized chunk JSONL. Legacy duplicate PMC IDs will fail.",
    )
    source.add_argument(
        "--normalized-corpus-root",
        type=Path,
        help="Repository root containing data/raw/pubmed and data/raw/pmc.",
    )
    args = parser.parse_args()

    if args.normalized_corpus_root is not None:
        root = args.normalized_corpus_root
        normalized = build_normalized_chunks(
            root / "data/raw/pubmed/abstracts.jsonl",
            root / "data/raw/pmc/full_texts.jsonl",
        )
        chunks = {row["chunk_id"]: row["text"] for row in normalized}
        errors = []
    else:
        chunks, errors = _read_chunks(args.chunks)
    questions, question_errors = _read_questions(args.questions)
    errors.extend(question_errors)
    errors.extend(validate_questions(questions, chunks))

    split_counts: dict[str, int] = {}
    for question in questions:
        split_counts[question.split.value] = split_counts.get(question.split.value, 0) + 1

    print(
        json.dumps(
            {
                "questions": len(questions),
                "chunks": len(chunks),
                "splits": split_counts,
                "errors": len(errors),
            },
            ensure_ascii=False,
        )
    )
    for error in errors:
        print(f"ERROR: {error}")
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
