"""Cross-row and source-aware benchmark validation."""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping

from medrag.benchmark.schema import BenchmarkQuestion


def _normalise_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def validate_questions(
    questions: Iterable[BenchmarkQuestion],
    chunks: Mapping[str, str],
) -> list[str]:
    """Return all duplicate, reference, and exact-quote validation failures."""

    errors: list[str] = []
    seen_question_ids: set[str] = set()

    for question in questions:
        if question.id in seen_question_ids:
            errors.append(f"{question.id}: duplicate question id")
        seen_question_ids.add(question.id)

        for claim in question.gold_claims:
            for evidence in claim.evidence:
                prefix = f"{question.id}/{claim.claim_id}/{evidence.chunk_id}"
                chunk_text = chunks.get(evidence.chunk_id)
                if chunk_text is None:
                    errors.append(
                        f"{prefix}: referenced chunk is missing from snapshot"
                    )
                    continue
                if _normalise_whitespace(evidence.quote) not in _normalise_whitespace(
                    chunk_text
                ):
                    errors.append(
                        f"{prefix}: evidence quote does not occur in referenced chunk"
                    )

        for label, chunk_ids in (
            ("supporting", question.supporting_chunk_ids),
            ("hard-negative", question.hard_negative_chunk_ids),
        ):
            for chunk_id in chunk_ids:
                if chunk_id not in chunks:
                    errors.append(
                        f"{question.id}/{label}/{chunk_id}: referenced chunk is "
                        "missing from snapshot"
                    )

    return errors


__all__ = ["validate_questions"]
