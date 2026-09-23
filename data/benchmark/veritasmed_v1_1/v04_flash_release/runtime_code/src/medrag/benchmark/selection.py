"""Deterministic freezing and composition checks for the curated benchmark."""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Set

from medrag.benchmark.schema import (
    BenchmarkQuestion,
    BenchmarkSplit,
    ReviewStatus,
)

TASK_TARGETS: Mapping[str, int] = {
    "single_evidence": 10,
    "within_document_synthesis": 15,
    "cross_document_comparison": 10,
    "limitations_and_safety": 10,
    "insufficient_evidence": 5,
}
ANSWERABILITY_TARGETS: Mapping[str, int] = {
    "complete": 40,
    "partial": 5,
    "unanswerable": 5,
}
SPLIT_TARGETS: Mapping[str, int] = {"development": 15, "test": 35}


def _count_error(label: str, actual: Counter[str], expected: Mapping[str, int]) -> str | None:
    normalized = {key: actual.get(key, 0) for key in expected}
    if normalized == dict(expected) and not (set(actual) - set(expected)):
        return None
    return f"{label} counts are {dict(sorted(actual.items()))}; expected {dict(expected)}"


def validate_composition(questions: Iterable[BenchmarkQuestion]) -> list[str]:
    """Return human-readable failures for the frozen 50-question contract."""

    rows = list(questions)
    errors: list[str] = []
    if len(rows) != 50:
        errors.append(f"question count is {len(rows)}; expected 50")

    counters = (
        (
            "task",
            Counter(row.task_type.value for row in rows),
            TASK_TARGETS,
        ),
        (
            "answerability",
            Counter(row.answerability.value for row in rows),
            ANSWERABILITY_TARGETS,
        ),
        (
            "split",
            Counter(row.split.value for row in rows),
            SPLIT_TARGETS,
        ),
    )
    for label, actual, expected in counters:
        error = _count_error(label, actual, expected)
        if error:
            errors.append(error)

    domains = Counter(row.category for row in rows)
    if len(domains) < 8:
        errors.append(f"benchmark has {len(domains)} domains; expected at least 8 domains")
    crowded = {domain: count for domain, count in domains.items() if count > 7}
    if crowded:
        errors.append(f"domain counts exceed the maximum 7: {crowded}")

    ids = [row.id for row in rows]
    if len(ids) != len(set(ids)):
        errors.append("benchmark contains duplicate question IDs")
    return errors


def freeze_questions(
    candidates: Iterable[BenchmarkQuestion],
    *,
    development_ids: Set[str],
) -> tuple[list[BenchmarkQuestion], dict[str, str]]:
    """Freeze an explicit curator selection and assign stable contiguous IDs.

    The input order is ignored. Candidate IDs are retained in the returned mapping so
    every final row remains traceable to the append-only candidate review ledger.
    """

    rows = sorted(candidates, key=lambda row: int(row.id.removeprefix("VMG-")))
    source_ids = {row.id for row in rows}
    unknown_development_ids = set(development_ids) - source_ids
    if unknown_development_ids:
        raise ValueError(
            "development IDs are not selected candidates: "
            + ", ".join(sorted(unknown_development_ids))
        )
    if len(development_ids) != 15:
        raise ValueError(f"development selection has {len(development_ids)} IDs; expected 15")

    ineligible = [
        row.id for row in rows if row.review.status is not ReviewStatus.FROZEN
    ]
    if ineligible:
        raise ValueError("selected candidates are not frozen: " + ", ".join(ineligible))

    frozen: list[BenchmarkQuestion] = []
    provenance: dict[str, str] = {}
    for number, row in enumerate(rows, start=1):
        final_id = f"VMG-{number:03d}"
        split = (
            BenchmarkSplit.DEVELOPMENT
            if row.id in development_ids
            else BenchmarkSplit.TEST
        )
        frozen.append(row.model_copy(update={"id": final_id, "split": split}))
        provenance[final_id] = row.id

    errors = validate_composition(frozen)
    if errors:
        raise ValueError("invalid benchmark composition:\n- " + "\n- ".join(errors))
    return frozen, provenance


__all__ = [
    "ANSWERABILITY_TARGETS",
    "SPLIT_TARGETS",
    "TASK_TARGETS",
    "freeze_questions",
    "validate_composition",
]
