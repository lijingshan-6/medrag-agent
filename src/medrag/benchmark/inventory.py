"""Build an auditable corpus inventory for the benchmark snapshot."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from medrag.ingest.chunker import chunk_pmc_record, chunk_pubmed_record


def sha256_file(path: Path) -> str:
    """Return the SHA-256 digest of a file without loading it all into memory."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _pmc_identity(record: dict[str, Any], row_index: int) -> tuple[str, list[str]]:
    pmcid = str(record.get("pmcid") or "").strip()
    if pmcid:
        return pmcid, []
    return f"doc{row_index}", ["missing_pmcid", "synthetic_stable_id"]


def build_normalized_chunks(pubmed_path: Path, pmc_path: Path) -> list[dict[str, Any]]:
    """Rebuild chunks with deterministic identities for PMC rows lacking PMCID."""

    chunks: list[dict[str, Any]] = []
    for record in _load_jsonl(pubmed_path):
        chunks.extend(chunk.__dict__ for chunk in chunk_pubmed_record(record))
    for row_index, record in enumerate(_load_jsonl(pmc_path)):
        normalized = dict(record)
        normalized["pmcid"], _ = _pmc_identity(record, row_index)
        chunks.extend(chunk.__dict__ for chunk in chunk_pmc_record(normalized))
    return chunks


def build_document_inventory(pubmed_path: Path, pmc_path: Path) -> list[dict[str, Any]]:
    """Describe every source document and flag records needing curator attention."""

    pubmed_records = _load_jsonl(pubmed_path)
    pmc_records = _load_jsonl(pmc_path)
    pubmed_ids = Counter(str(row.get("pmid") or "").strip() for row in pubmed_records)
    pmc_ids = Counter(str(row.get("pmcid") or "").strip() for row in pmc_records)
    rows: list[dict[str, Any]] = []

    for row_index, record in enumerate(pubmed_records):
        pmid = str(record.get("pmid") or "").strip()
        flags: list[str] = []
        if not pmid:
            flags.append("missing_pmid")
        elif pubmed_ids[pmid] > 1:
            flags.append("duplicate_identifier")
        pub_types = [str(item) for item in record.get("pub_types", [])]
        if any(item.casefold() == "retracted publication" for item in pub_types):
            flags.append("retracted_publication")
        abstract = str(record.get("abstract") or "")
        if len(abstract.split()) < 80:
            flags.append("short_text")
        if "\ufffd" in abstract or "\ufffd" in str(record.get("title") or ""):
            flags.append("replacement_character")
        rows.append(
            {
                "document_id": f"pubmed:{pmid or f'row{row_index}'}",
                "source": "pubmed",
                "source_identifier": pmid or None,
                "raw_row_index": row_index,
                "title": record.get("title", ""),
                "year": record.get("year"),
                "word_count": len(abstract.split()),
                "chunk_count": len(chunk_pubmed_record(record)) if pmid else 0,
                "flags": sorted(flags),
            }
        )

    for row_index, record in enumerate(pmc_records):
        pmcid, identity_flags = _pmc_identity(record, row_index)
        original_pmcid = str(record.get("pmcid") or "").strip()
        flags = list(identity_flags)
        if original_pmcid and pmc_ids[original_pmcid] > 1:
            flags.append("duplicate_identifier")
        word_count = int(record.get("word_count") or len(str(record.get("full_text") or "").split()))
        if word_count < 1000:
            flags.append("short_text")
        full_text = str(record.get("full_text") or "")
        if "\ufffd" in full_text or "\ufffd" in str(record.get("title") or ""):
            flags.append("replacement_character")
        normalized = dict(record)
        normalized["pmcid"] = pmcid
        rows.append(
            {
                "document_id": f"pmc:{pmcid}",
                "source": "pmc",
                "source_identifier": original_pmcid or None,
                "raw_row_index": row_index,
                "title": record.get("title", ""),
                "year": record.get("year"),
                "word_count": word_count,
                "chunk_count": len(chunk_pmc_record(normalized)),
                "flags": sorted(flags),
            }
        )
    return rows


def _line_count(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for line in handle if line.strip())


def build_manifest(root: Path, relative_paths: list[str]) -> dict[str, Any]:
    """Freeze input file hashes and row counts under a reproducible snapshot ID."""

    inputs: list[dict[str, Any]] = []
    for relative_path in relative_paths:
        path = root / relative_path
        inputs.append(
            {
                "path": relative_path.replace("\\", "/"),
                "sha256": sha256_file(path),
                "size_bytes": path.stat().st_size,
                "row_count": _line_count(path),
            }
        )
    snapshot_seed = "\n".join(
        f"{item['path']}:{item['sha256']}" for item in inputs
    ).encode("utf-8")
    return {
        "schema_version": "1.0",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "snapshot_id": hashlib.sha256(snapshot_seed).hexdigest()[:16],
        "inputs": inputs,
    }


def verify_manifest(manifest: dict[str, Any], root: Path) -> list[str]:
    """Return every missing or changed input instead of failing at the first drift."""

    errors: list[str] = []
    for expected in manifest.get("inputs", []):
        relative_path = expected["path"]
        path = root / relative_path
        if not path.exists():
            errors.append(f"{relative_path}: file is missing")
            continue
        actual_hash = sha256_file(path)
        if actual_hash != expected["sha256"]:
            errors.append(
                f"{relative_path}: SHA-256 changed "
                f"({expected['sha256']} -> {actual_hash})"
            )
        actual_rows = _line_count(path)
        if actual_rows != expected["row_count"]:
            errors.append(
                f"{relative_path}: row count changed "
                f"({expected['row_count']} -> {actual_rows})"
            )
    return errors


__all__ = [
    "build_document_inventory",
    "build_manifest",
    "build_normalized_chunks",
    "sha256_file",
    "verify_manifest",
]
