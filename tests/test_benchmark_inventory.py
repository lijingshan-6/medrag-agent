from __future__ import annotations

import hashlib
import json
from pathlib import Path

from medrag.benchmark.inventory import (
    build_document_inventory,
    build_manifest,
    build_normalized_chunks,
    sha256_file,
    verify_manifest,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _pubmed(pmid: str, *, pub_types: list[str] | None = None) -> dict:
    return {
        "pmid": pmid,
        "title": f"Study {pmid}",
        "abstract": "word " * 100,
        "authors": ["Example A"],
        "journal": "Journal",
        "year": 2024,
        "pub_types": pub_types or ["Journal Article"],
        "mesh_terms": [],
        "language": "eng",
    }


def _pmc(title: str, *, pmcid: str = "", words: int = 1100) -> dict:
    text = "word " * words
    return {
        "pmcid": pmcid,
        "pmid": None,
        "title": title,
        "full_text": text,
        "sections": [{"name": "RESULTS", "text": text}],
        "word_count": words,
    }


def test_sha256_file_is_stable(tmp_path: Path) -> None:
    path = tmp_path / "source.jsonl"
    path.write_bytes(b"one\ntwo\n")

    assert sha256_file(path) == hashlib.sha256(b"one\ntwo\n").hexdigest()


def test_inventory_reports_duplicates_retractions_and_missing_ids(tmp_path: Path) -> None:
    pubmed_path = tmp_path / "pubmed.jsonl"
    pmc_path = tmp_path / "pmc.jsonl"
    _write_jsonl(
        pubmed_path,
        [
            _pubmed("1", pub_types=["Retracted Publication"]),
            _pubmed("1"),
        ],
    )
    _write_jsonl(pmc_path, [_pmc("Missing identifier", words=20)])

    rows = build_document_inventory(pubmed_path, pmc_path)

    first, second, pmc = rows
    assert "duplicate_identifier" in first["flags"]
    assert "retracted_publication" in first["flags"]
    assert "duplicate_identifier" in second["flags"]
    assert pmc["document_id"] == "pmc:doc0"
    assert "missing_pmcid" in pmc["flags"]
    assert "synthetic_stable_id" in pmc["flags"]
    assert "short_text" in pmc["flags"]


def test_normalized_chunks_assign_unique_deterministic_pmc_ids(tmp_path: Path) -> None:
    pubmed_path = tmp_path / "pubmed.jsonl"
    pmc_path = tmp_path / "pmc.jsonl"
    _write_jsonl(pubmed_path, [])
    _write_jsonl(pmc_path, [_pmc("First"), _pmc("Second")])

    chunks = build_normalized_chunks(pubmed_path, pmc_path)

    assert chunks[0]["chunk_id"] == "pmc:doc0:0"
    assert any(chunk["chunk_id"] == "pmc:doc1:0" for chunk in chunks)
    assert len({chunk["chunk_id"] for chunk in chunks}) == len(chunks)


def test_manifest_drift_names_changed_path(tmp_path: Path) -> None:
    pubmed_path = tmp_path / "data/raw/pubmed/abstracts.jsonl"
    pmc_path = tmp_path / "data/raw/pmc/full_texts.jsonl"
    chunks_path = tmp_path / "data/index_cache/chunks.jsonl"
    _write_jsonl(pubmed_path, [_pubmed("1")])
    _write_jsonl(pmc_path, [_pmc("Full text")])
    _write_jsonl(
        chunks_path,
        [{"chunk_id": "pubmed:1:0", "source": "pubmed", "doc_id": "1", "text": "x"}],
    )
    manifest = build_manifest(
        tmp_path,
        [
            "data/raw/pubmed/abstracts.jsonl",
            "data/raw/pmc/full_texts.jsonl",
            "data/index_cache/chunks.jsonl",
        ],
    )

    pubmed_path.write_text("changed\n", encoding="utf-8")

    errors = verify_manifest(manifest, tmp_path)

    assert any("data/raw/pubmed/abstracts.jsonl" in error for error in errors)
    assert not any("data/raw/pmc/full_texts.jsonl" in error for error in errors)
