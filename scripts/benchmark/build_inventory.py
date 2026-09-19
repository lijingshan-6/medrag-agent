"""Build the frozen corpus manifest and curator-facing document inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from medrag.benchmark.inventory import (
    build_document_inventory,
    build_manifest,
    build_normalized_chunks,
)

INPUTS = [
    "data/raw/pubmed/abstracts.jsonl",
    "data/raw/pmc/full_texts.jsonl",
    "data/index_cache/chunks.jsonl",
]


def _canonical_chunks_sha(chunks: list[dict[str, Any]]) -> str:
    digest = hashlib.sha256()
    for chunk in chunks:
        line = json.dumps(chunk, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        digest.update(line.encode("utf-8") + b"\n")
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def build(root: Path, output_dir: Path) -> dict[str, Any]:
    pubmed_path = root / INPUTS[0]
    pmc_path = root / INPUTS[1]
    cache_path = root / INPUTS[2]
    inventory = build_document_inventory(pubmed_path, pmc_path)
    normalized = build_normalized_chunks(pubmed_path, pmc_path)
    cached = _read_jsonl(cache_path)

    if len(normalized) != len(cached):
        raise SystemExit(
            f"normalized chunk count {len(normalized)} does not match cache {len(cached)}"
        )
    for row_index, (rebuilt, old) in enumerate(zip(normalized, cached, strict=True)):
        if rebuilt["text"] != old["text"] or rebuilt["source"] != old["source"]:
            raise SystemExit(f"chunk text/source alignment differs at row {row_index}")

    cached_counts = Counter(row["chunk_id"] for row in cached)
    normalized_ids = [row["chunk_id"] for row in normalized]
    if len(set(normalized_ids)) != len(normalized_ids):
        raise SystemExit("normalized chunk IDs are not unique")

    manifest = build_manifest(root, INPUTS)
    flag_counts = Counter(flag for row in inventory for flag in row["flags"])
    manifest.update(
        {
            "identity_policy": {
                "pubmed": "pubmed:{pmid}:{chunk_index}",
                "pmc_with_id": "pmc:{pmcid}:{chunk_index}",
                "pmc_missing_id": "pmc:doc{raw_row_index}:{chunk_index}",
                "stability_note": "raw input hashes freeze the row ordering used by synthetic PMC IDs",
            },
            "documents": {
                "total": len(inventory),
                "pubmed": sum(row["source"] == "pubmed" for row in inventory),
                "pmc": sum(row["source"] == "pmc" for row in inventory),
                "flag_counts": dict(sorted(flag_counts.items())),
            },
            "normalized_chunk_snapshot": {
                "row_count": len(normalized),
                "unique_chunk_ids": len(set(normalized_ids)),
                "canonical_sha256": _canonical_chunks_sha(normalized),
            },
            "legacy_chunk_cache": {
                "row_count": len(cached),
                "unique_chunk_ids": len(cached_counts),
                "duplicated_id_values": sum(count > 1 for count in cached_counts.values()),
                "identity_status": "superseded_for_benchmark_references",
                "text_order_matches_normalized_snapshot": True,
            },
        }
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output_dir / "corpus_inventory.jsonl", inventory)
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1"),
    )
    args = parser.parse_args()
    manifest = build(args.root.resolve(), args.output_dir)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
