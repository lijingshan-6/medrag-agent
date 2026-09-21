"""Build an isolated Qdrant index from the frozen normalized corpus cache."""

from __future__ import annotations

# Load PyTorch before Qdrant's native runtime on Windows.
import sentence_transformers  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
from qdrant_client import QdrantClient

from medrag.benchmark.inventory import build_normalized_chunks
from medrag.index.indexer import index_chunks
from medrag.index.qdrant_setup import create_collection
from medrag.ingest.chunker import Chunk


def _load_sparse(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def run(args: argparse.Namespace) -> None:
    root = args.root.resolve()
    runtime_path = root / args.runtime_path
    chunks_raw = build_normalized_chunks(
        root / "data/raw/pubmed/abstracts.jsonl",
        root / "data/raw/pmc/full_texts.jsonl",
    )
    chunks = [Chunk(**row) for row in chunks_raw]
    dense = np.load(root / "data/index_cache/dense.npy", mmap_mode="r")
    sparse = _load_sparse(root / "data/index_cache/sparse.jsonl")
    if len(chunks) != dense.shape[0] or len(chunks) != len(sparse):
        raise ValueError(
            f"frozen cache row mismatch: chunks={len(chunks)}, "
            f"dense={dense.shape[0]}, sparse={len(sparse)}"
        )

    runtime_path.mkdir(parents=True, exist_ok=True)
    client = QdrantClient(path=runtime_path, force_disable_check_same_thread=True)
    try:
        if client.collection_exists(args.collection) and not args.recreate:
            count = client.count(collection_name=args.collection).count
            if count == len(chunks):
                print(f"Ready: {args.collection} already contains {count} chunks.")
                return
        create_collection(client, args.collection, recreate=True)

        def progress(done: int, total: int) -> None:
            print(f"[index] {done}/{total}", flush=True)

        index_chunks(
            client,
            chunks,
            dense,
            sparse_weights=sparse,
            collection=args.collection,
            batch=args.batch_size,
            on_batch=progress,
        )
        count = client.count(collection_name=args.collection).count
        if count != len(chunks):
            raise ValueError(f"indexed {count} of {len(chunks)} chunks")
        print(f"Ready: {args.collection} contains {count} normalized chunks.")
    finally:
        client.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--runtime-path",
        type=Path,
        default=Path(".benchmark-runtime/qdrant"),
    )
    parser.add_argument("--collection", default="veritasmed_benchmark_v1_1")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--recreate", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
