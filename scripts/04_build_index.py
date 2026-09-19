"""Build Qdrant index from all ingested data (PubMed + PMC).

Three-phase design:
  Phase embed  — BGE-M3 dense encoding  → data/index_cache/dense.npy + chunks.jsonl
  Phase sparse — BGE-M3 sparse encoding → data/index_cache/sparse.jsonl
  Phase index  — load cached arrays, upsert into Qdrant (dense + sparse)

Run with --phase=embed, --phase=sparse, --phase=index, or --phase=all (default).
Existing cache files are skipped automatically in --phase=all mode.
"""
# Windows + CUDA: import sentence_transformers before qdrant_client so PyTorch
# loads before the gRPC C++ runtime, avoiding STATUS_ACCESS_VIOLATION (exit 5).
import sentence_transformers  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np
# Bypass FlagEmbedding.__init__ — the decoder-only reranker in that init chain
# crashes on Windows.  The encoder-only M3Embedder sub-module is safe to import.
from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder
from qdrant_client import QdrantClient

from medrag.config import COLLECTION_NAME, qdrant_url
from medrag.ingest.chunker import Chunk, chunk_pubmed_record, chunk_pmc_record
from medrag.index.qdrant_setup import create_collection
from medrag.index.indexer import index_chunks

CACHE_DIR = Path("data/index_cache")
DENSE_FILE = CACHE_DIR / "dense.npy"
SPARSE_FILE = CACHE_DIR / "sparse.jsonl"
CHUNKS_FILE = CACHE_DIR / "chunks.jsonl"


def load_chunks_from_cache() -> list:
    """Load chunk metadata from data/index_cache/chunks.jsonl (matches dense.npy rows)."""
    if not CHUNKS_FILE.exists():
        raise SystemExit(f"[error] {CHUNKS_FILE} not found. Run --phase=embed or --phase=all first.")
    chunks: list = []
    print(f"[index] loading chunks from {CHUNKS_FILE} ...", flush=True)
    with CHUNKS_FILE.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            row = json.loads(line)
            chunks.append(Chunk(
                chunk_id=row["chunk_id"],
                source=row["source"],
                doc_id=row["doc_id"],
                text=row["text"],
                metadata=row.get("metadata") or {},
            ))
            if i % 10000 == 0:
                print(f"[index]   ... {i} chunks loaded", flush=True)
    print(f"[index] loaded {len(chunks)} chunks from cache", flush=True)
    return chunks


def load_chunks() -> list:
    chunks = []
    pubmed_path = Path("data/raw/pubmed/abstracts.jsonl")
    pmc_path = Path("data/raw/pmc/full_texts.jsonl")
    if pubmed_path.exists():
        with pubmed_path.open(encoding="utf-8") as f:
            for line in f:
                chunks.extend(chunk_pubmed_record(json.loads(line)))
        print(f"[index] pubmed chunks: {len(chunks)}", flush=True)
    if pmc_path.exists():
        n = len(chunks)
        with pmc_path.open(encoding="utf-8") as f:
            for doc_idx, line in enumerate(f):
                rec = json.loads(line)
                if not rec.get("pmcid"):
                    rec["pmcid"] = f"doc{doc_idx}"
                chunks.extend(chunk_pmc_record(rec))
        print(f"[index] pmc chunks: {len(chunks) - n}", flush=True)
    print(f"[index] total chunks: {len(chunks)}", flush=True)
    return chunks


def phase_embed(chunks) -> np.ndarray:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[embed] loading BGE-M3 on cuda...", flush=True)
    model = M3Embedder("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
    print("[embed] model loaded", flush=True)
    texts = [c.text for c in chunks]
    print(f"[embed] encoding {len(texts)} texts ...", flush=True)
    out = model.encode(
        texts,
        batch_size=4,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
        max_length=512,
    )
    dense = np.array(out["dense_vecs"], dtype=np.float32)
    print(f"[embed] dense shape: {dense.shape}", flush=True)
    np.save(DENSE_FILE, dense)
    with CHUNKS_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c.__dict__, ensure_ascii=False) + "\n")
    print(f"[embed] saved to {CACHE_DIR}", flush=True)
    return dense


def phase_sparse(chunks) -> list[dict]:
    """Encode sparse (lexical) weights for all chunks and save to sparse.jsonl."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print("[sparse] loading BGE-M3 (fp16) on cuda...", flush=True)
    model = M3Embedder("BAAI/bge-m3", use_fp16=True, devices=["cuda:0"])
    print("[sparse] model loaded on cuda", flush=True)

    texts = [c.text for c in chunks]
    print(f"[sparse] encoding {len(texts)} texts (batch_size=256)...", flush=True)
    out = model.encode(
        texts,
        batch_size=256,
        return_dense=False,
        return_sparse=True,
        return_colbert_vecs=False,
        max_length=512,
    )
    # lexical_weights: list[dict[str, float]]
    sparse_weights: list[dict] = out["lexical_weights"]
    print(f"[sparse] encoded {len(sparse_weights)} items", flush=True)

    with SPARSE_FILE.open("w", encoding="utf-8") as f:
        for w in sparse_weights:
            f.write(json.dumps({k: float(v) for k, v in w.items()}, ensure_ascii=False) + "\n")
    print(f"[sparse] saved to {SPARSE_FILE}", flush=True)
    return sparse_weights


def phase_index(chunks, dense: np.ndarray, sparse_weights: list[dict] | None) -> None:
    print("[qdrant] connecting (timeout=120s)...", flush=True)
    client = QdrantClient(url=qdrant_url(), timeout=120)
    create_collection(client, COLLECTION_NAME, recreate=True)
    print(f"[qdrant] upserting {len(chunks)} points (batch=256, may take several minutes)...", flush=True)

    def _progress(done: int, total: int) -> None:
        pct = int(100 * done / total) if total else 0
        print(f"[qdrant]   uploaded {done}/{total} ({pct}%)", flush=True)

    index_chunks(
        client,
        chunks,
        dense,
        sparse_weights=sparse_weights,
        collection=COLLECTION_NAME,
        batch=256,
        on_batch=_progress,
    )
    count = client.count(collection_name=COLLECTION_NAME).count
    print(f"[done] qdrant points: {count}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--phase",
        choices=["embed", "sparse", "index", "all"],
        default="all",
    )
    args = parser.parse_args()

    chunks = None
    dense = None
    sparse_weights = None

    # ── Phase embed ───────────────────────────────────────────────────────────
    if args.phase in ("embed", "all"):
        if args.phase == "all" and DENSE_FILE.exists():
            print(f"[skip] {DENSE_FILE} already exists, loading from cache", flush=True)
            chunks = load_chunks()
            dense = np.load(DENSE_FILE)
            print(f"[embed] loaded dense shape: {dense.shape}", flush=True)
        else:
            chunks = load_chunks()
            if not chunks:
                raise SystemExit("[error] No chunks found.")
            dense = phase_embed(chunks)

    # ── Phase sparse ──────────────────────────────────────────────────────────
    if args.phase in ("sparse", "all"):
        if chunks is None:
            chunks = load_chunks()
        if args.phase == "all" and SPARSE_FILE.exists():
            print(f"[skip] {SPARSE_FILE} already exists, loading from cache", flush=True)
            with SPARSE_FILE.open(encoding="utf-8") as f:
                sparse_weights = [json.loads(line) for line in f]
            print(f"[sparse] loaded {len(sparse_weights)} entries", flush=True)
        else:
            sparse_weights = phase_sparse(chunks)

    # ── Phase index ───────────────────────────────────────────────────────────
    if args.phase == "index":
        print("[index] phase=index: upload index_cache → Qdrant (no re-embedding)", flush=True)
        chunks = load_chunks_from_cache()
        print(f"[index] loading dense vectors from {DENSE_FILE} ...", flush=True)
        dense = np.load(DENSE_FILE)
        print(f"[index] loaded dense shape: {dense.shape}", flush=True)
        if len(chunks) != dense.shape[0]:
            raise SystemExit(
                f"[error] chunks ({len(chunks)}) != dense rows ({dense.shape[0]}). "
                "Re-run --phase=all or replace index_cache from data bundle."
            )
        if SPARSE_FILE.exists():
            print(f"[index] loading sparse weights from {SPARSE_FILE} ...", flush=True)
            sparse_weights = []
            with SPARSE_FILE.open(encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    sparse_weights.append(json.loads(line))
                    if i % 10000 == 0:
                        print(f"[index]   ... {i} sparse rows loaded", flush=True)
            print(f"[index] loaded {len(sparse_weights)} sparse entries", flush=True)
        else:
            sparse_weights = None
            print("[index] no sparse.jsonl found, indexing dense-only", flush=True)

    if args.phase in ("index", "all"):
        phase_index(chunks, dense, sparse_weights)


if __name__ == "__main__":
    main()
