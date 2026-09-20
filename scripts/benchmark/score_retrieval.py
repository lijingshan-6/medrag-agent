"""Score P1/P2/P3 directly from the frozen BGE-M3 cache.

This runner avoids a Qdrant service dependency while preserving the deployed
pipeline math: P1 cosine-ranked dense vectors, P2 dense+sparse RRF, and P3 the
same RRF candidates followed by the configured BGE cross-encoder.
"""

from __future__ import annotations

# Load PyTorch before any optional Qdrant/grpc imports on Windows.
import sentence_transformers  # noqa: F401

import argparse
import gc
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import numpy as np

from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.scoring import score_retrieval
from medrag.index.embedder import BGEM3Embedder
from medrag.retrieval.retriever import RetrievedChunk


def _read_questions(path: Path, split: str) -> list[BenchmarkQuestion]:
    with path.open(encoding="utf-8") as handle:
        rows = [BenchmarkQuestion.model_validate_json(line) for line in handle if line.strip()]
    if split == "all":
        return rows
    return [row for row in rows if row.split.value == split]


def _top_indices(scores: np.ndarray, count: int) -> list[int]:
    count = min(count, len(scores))
    if count == 0:
        return []
    selected = np.argpartition(scores, -count)[-count:]
    return selected[np.argsort(scores[selected])[::-1]].tolist()


def _rrf(first: list[int], second: list[int], constant: int = 60) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for ranking in (first, second):
        for rank, row_index in enumerate(ranking, start=1):
            scores[row_index] += 1.0 / (constant + rank)
    return [row_index for row_index, _ in sorted(scores.items(), key=lambda item: -item[1])]


def _load_sparse_rows(path: Path) -> list[dict[str, float]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _sparse_scores(
    rows: list[dict[str, float]],
    query: dict[str, float],
) -> np.ndarray:
    if not query:
        return np.zeros(len(rows), dtype=np.float32)
    items = list(query.items())
    return np.fromiter(
        (
            sum(float(weight) * float(row.get(token, 0.0)) for token, weight in items)
            for row in rows
        ),
        dtype=np.float32,
        count=len(rows),
    )


def _summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    answerable = [row for row in rows if row["answerability"] != "unanswerable"]

    def mean(group: list[dict[str, Any]], metric: str) -> float:
        return fmean(float(row["metrics"][metric]) for row in group) if group else 0.0

    return {
        "questions": len(rows),
        "answerable_questions": len(answerable),
        "required_claim_recall": mean(answerable, "required_claim_recall"),
        "supporting_chunk_recall": mean(answerable, "supporting_chunk_recall"),
        "all_required_found_rate": mean(answerable, "all_required_found"),
        "ndcg": mean(answerable, "ndcg"),
        "mrr": mean(answerable, "reciprocal_rank"),
        "hard_negative_retrieval_rate": mean(rows, "hard_negative_hit"),
    }


def _group_summaries(rows: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {"overall": _summarize(rows)}
    for field in ("split", "task_type", "answerability", "category"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            groups[row[field]].append(row)
        output[f"by_{field}"] = {
            name: _summarize(group) for name, group in sorted(groups.items())
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    questions = _read_questions(root / args.questions, args.split)
    normalized = build_normalized_chunks(
        root / "data/raw/pubmed/abstracts.jsonl",
        root / "data/raw/pmc/full_texts.jsonl",
    )
    dense = np.load(root / "data/index_cache/dense.npy", mmap_mode="r")
    if len(normalized) != dense.shape[0]:
        raise SystemExit(f"chunk/vector row mismatch: {len(normalized)} vs {dense.shape[0]}")

    pipelines = [item.strip() for item in args.pipelines.split(",") if item.strip()]
    unknown = set(pipelines) - {"p1", "p2", "p3"}
    if unknown:
        raise SystemExit("unknown pipelines: " + ", ".join(sorted(unknown)))

    print(f"[embed] loading BGE-M3 on {args.device} for {len(questions)} questions", flush=True)
    started = time.perf_counter()
    embedder = BGEM3Embedder(device=args.device)
    encoded = embedder.encode(
        [row.question for row in questions],
        batch_size=args.batch_size,
        return_sparse=("p2" in pipelines or "p3" in pipelines),
    )
    embedding_seconds = time.perf_counter() - started
    del embedder
    gc.collect()

    sparse_rows: list[dict[str, float]] | None = None
    if "p2" in pipelines or "p3" in pipelines:
        print("[cache] loading sparse corpus weights", flush=True)
        sparse_rows = _load_sparse_rows(root / "data/index_cache/sparse.jsonl")
        if len(sparse_rows) != len(normalized):
            raise SystemExit("sparse/chunk row mismatch")

    rankings: dict[tuple[str, str], list[int]] = {}
    latencies: dict[tuple[str, str], float] = {}
    for offset, question in enumerate(questions):
        began = time.perf_counter()
        dense_scores = dense @ encoded["dense"][offset]
        dense_ranking = _top_indices(dense_scores, args.candidate_k)
        if "p1" in pipelines:
            rankings[(question.id, "p1")] = dense_ranking[: args.top_k]
            latencies[(question.id, "p1")] = time.perf_counter() - began

        if "p2" in pipelines or "p3" in pipelines:
            assert sparse_rows is not None
            sparse_scores = _sparse_scores(sparse_rows, encoded["sparse"][offset])
            sparse_ranking = _top_indices(sparse_scores, args.candidate_k)
            fused = _rrf(dense_ranking, sparse_ranking)[: args.candidate_k]
            if "p2" in pipelines:
                rankings[(question.id, "p2")] = fused[: args.top_k]
                latencies[(question.id, "p2")] = time.perf_counter() - began
            if "p3" in pipelines:
                rankings[(question.id, "p3")] = fused
                latencies[(question.id, "p3")] = time.perf_counter() - began
        print(f"[rank] {offset + 1}/{len(questions)} {question.id}", flush=True)

    if "p3" in pipelines:
        from medrag.retrieval.reranker import BGEReranker

        print(f"[rerank] loading cross-encoder on {args.reranker_device}", flush=True)
        reranker = BGEReranker(
            device=args.reranker_device,
            use_fp16=args.reranker_device == "cuda",
            batch_size=args.reranker_batch_size,
        )
        for offset, question in enumerate(questions):
            began = time.perf_counter()
            candidate_indices = rankings[(question.id, "p3")]
            candidates = [
                RetrievedChunk(
                    chunk_id=normalized[index]["chunk_id"],
                    text=normalized[index]["text"],
                    score=0.0,
                    payload={
                        "source": normalized[index]["source"],
                        "doc_id": normalized[index]["doc_id"],
                    },
                )
                for index in candidate_indices
            ]
            reranked = reranker.rerank(question.question, candidates, top_k=args.top_k)
            index_by_id = {
                normalized[index]["chunk_id"]: index for index in candidate_indices
            }
            rankings[(question.id, "p3")] = [index_by_id[row.chunk_id] for row in reranked]
            latencies[(question.id, "p3")] += time.perf_counter() - began
            print(f"[rerank] {offset + 1}/{len(questions)} {question.id}", flush=True)

    results: list[dict[str, Any]] = []
    for question in questions:
        for pipeline in pipelines:
            indices = rankings[(question.id, pipeline)]
            chunk_ids = [normalized[index]["chunk_id"] for index in indices]
            score = score_retrieval(question, chunk_ids, k=args.top_k)
            results.append(
                {
                    "id": question.id,
                    "split": question.split.value,
                    "task_type": question.task_type.value,
                    "answerability": question.answerability.value,
                    "category": question.category,
                    "pipeline": pipeline,
                    "retrieved_chunk_ids": chunk_ids,
                    "metrics": score.as_dict(),
                    "latency_seconds": latencies[(question.id, pipeline)],
                }
            )

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    ).stdout.strip()
    output = {
        "schema_version": "1.0",
        "questions": str(args.questions).replace("\\", "/"),
        "split": args.split,
        "pipelines": pipelines,
        "top_k": args.top_k,
        "candidate_k": args.candidate_k,
        "embedding_seconds": embedding_seconds,
        "device": args.device,
        "reranker_device": args.reranker_device if "p3" in pipelines else None,
        "platform": platform.platform(),
        "code_commit": commit,
        "summaries": {
            pipeline: _group_summaries(
                [row for row in results if row["pipeline"] == pipeline]
            )
            for pipeline in pipelines
        },
        "results": results,
    }
    output_path = root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/questions.jsonl"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/baseline_retrieval.json"),
    )
    parser.add_argument("--pipelines", default="p1,p2,p3")
    parser.add_argument("--split", choices=["development", "test", "all"], default="all")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-k", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reranker-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    args = parser.parse_args()
    output = run(args)
    print(json.dumps(output["summaries"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
