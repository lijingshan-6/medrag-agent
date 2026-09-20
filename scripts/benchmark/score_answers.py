"""Generate and score reviewable benchmark answers with local Ollama models."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from statistics import fmean
from typing import Any

import httpx
from pydantic import BaseModel, ConfigDict, Field

from medrag.benchmark.inventory import build_normalized_chunks
from medrag.benchmark.review import ollama_json
from medrag.benchmark.schema import BenchmarkQuestion
from medrag.benchmark.scoring import score_answer


class ProposedClaimMapping(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_id: str
    citation_chunk_ids: list[str] = Field(default_factory=list)


class ProposedAnswerAssessment(BaseModel):
    model_config = ConfigDict(extra="forbid")
    claim_mappings: list[ProposedClaimMapping] = Field(default_factory=list)
    abstained: bool
    boundary_acknowledged: bool
    forbidden_claims_present: bool
    notes: list[str] = Field(default_factory=list, max_length=4)


def _read_questions(path: Path, split: str) -> dict[str, BenchmarkQuestion]:
    with path.open(encoding="utf-8") as handle:
        rows = [BenchmarkQuestion.model_validate_json(line) for line in handle if line.strip()]
    return {
        row.id: row
        for row in rows
        if split == "all" or row.split.value == split
    }


def _citation(chunk: dict[str, Any]) -> str:
    if chunk["source"] == "pubmed":
        return f"PMID:{chunk['doc_id']}"
    return f"PMC:{chunk['doc_id']}"


def _generate_answer(
    *,
    model: str,
    question: BenchmarkQuestion,
    chunks: list[dict[str, Any]],
    base_url: str,
    timeout_seconds: float,
) -> tuple[str, str, float]:
    system = (
        "You are a medical literature assistant. Use only the retrieved passages. "
        "Preserve study population, comparator, time point, uncertainty, and study-design limits. "
        "Cite sources inline as [PMID:123] or [PMC:123]. If the passages do not support the "
        "requested comparison or outcome, state that clearly and name what evidence is missing. "
        "Do not use outside medical knowledge."
    )
    context = "\n\n".join(
        f"<doc id='{_citation(chunk)}' chunk_id='{chunk['chunk_id']}'>\n{chunk['text']}\n</doc>"
        for chunk in chunks
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {
                "role": "user",
                "content": f"Question: {question.question}\n\nRetrieved passages:\n{context}\n\nAnswer with inline citations.",
            },
        ],
        "stream": False,
        "think": False,
        "options": {"temperature": 0, "num_ctx": 8192},
    }
    began = time.perf_counter()
    with httpx.Client(timeout=timeout_seconds) as client:
        response = client.post(f"{base_url.rstrip('/')}/api/chat", json=payload)
        response.raise_for_status()
    answer = response.json()["message"]["content"].strip()
    elapsed = time.perf_counter() - began
    return answer, hashlib.sha256(answer.encode("utf-8")).hexdigest(), elapsed


def _assessment_prompt(question: BenchmarkQuestion, answer: str) -> str:
    gold = [
        {
            "claim_id": claim.claim_id,
            "claim": claim.text,
            "allowed_citation_chunk_ids": [item.chunk_id for item in claim.evidence],
        }
        for claim in question.gold_claims
    ]
    cited_sources = sorted(set(re.findall(r"\[(?:PMID|PMC):[^\]]+\]", answer)))
    return f"""Audit one answer against the supplied gold contract. Do not use outside knowledge.

Question answerability: {question.answerability.value}
Question: {question.question}
Gold claims and the only allowed citation chunk IDs:
{json.dumps(gold, ensure_ascii=False, indent=2)}
Missing evidence contract:
{json.dumps(question.missing_evidence, ensure_ascii=False)}
Forbidden claims:
{json.dumps(question.forbidden_claims, ensure_ascii=False)}

Answer:
{answer}

Visible source citations in answer: {json.dumps(cited_sources, ensure_ascii=False)}

For each gold claim actually stated with its required qualifiers, return a claim mapping. Attach only
allowed chunk IDs whose PMID/PMC source is visibly cited in the answer. Mark abstained only when the
answer refuses the requested unsupported conclusion. Mark boundary_acknowledged when it explicitly
states the declared missing evidence or equivalent design boundary. Mark forbidden_claims_present
when it makes any forbidden or causal/clinical claim that exceeds the passages. Notes must be short.
"""


def _load_overrides(path: Path | None) -> dict[str, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        return {
            row["id"]: row
            for line in handle
            if line.strip()
            for row in [json.loads(line)]
        }


def _summary(results: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = [
        "claim_completeness",
        "claim_support_precision",
        "citation_coverage",
        "answerability_score",
        "strict_pass",
    ]

    def summarize(group: list[dict[str, Any]]) -> dict[str, Any]:
        return {
            "questions": len(group),
            **{
                name: fmean(float(row["metrics"][name]) for row in group)
                if group
                else 0.0
                for name in metric_names
            },
        }

    output: dict[str, Any] = {"overall": summarize(results)}
    for field in ("answerability", "task_type", "category"):
        groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in results:
            groups[row[field]].append(row)
        output[f"by_{field}"] = {
            name: summarize(group) for name, group in sorted(groups.items())
        }
    return output


def run(args: argparse.Namespace) -> dict[str, Any]:
    root = args.root.resolve()
    questions = _read_questions(root / args.questions, args.split)
    retrieval = json.loads((root / args.retrieval).read_text(encoding="utf-8"))
    rankings = {
        row["id"]: row["retrieved_chunk_ids"]
        for row in retrieval["results"]
        if row["pipeline"] == args.pipeline and row["id"] in questions
    }
    missing_rankings = set(questions) - set(rankings)
    if missing_rankings:
        raise SystemExit("retrieval artifact is missing: " + ", ".join(sorted(missing_rankings)))

    chunks = build_normalized_chunks(
        root / "data/raw/pubmed/abstracts.jsonl",
        root / "data/raw/pmc/full_texts.jsonl",
    )
    chunk_map = {row["chunk_id"]: row for row in chunks}
    output_path = root / args.output
    if output_path.exists() and args.resume:
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        existing = {row["id"]: row for row in payload.get("results", [])}
    else:
        existing = {}

    overrides = _load_overrides(root / args.overrides if args.overrides else None)
    results: list[dict[str, Any]] = []
    for offset, (question_id, question) in enumerate(questions.items(), start=1):
        prior = existing.get(question_id, {})
        if prior.get("assessment") and args.resume and question_id not in overrides:
            results.append(prior)
            print(f"[resume] {offset}/{len(questions)} {question_id}", flush=True)
            continue

        selected_chunks = [chunk_map[chunk_id] for chunk_id in rankings[question_id]]
        answer = prior.get("answer")
        if not answer:
            answer, answer_hash, generation_seconds = _generate_answer(
                model=args.answer_model,
                question=question,
                chunks=selected_chunks,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
            )
        else:
            answer_hash = prior["answer_sha256"]
            generation_seconds = prior["generation_seconds"]

        if question_id in overrides:
            assessment = overrides[question_id]["assessment"]
            review_status = "curator_override"
            judge_hash = prior.get("judge_raw_output_sha256")
        else:
            proposed = ollama_json(
                model=args.judge_model,
                prompt=_assessment_prompt(question, answer),
                response_model=ProposedAnswerAssessment,
                base_url=args.base_url,
                timeout_seconds=args.timeout,
            )
            assessment = proposed.value
            review_status = "model_proposed"
            judge_hash = proposed.raw_output_sha256

        claim_mappings = {
            row["claim_id"]: row["citation_chunk_ids"]
            for row in assessment["claim_mappings"]
        }
        metrics = score_answer(
            question,
            claim_mappings=claim_mappings,
            abstained=assessment["abstained"],
            boundary_acknowledged=assessment["boundary_acknowledged"],
            forbidden_claims_present=assessment["forbidden_claims_present"],
        )
        results.append(
            {
                "id": question_id,
                "split": question.split.value,
                "task_type": question.task_type.value,
                "answerability": question.answerability.value,
                "category": question.category,
                "pipeline": args.pipeline,
                "retrieved_chunk_ids": rankings[question_id],
                "answer": answer,
                "answer_sha256": answer_hash,
                "generation_seconds": generation_seconds,
                "answer_model": args.answer_model,
                "judge_model": args.judge_model,
                "judge_raw_output_sha256": judge_hash,
                "assessment_status": review_status,
                "assessment": assessment,
                "metrics": metrics.as_dict(),
            }
        )
        checkpoint = {
            "schema_version": "1.0",
            "split": args.split,
            "pipeline": args.pipeline,
            "answer_model": args.answer_model,
            "judge_model": args.judge_model,
            "summary": _summary(results),
            "results": results,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(checkpoint, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
            newline="\n",
        )
        print(f"[answer] {offset}/{len(questions)} {question_id}", flush=True)

    payload = {
        "schema_version": "1.0",
        "split": args.split,
        "pipeline": args.pipeline,
        "answer_model": args.answer_model,
        "judge_model": args.judge_model,
        "summary": _summary(results),
        "results": results,
    }
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/questions.jsonl"),
    )
    parser.add_argument(
        "--retrieval",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/baseline_retrieval.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/benchmark/veritasmed_v1/baseline_answers_dev.json"),
    )
    parser.add_argument("--overrides", type=Path)
    parser.add_argument("--split", choices=["development", "test", "all"], default="development")
    parser.add_argument("--pipeline", choices=["p1", "p2", "p3"], default="p2")
    parser.add_argument("--answer-model", default="qwen3:8b")
    parser.add_argument("--judge-model", default="medgemma1.5:4b")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    payload = run(args)
    print(json.dumps(payload["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
