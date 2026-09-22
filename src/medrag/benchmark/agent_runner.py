"""Run one isolated benchmark question through the production Agent graph."""

from __future__ import annotations

import hashlib
import time
from typing import Any
from uuid import uuid4


def initial_agent_state(query: str) -> dict[str, Any]:
    """Match the state contract used by the Ask WebSocket route."""

    return {
        "query": query,
        "original_query": "",
        "query_type": "factual",
        "source_scope": "general",
        "answer_mode": "direct",
        "search_queries": [],
        "answer_requirements": [],
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "retrieval_groups": [],
        "relevance_score": 0.0,
        "relevant": False,
        "grade_reason": "",
        "rewrite_hint": "",
        "iterations": 0,
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "evidence_status": "complete",
        "evidence_gap": "",
        "faithful": False,
        "answer_supported": False,
        "answer_complete": False,
        "boundary_correct": False,
        "faithfulness_issues": "",
        "regen_count": 0,
        "history": [],
        "summary": "",
    }


def _chunk_id(chunk: Any) -> str | None:
    if hasattr(chunk, "chunk_id"):
        return str(chunk.chunk_id)
    if isinstance(chunk, dict):
        value = chunk.get("chunk_id") or (chunk.get("payload") or {}).get("chunk_id")
        return str(value) if value else None
    return None


def run_agent_question(agent: Any, *, question_id: str, query: str) -> dict[str, Any]:
    """Invoke the compiled graph once and retain the final auditable state."""

    thread_id = f"benchmark-{question_id}-{uuid4()}"
    config = {"configurable": {"thread_id": thread_id}}
    began = time.perf_counter()
    final = agent.invoke(initial_agent_state(query), config=config)
    elapsed = time.perf_counter() - began
    answer = str(final.get("answer", ""))
    retrieved_chunk_ids = [
        chunk_id
        for chunk in final.get("retrieved_chunks", [])
        if (chunk_id := _chunk_id(chunk)) is not None
    ]
    return {
        "id": question_id,
        "thread_id": thread_id,
        "query": query,
        "answer": answer,
        "answer_sha256": hashlib.sha256(answer.encode("utf-8")).hexdigest(),
        "citations": list(final.get("citations", [])),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "confidence": float(final.get("confidence", 0.0)),
        "faithful": bool(final.get("faithful", False)),
        "faithfulness_issues": str(final.get("faithfulness_issues", "")),
        "answer_supported": bool(final.get("answer_supported", False)),
        "answer_complete": bool(final.get("answer_complete", False)),
        "boundary_correct": bool(final.get("boundary_correct", False)),
        "evidence_status": str(final.get("evidence_status", "complete")),
        "evidence_gap": str(final.get("evidence_gap", "")),
        "search_queries": list(final.get("search_queries", [])),
        "answer_requirements": list(final.get("answer_requirements", [])),
        "source_scope": str(final.get("source_scope", "general")),
        "answer_mode": str(final.get("answer_mode", "direct")),
        "relevance_score": float(final.get("relevance_score", 0.0)),
        "iterations": int(final.get("iterations", 0)),
        "regen_count": int(final.get("regen_count", 0)),
        "rewritten_queries": list(final.get("rewritten_queries", [])),
        "latency_seconds": elapsed,
    }


__all__ = ["initial_agent_state", "run_agent_question"]
