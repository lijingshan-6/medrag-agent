"""LangGraph node functions for MedRAG-Agent.

Each node receives the full AgentState and returns a *partial* dict
with only the fields it modifies.  Heavy resources (embedder, Qdrant,
reranker) are created lazily at first call so that importing this module
does not spin up GPU-heavy processes.

Node responsibilities
---------------------
route_query       — classify query as factual / synthesis / multihop
hybrid_retrieve   — dense+sparse RRF retrieval
rerank_chunks     — cross-encoder reranking (P3 quality)
grade_relevance   — score whether chunks can answer the query
rewrite_query     — rewrite a failed query; increment iterations
generate_answer_node — structured-JSON answer generation
check_faithfulness — verify every claim is grounded in context
summarize_history  — L2 memory: compress history when > 10 turns
"""
from __future__ import annotations

import json
import logging
import os
import re
from functools import lru_cache
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from medrag.agent.llms import make_llm_fast, make_llm_think
from medrag.agent.prompts import (
    CHECK_SYSTEM,
    CHECK_USER,
    GENERATE_SYSTEM,
    GENERATE_USER,
    GRADE_SYSTEM,
    GRADE_USER,
    REGEN_SYSTEM,
    REGEN_USER,
    REWRITE_SYSTEM,
    REWRITE_USER,
    ROUTER_SYSTEM,
    ROUTER_USER,
    SUMMARIZE_SYSTEM,
    SUMMARIZE_USER,
)
from medrag.agent.state import AgentState
from medrag.agent.utils import build_answer_from_claims, strip_thinking, validate_citations
from medrag.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────

MAX_REWRITES = 2          # up to 3 retrieval attempts total
MAX_REGEN    = 2          # up to 2 regen attempts; no confidence-skip bypass
GRADE_THRESHOLD = 0.75    # raised from 0.6 — stricter pass so rewrite fires meaningfully

# Dynamic grade thresholds by query type (router output)
_GRADE_THRESHOLDS = {
    "factual":   0.6,   # raised from 0.5
    "synthesis": 0.75,  # raised from 0.6
    "multihop":  0.8,   # raised from 0.7
}
HISTORY_SUMMARIZE_EVERY = 10  # L2 compression after this many turns
CANDIDATE_K  = 20         # hybrid retrieval candidate pool
TOP_K        = 5          # chunks passed to generator

# ── Lazy resource factories ────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_retriever():
    """Singleton HybridRetriever — created once per process."""
    from medrag.index.embedder import BGEM3Embedder
    from medrag.retrieval.hybrid import HybridRetriever

    device = os.environ.get("EMBEDDER_DEVICE", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info("[retriever] using device=%s", device)

    # Embedder MUST be created before QdrantClient — loading sentence_transformers
    # after qdrant_client's gRPC layer is initialized causes a segfault on Windows
    # due to a native library conflict between grpc and torch C++ runtimes.
    embedder = BGEM3Embedder(device=device)
    from medrag.config import COLLECTION_NAME, get_qdrant_client

    qdrant = get_qdrant_client()
    return HybridRetriever(qdrant, embedder, collection=COLLECTION_NAME, candidate_k=CANDIDATE_K)


@lru_cache(maxsize=1)
def _get_reranker():
    """Singleton BGEReranker — created once per process. Prefers GPU if available."""
    import os
    from medrag.retrieval.reranker import BGEReranker
    device = os.environ.get("RERANKER_DEVICE", "auto")
    if device == "auto":
        try:
            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            device = "cpu"
    logger.info("[reranker] using device=%s", device)
    return BGEReranker(device=device)


# ── JSON parsing helper ────────────────────────────────────────────────────────

def _invoke_with_retry(llm, messages, retries: int = 1) -> str:
    """Invoke LLM and retry once if response is empty (transient API issue)."""
    import time
    for attempt in range(1 + retries):
        resp = llm.invoke(messages)
        content = resp.content or ""
        raw = strip_thinking(content)
        if raw and raw.strip():
            return raw
        # If strip_thinking removed everything but original had content, use original
        if content and content.strip():
            logger.warning("[llm] strip_thinking returned empty but raw has %d chars — using raw", len(content))
            return content.strip()
        if attempt < retries:
            logger.warning("[llm] empty response — retrying (attempt %d/%d)", attempt + 1, retries)
            time.sleep(2)
    return raw  # return empty on final attempt


def _parse_json(text: str) -> dict[str, Any]:
    """Strip markdown fences and parse JSON; return {} on failure."""
    text = re.sub(r"```(?:json)?", "", text).strip().rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                pass
    logger.warning("Failed to parse LLM JSON output: %s", text[:200])
    return {}


def _format_context(chunks: list[RetrievedChunk]) -> str:
    """Format retrieved chunks for the generate prompt.

    Each chunk is prefixed with its citation key in square brackets so the
    model can reference it exactly in the 'cite' field of each claim.
    """
    parts = [
        f"[{c.citation}] (score={c.score:.3f}):\n{c.text}"
        for c in chunks
    ]
    return "\n\n".join(parts)


# ── Node: route_query ──────────────────────────────────────────────────────────

def route_query(state: AgentState) -> dict:
    """Classify the query as factual / synthesis / multihop.

    Uses llm_fast (thinking=OFF) — just a lightweight classification.
    Result stored in state but not used for routing in the graph edges;
    it is preserved for audit / downstream use.
    """
    llm = make_llm_fast()
    query = state["query"]

    raw = _invoke_with_retry(llm, [
        SystemMessage(content=ROUTER_SYSTEM),
        HumanMessage(content=ROUTER_USER.format(query=query)),
    ])
    parsed = _parse_json(raw)

    query_type = parsed.get("type", "factual")
    logger.info("[route] query_type=%s  reason=%s", query_type, parsed.get("reason", ""))

    # Preserve the original query before any rewrites happen; used by append_history
    # to record what the user actually asked regardless of query reformulations.
    return {
        "query": query,
        "original_query": query,
        "query_type": query_type,
        "iterations": state.get("iterations", 0),
        "regen_count": state.get("regen_count", 0),
    }


# ── Node: hybrid_retrieve ──────────────────────────────────────────────────────

def hybrid_retrieve(state: AgentState) -> dict:
    """Hybrid dense+sparse RRF retrieval.

    Returns top-CANDIDATE_K candidates (reranker will shrink to TOP_K).
    If retrieval fails, returns empty list so grade node can handle it.
    """
    query = state["query"]
    logger.info("[retrieve] query=%s", query[:80])

    try:
        retriever = _get_retriever()
        chunks = retriever.retrieve(query, k=CANDIDATE_K)
    except Exception as exc:
        logger.error("[retrieve] error: %s", exc)
        chunks = []

    logger.info("[retrieve] got %d candidates", len(chunks))
    return {"retrieved_chunks": chunks}


# ── Node: rerank_chunks ────────────────────────────────────────────────────────

def rerank_chunks(state: AgentState) -> dict:
    """Cross-encoder reranking: shrink CANDIDATE_K → TOP_K."""
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        return {"retrieved_chunks": []}

    try:
        reranker = _get_reranker()
        reranked = reranker.rerank(query, chunks, top_k=TOP_K)
    except Exception as exc:
        logger.error("[rerank] error: %s — falling back to top-%d by score", exc, TOP_K)
        reranked = sorted(chunks, key=lambda c: -c.score)[:TOP_K]

    logger.info("[rerank] kept top %d chunks", len(reranked))
    return {"retrieved_chunks": reranked}


# ── Node: grade_relevance ──────────────────────────────────────────────────────

def grade_relevance(state: AgentState) -> dict:
    """Score whether the retrieved chunks can fully answer the query.

    Uses llm_think (thinking=ON) for careful reasoning.
    Returns relevance_score (0-1), grade_reason, rewrite_hint.
    """
    llm = make_llm_think()
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    context = _format_context(chunks) if chunks else "(no chunks retrieved)"

    raw = _invoke_with_retry(llm, [
        SystemMessage(content=GRADE_SYSTEM),
        HumanMessage(content=GRADE_USER.format(query=query, context=context)),
    ])
    parsed = _parse_json(raw)

    score       = float(parsed.get("score", 0.0))
    reason      = str(parsed.get("reason", ""))
    rewrite_hint = str(parsed.get("rewrite_hint", ""))

    # Dynamic threshold based on query type from router
    query_type = state.get("query_type", "synthesis")
    threshold = _GRADE_THRESHOLDS.get(query_type, GRADE_THRESHOLD)

    relevant = bool(parsed.get("relevant", score >= threshold))

    # If LLM says relevant=true but score is low, trust the boolean
    if relevant and score < threshold:
        score = threshold

    logger.info("[grade] score=%.2f relevant=%s threshold=%.1f type=%s",
                score, relevant, threshold, query_type)
    return {
        "relevance_score": score,
        "relevant": relevant,
        "grade_reason": reason,
        "rewrite_hint": rewrite_hint,
    }


# ── Node: rewrite_query ────────────────────────────────────────────────────────

def rewrite_query(state: AgentState) -> dict:
    """Rewrite a failed query to improve retrieval.

    Uses llm_think (thinking=ON).  Increments the iterations counter.
    Also appends the old query to rewritten_queries for audit.
    """
    llm = make_llm_think()
    original_query = state["query"]
    previous_rewrites = state.get("rewritten_queries", [])
    reason = state.get("grade_reason", "")
    hint   = state.get("rewrite_hint", "")

    raw = _invoke_with_retry(llm, [
        SystemMessage(content=REWRITE_SYSTEM),
        HumanMessage(content=REWRITE_USER.format(
            query=original_query,
            previous_rewrites=", ".join(previous_rewrites) or "none",
            reason=reason,
            hint=hint,
        )),
    ])
    new_query = raw.strip().strip('"').strip("'")

    iterations = state.get("iterations", 0) + 1
    logger.info("[rewrite] iter=%d  new_query=%s", iterations, new_query[:80])

    return {
        "query": new_query,
        "rewritten_queries": [new_query],   # Annotated[list, add] — appends
        "iterations": iterations,
    }


# ── Node: generate_answer_node ─────────────────────────────────────────────────

def generate_answer_node(state: AgentState) -> dict:
    """Generate a citation-grounded answer from retrieved context.

    Uses llm_fast (thinking=OFF) — retrieval-grounded, low latency.

    Pipeline:
      1. LLM outputs {"claims": [{"text":…, "cite":[…]}], "confidence":…}
      2. validate_citations() filters out claims whose cite keys are not in
         the current retrieval context (prevents hallucinated references).
      3. build_answer_from_claims() reconstructs a readable answer string
         with inline [PMID:xxx] / [PMC:xxx] markers.
      4. If 0 claims survive validation, the answer is set to a disclaimer
         and confidence=0.0; check_faithfulness will mark it unfaithful,
         triggering one regen attempt via the graph's inc_regen path.
    """
    llm = make_llm_fast()
    query  = state["query"]
    chunks = state.get("retrieved_chunks", [])
    context = _format_context(chunks) if chunks else "(no context available)"
    regen_count = state.get("regen_count", 0)
    faith_issues = state.get("faithfulness_issues", "")

    # Use REGEN prompt if this is a re-generation attempt
    if regen_count > 0 and faith_issues:
        system_prompt = REGEN_SYSTEM.format(faithfulness_issues=faith_issues)
        user_prompt = REGEN_USER.format(
            query=query, context=context, faithfulness_issues=faith_issues)
        logger.info("[generate] regen attempt #%d — using REGEN prompt", regen_count)
    else:
        system_prompt = GENERATE_SYSTEM
        user_prompt = GENERATE_USER.format(query=query, context=context)

    raw = _invoke_with_retry(llm, [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt),
    ])
    if not raw.strip():
        logger.warning("[generate] LLM returned completely empty response after retry")
    else:
        logger.debug("[generate] raw response (%d chars): %s", len(raw), raw[:300])
    parsed = _parse_json(raw)

    # ── Citation-grounded validation ──────────────────────────────────────
    claims_raw: list[dict] = parsed.get("claims", [])
    confidence: float = float(parsed.get("confidence", 0.5))

    if not claims_raw:
        # LLM returned old-style {answer, citations} or empty claims
        # Graceful fallback: wrap entire answer as a single unverified claim
        legacy_answer = str(parsed.get("answer", raw))
        legacy_cites  = list(parsed.get("citations", []))
        if legacy_answer and legacy_cites:
            claims_raw = [{"text": legacy_answer, "cite": legacy_cites}]
            logger.info("[generate] legacy answer format detected, wrapping as single claim")
        else:
            logger.warning("[generate] LLM returned no claims and no legacy answer")

    validated_claims = validate_citations(claims_raw, chunks)
    answer, citations = build_answer_from_claims(validated_claims)

    if not validated_claims:
        # All claims failed citation validation — signal to check node
        confidence = 0.0
        logger.warning("[generate] all claims failed citation validation — answer set to disclaimer")

    logger.info("[generate] confidence=%.2f  valid_claims=%d  citations=%s",
                confidence, len(validated_claims), citations)
    return {
        "answer": answer,
        "citations": citations,
        "confidence": confidence,
    }


# ── Node: check_faithfulness ───────────────────────────────────────────────────

def check_faithfulness(state: AgentState) -> dict:
    """Verify that every factual claim in the answer is grounded in context.

    Uses llm_think (thinking=ON) for careful cross-referencing.
    Returns faithful (bool) and faithfulness_issues (str).
    """
    llm = make_llm_think()
    chunks = state.get("retrieved_chunks", [])
    answer = state.get("answer", "")
    context = _format_context(chunks) if chunks else "(no context)"

    raw = _invoke_with_retry(llm, [
        SystemMessage(content=CHECK_SYSTEM),
        HumanMessage(content=CHECK_USER.format(context=context, answer=answer)),
    ])
    parsed = _parse_json(raw)

    faithful = bool(parsed.get("faithful", False))
    issues   = str(parsed.get("issues", ""))

    logger.info("[check] faithful=%s", faithful)
    return {
        "faithful": faithful,
        "faithfulness_issues": issues,
    }


# ── Node: increment_regen ─────────────────────────────────────────────────────

def increment_regen(state: AgentState) -> dict:
    """Increment the regen counter before looping back to generate.

    Separated from check_faithfulness so the counter update is persisted
    correctly by the LangGraph checkpointer (edge functions are read-only).
    """
    new_count = state.get("regen_count", 0) + 1
    logger.info("[regen] regen_count → %d", new_count)
    return {"regen_count": new_count}


# ── Node: append_history ──────────────────────────────────────────────────────

def append_history(state: AgentState) -> dict:
    """Append the completed Q&A turn to conversation history.

    Called once per turn, just before summarize_gate.  Uses original_query
    (pre-rewrite) so history records what the user actually asked, not the
    internally reformulated query.
    """
    original = state.get("original_query") or state.get("query", "")
    answer   = state.get("answer", "")
    return {"history": [{"query": original, "answer": answer}]}


# ── Node: summarize_history ────────────────────────────────────────────────────

def summarize_history(state: AgentState) -> dict:
    """L2 memory: compress conversation history into a rolling summary.

    Triggered when len(history) is a multiple of HISTORY_SUMMARIZE_EVERY.
    Uses llm_fast (thinking=OFF) — compression, not reasoning.
    Returns updated summary; history list itself is NOT cleared here
    (LangGraph checkpointer preserves it for crash recovery).
    """
    history  = state.get("history", [])
    summary  = state.get("summary", "")

    if not history:
        return {}

    # Format new turns to incorporate
    turns_text = "\n".join(
        f"Q: {h.get('query', '')}\nA: {h.get('answer', '')}"
        for h in history[-(HISTORY_SUMMARIZE_EVERY):]
    )

    llm = make_llm_fast()
    raw = _invoke_with_retry(llm, [
        SystemMessage(content=SUMMARIZE_SYSTEM),
        HumanMessage(content=SUMMARIZE_USER.format(
            previous_summary=summary or "(none)",
            turns=turns_text,
        )),
    ])
    new_summary = raw.strip()
    logger.info("[summarize] updated summary (%d chars)", len(new_summary))
    return {"summary": new_summary}


__all__ = [
    "route_query",
    "hybrid_retrieve",
    "rerank_chunks",
    "grade_relevance",
    "rewrite_query",
    "generate_answer_node",
    "check_faithfulness",
    "increment_regen",
    "append_history",
    "summarize_history",
    "MAX_REWRITES",
    "MAX_REGEN",
    "GRADE_THRESHOLD",
    "_GRADE_THRESHOLDS",
    "HISTORY_SUMMARIZE_EVERY",
]
