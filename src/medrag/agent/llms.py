"""LLM factory: dual-backend strategy with thinking control per node.

Backend selection via environment variable LLM_BACKEND (default: mimo):

  LLM_BACKEND=mimo    → ChatOpenAI pointing at MiMo-V2.5 API
  LLM_BACKEND=ollama  → ChatOllama pointing at the configured local model

Two tiers per backend:
  make_llm_fast()   → route, generate, summarize nodes (thinking disabled)
  make_llm_think()  → grade, rewrite, check nodes (thinking disabled; larger context)

Note: MiMo's internal reasoning is disabled on both tiers via extra_body.
The "think" tier still uses the heavier Pro model for better accuracy.
Enabling reasoning (removing thinking.type=disabled) adds 15-27s per call.

MiMo env vars (read from .env):
  OPENAI_BASE_URL   — MiMo API base URL
  OPENAI_API_KEY    — MiMo API key
  MIMO_MODEL_FAST   — override fast model name  (default: mimo-v2.5)
  MIMO_MODEL_THINK  — override think model name (default: mimo-v2.5-pro)

Ollama env vars:
  OLLAMA_MODEL      — override model name (default: qwen3.5:9b)

See docs/architecture.md §4.1.1 for design rationale.
"""
from __future__ import annotations

import logging
import os

from medrag.config import DEFAULT_OLLAMA_MODEL

logger = logging.getLogger(__name__)

# ── Backend selection ──────────────────────────────────────────────────────────

_BACKEND = os.environ.get("LLM_BACKEND", "mimo").strip().lower()

# ── MiMo model names ───────────────────────────────────────────────────────────
_MIMO_FAST  = os.environ.get("MIMO_MODEL_FAST",  "mimo-v2.5")
_MIMO_THINK = os.environ.get("MIMO_MODEL_THINK", "mimo-v2.5-pro")

# ── Ollama model name ──────────────────────────────────────────────────────────
_OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL)


def _mimo_base_url() -> str:
    url = (
        os.environ.get("OPENAI_BASE_URL")
        or os.environ.get("OPENAI_API_BASE", "")
    ).rstrip("/")
    if not url:
        raise EnvironmentError(
            "MiMo backend requires OPENAI_BASE_URL (or OPENAI_API_BASE) in .env"
        )
    return url


def _mimo_api_key() -> str:
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise EnvironmentError("MiMo backend requires OPENAI_API_KEY in .env")
    return key


# ── Internal factory ───────────────────────────────────────────────────────────

def _make_llm(thinking: bool):
    """Shared factory — `thinking` selects tier (fast=OFF / think=ON/Pro)."""
    temp  = 0.6 if thinking else 0.2
    model = _MIMO_THINK if thinking else _MIMO_FAST

    backend = os.environ.get("LLM_BACKEND", "mimo").strip().lower()
    timeout = float(os.environ.get("LLM_TIMEOUT_SECONDS", "60"))
    if backend not in {"mimo", "ollama"}:
        raise ValueError("LLM_BACKEND must be mimo or ollama")
    if backend == "ollama":
        from langchain_ollama import ChatOllama
        # Direct output is required for the graph's small JSON contracts.  Qwen
        # 3.5's explicit reasoning mode can spend the entire request timeout in
        # hidden reasoning and then return a truncated or empty JSON response.
        logger.debug("[llm] %s → Ollama %s (reasoning disabled)",
                     "think" if thinking else "fast", _OLLAMA_MODEL)
        return ChatOllama(
            model=_OLLAMA_MODEL,
            base_url=os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/"),
            client_kwargs={"timeout": timeout},
            reasoning=False,
            temperature=temp,
            num_ctx=6144 if thinking else 4096,
        )

    # Default: mimo
    from langchain_openai import ChatOpenAI
    logger.debug("[llm] %s → MiMo %s", "think" if thinking else "fast", model)

    # MiMo models always reason internally unless explicitly disabled.
    # With thinking enabled and max_tokens=4096, the model burns 1000-5000+
    # reasoning tokens before producing content → 15-27 s for grade/check calls.
    # budget_tokens has no effect on this API; the only working control is
    # {"type": "disabled"}.  Grade/check quality stays high using the pro model
    # (mimo-v2.5-pro) even without explicit CoT; the fast model uses v2.5.
    return ChatOpenAI(
        model=model,
        base_url=_mimo_base_url(),
        api_key=_mimo_api_key(),
        temperature=temp,
        timeout=timeout,
        max_retries=1,
        max_tokens=4096 if thinking else 1024,
        extra_body={"thinking": {"type": "disabled"}},
    )


# ── Public factories ───────────────────────────────────────────────────────────

def make_llm_fast():
    """Low-latency LLM — thinking OFF. Used by: route_query, generate_answer_node, summarize_history."""
    return _make_llm(False)


def make_llm_think():
    """Pro-tier LLM (mimo-v2.5-pro, thinking disabled). Used by: grade_relevance, rewrite_query, check_faithfulness."""
    return _make_llm(True)


__all__ = ["make_llm_fast", "make_llm_think"]
