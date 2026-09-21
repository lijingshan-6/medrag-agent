"""Graph structure and unit tests for the LangGraph agentic loop.

These tests do NOT require Qdrant or Ollama — they mock all external calls
and only verify:
  1. Graph topology (all expected nodes present, edges correct)
  2. Conditional routing logic (_after_grade, _after_check)
  3. State transformations for rewrite_query, increment_regen, route_query
  4. SqliteSaver checkpointer wires up without error

Run with:
    pytest tests/test_agent.py -v
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def sample_state():
    """Minimal AgentState-compatible dict for testing."""
    return {
        "query": "What is the mechanism of aspirin?",
        "original_query": "",
        "rewritten_queries": [],
        "retrieved_chunks": [],
        "relevance_score": 0.0,
        "relevant": False,
        "grade_reason": "No relevant context found.",
        "rewrite_hint": "Try expanding to include COX inhibition.",
        "iterations": 0,
        "answer": "",
        "citations": [],
        "confidence": 0.0,
        "faithful": False,
        "faithfulness_issues": "",
        "regen_count": 0,
        "history": [],
        "summary": "",
    }


# ── Test 1: Graph topology ─────────────────────────────────────────────────────

class TestGraphTopology:
    """Verify the compiled graph contains all expected nodes."""

    def test_all_nodes_present(self):
        from medrag.agent.graph import app

        nodes = set(app.nodes.keys())
        expected = {
            "__start__",
            "route", "retrieve", "rerank", "grade",
            "rewrite", "generate", "check", "inc_regen",
            "append_history", "summarize_gate", "summarize",
        }
        missing = expected - nodes
        assert not missing, f"Missing nodes: {missing}"

    def test_graph_type(self):
        from langgraph.graph.state import CompiledStateGraph
        from medrag.agent.graph import app

        assert isinstance(app, CompiledStateGraph)

    def test_checkpointer_attached(self):
        from medrag.agent.graph import app

        assert app.checkpointer is not None


# ── Test 2: Conditional routing ───────────────────────────────────────────────

class TestConditionalRouting:
    """Unit tests for _after_grade and _after_check edge functions."""

    def test_after_grade_relevant(self, sample_state):
        from medrag.agent.graph import _after_grade

        state = {**sample_state, "relevance_score": 0.8, "iterations": 0}
        assert _after_grade(state) == "generate"

    def test_after_grade_rewrite(self, sample_state):
        from medrag.agent.graph import _after_grade

        state = {**sample_state, "relevance_score": 0.3, "iterations": 0}
        assert _after_grade(state) == "rewrite"

    def test_after_grade_max_rewrites_hit(self, sample_state):
        from medrag.agent.graph import _after_grade

        state = {**sample_state, "relevance_score": 0.2, "iterations": 2}
        assert _after_grade(state) == "generate"   # cap hit → generate anyway

    def test_after_grade_exact_threshold(self, sample_state):
        from medrag.agent.graph import _after_grade

        state = {**sample_state, "relevance_score": 0.75, "iterations": 0}
        assert _after_grade(state) == "generate"   # default synthesis threshold reached

    def test_after_check_faithful(self, sample_state):
        from medrag.agent.graph import _after_check

        state = {**sample_state, "faithful": True, "regen_count": 0}
        assert _after_check(state) == "end"

    def test_after_check_unfaithful_first_regen(self, sample_state):
        from medrag.agent.graph import _after_check

        state = {**sample_state, "faithful": False, "regen_count": 0}
        assert _after_check(state) == "regenerate"

    def test_after_check_unfaithful_cap_hit(self, sample_state):
        from medrag.agent.graph import _after_check

        state = {**sample_state, "faithful": False, "regen_count": 2}
        assert _after_check(state) == "end"   # regeneration cap reached

    def test_after_check_confidence_does_not_bypass_evidence_check(self, sample_state):
        from medrag.agent.graph import _after_check

        # first-gen, unfaithful, but has citations + confidence ≥ threshold
        state = {
            **sample_state,
            "faithful": False,
            "regen_count": 0,
            "citations": ["PMID:12345", "PMC:doc196"],
            "confidence": 0.99,
        }
        assert _after_check(state) == "regenerate"

    def test_after_check_regen_fires_without_citations(self, sample_state):
        from medrag.agent.graph import _after_check

        # first-gen, unfaithful, no citations → smart gate does not fire
        state = {**sample_state, "faithful": False, "regen_count": 0, "citations": [], "confidence": 0.9}
        assert _after_check(state) == "regenerate"


# ── Test 3: Node state transformations ────────────────────────────────────────

class TestNodeTransformations:
    """Unit tests for node functions with mocked LLM."""

    def test_rewrite_increments_counter(self, sample_state):
        from medrag.agent.nodes import rewrite_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="aspirin COX inhibitor prostaglandin synthesis"
        )

        with patch("medrag.agent.nodes.make_llm_think", return_value=mock_llm):
            state = {**sample_state, "iterations": 1}
            result = rewrite_query(state)

        assert result["iterations"] == 2
        assert result["query"] == "aspirin COX inhibitor prostaglandin synthesis"
        assert len(result["rewritten_queries"]) == 1

    def test_increment_regen_increments_counter(self, sample_state):
        from medrag.agent.nodes import increment_regen

        state = {**sample_state, "regen_count": 0}
        result = increment_regen(state)
        assert result["regen_count"] == 1

        state2 = {**sample_state, "regen_count": 1}
        result2 = increment_regen(state2)
        assert result2["regen_count"] == 2

    def test_route_query_initialises_counters(self, sample_state):
        from medrag.agent.nodes import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"type": "factual", "reason": "single fact requested"}'
        )

        with patch("medrag.agent.nodes.make_llm_fast", return_value=mock_llm):
            result = route_query(sample_state)

        assert result["iterations"] == 0
        assert result["regen_count"] == 0

    def test_route_query_preserves_original_query(self, sample_state):
        from medrag.agent.nodes import route_query

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content='{"type": "factual", "reason": "single fact"}'
        )

        with patch("medrag.agent.nodes.make_llm_fast", return_value=mock_llm):
            result = route_query(sample_state)

        assert result["original_query"] == sample_state["query"]

    def test_append_history_records_original_query_and_answer(self, sample_state):
        from medrag.agent.nodes import append_history

        state = {
            **sample_state,
            "original_query": "original question about aspirin",
            "answer": "Aspirin inhibits COX.",
        }
        result = append_history(state)
        assert len(result["history"]) == 1
        assert result["history"][0]["query"] == "original question about aspirin"
        assert result["history"][0]["answer"] == "Aspirin inhibits COX."

    def test_append_history_falls_back_to_query_when_original_missing(self, sample_state):
        from medrag.agent.nodes import append_history

        state = {**sample_state, "original_query": "", "answer": "Some answer."}
        result = append_history(state)
        assert result["history"][0]["query"] == sample_state["query"]

    def test_generate_falls_back_on_json_parse_error(self, sample_state):
        from medrag.agent.nodes import generate_answer_node

        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(
            content="Aspirin inhibits COX-1 and COX-2 enzymes."   # plain text, not JSON
        )

        with patch("medrag.agent.nodes.make_llm_fast", return_value=mock_llm):
            result = generate_answer_node(sample_state)

        # Should not raise; uncited claims are dropped → disclaimer
        assert "do not contain sufficient cited evidence" in result["answer"]
        assert result["citations"] == []
        assert result["confidence"] == 0.0

    def test_regen_prompt_repeats_the_json_contract(self):
        from medrag.agent.prompts import REGEN_SYSTEM

        assert '"claims"' in REGEN_SYSTEM
        assert '"cite"' in REGEN_SYSTEM
        assert '"confidence"' in REGEN_SYSTEM
        rendered = REGEN_SYSTEM.format(faithfulness_issues="unsupported claim")
        assert '"claims"' in rendered

    def test_grade_relevance_trusts_boolean_over_score(self, sample_state):
        from medrag.agent.nodes import grade_relevance

        mock_llm = MagicMock()
        # LLM says relevant=true but score=0.4 (below threshold)
        mock_llm.invoke.return_value = MagicMock(
            content='{"relevant": true, "score": 0.4, "reason": "partial", "rewrite_hint": ""}'
        )

        with patch("medrag.agent.nodes.make_llm_think", return_value=mock_llm):
            result = grade_relevance(sample_state)

        # relevant=true should bump score to at least GRADE_THRESHOLD
        assert result["relevance_score"] >= 0.6


# ── Test 4: Memory helpers ────────────────────────────────────────────────────

class TestMemoryHelpers:
    """Test L2 summarisation trigger logic."""

    def test_maybe_summarize_triggers_at_10(self):
        from medrag.agent.graph import _maybe_summarize

        state_10 = {"history": [{}] * 10}
        assert _maybe_summarize(state_10) == "summarize"

    def test_maybe_summarize_skips_at_9(self):
        from medrag.agent.graph import _maybe_summarize

        state_9 = {"history": [{}] * 9}
        assert _maybe_summarize(state_9) == "end"

    def test_maybe_summarize_triggers_at_20(self):
        from medrag.agent.graph import _maybe_summarize

        state_20 = {"history": [{}] * 20}
        assert _maybe_summarize(state_20) == "summarize"

    def test_maybe_summarize_empty_history(self):
        from medrag.agent.graph import _maybe_summarize

        state_empty = {"history": []}
        assert _maybe_summarize(state_empty) == "end"
