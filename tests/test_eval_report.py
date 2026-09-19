"""Release reporting uses hand-derived metrics and immutable source artifacts."""

import json
from pathlib import Path

import pytest

from medrag.eval.report import build_summary, render_report, write_or_check


def _agent_results():
    strict = {
        "pipeline": "p4-agentic", "model_judge": "mimo-v2.5-pro", "n": 2,
        "results": [
            {"id": "Q001", "faithfulness": 0.9, "relevance": 0.6,
             "correctness": 0.3, "composite": 0.6, "agent_latency_s": 10.0},
            {"id": "Q002", "error": "example failure"},
        ],
    }
    hard = {
        "pipeline": "p4-agentic", "model_judge": "mimo-v2.5-pro", "n": 1,
        "results": [{"id": "H001", "faithfulness": 0.8, "relevance": 0.7,
                     "correctness": 0.6, "composite": 0.7, "agent_latency_s": 20.0}],
    }
    return strict, hard


def _retrieval_results():
    # Q001: two gold chunks, one at rank 1. Q002: one gold chunk at rank 2.
    # P1 => Hit@5=1, macro recall=.75, MRR@20=.75.
    rows = []
    for pipeline, rankings in {
        "p1": [["a", "noise"], ["noise", "c"]],
        "p2": [["noise", "noise2", "a"], ["noise", "c"]],
        "p3": [["a", "b"], ["noise", "noise2"]],
    }.items():
        for i, retrieved in enumerate(rankings):
            rows.append({"id": f"Q00{i + 1}", "pipeline": pipeline,
                         "gold_chunk_ids": ["a", "b"] if i == 0 else ["c"],
                         "retrieved_ids": retrieved, "latency_s": 1.0})
    return {"n_questions": 2, "pipelines": ["p1", "p2", "p3"], "results": rows}


def test_hand_derived_metrics_keep_errors_and_evidence_denominators_distinct():
    strict, hard = _agent_results()
    summary = build_summary(strict, hard, _retrieval_results(), {"sample": "abc"})

    assert summary["strict"]["attempted"] == 2
    assert summary["strict"]["scored"] == 1
    assert summary["strict"]["errors"] == [{"id": "Q002", "error": "example failure"}]
    assert summary["strict"]["composite_scored"] == pytest.approx(0.6)
    assert summary["strict"]["composite_all_attempts"] == pytest.approx(0.3)
    assert summary["strict"]["latency_s"]["mean"] == pytest.approx(10)
    assert summary["hard"]["composite_scored"] == pytest.approx(0.7)
    assert summary["retrieval"]["p1"]["hit_at_5"] == pytest.approx(1)
    assert summary["retrieval"]["p1"]["macro_evidence_recall_at_5"] == pytest.approx(.75)
    assert summary["retrieval"]["p1"]["mrr_at_20"] == pytest.approx(.75)
    assert summary["retrieval"]["p2"]["mrr_at_20"] == pytest.approx(5 / 12)
    assert summary["retrieval"]["p3"]["hit_at_5"] == pytest.approx(.5)
    assert summary["retrieval"]["p3"]["macro_evidence_recall_at_5"] == pytest.approx(.5)
    assert "Q002" in render_report(summary)


@pytest.mark.parametrize("gold,retrieved", [([], ["x"]), (None, ["x"]), (["x"], None)])
def test_retrieval_requires_real_gold_evidence_and_ranking(gold, retrieved):
    strict, hard = _agent_results()
    retrieval = _retrieval_results()
    retrieval["results"][0]["gold_chunk_ids"] = gold
    retrieval["results"][0]["retrieved_ids"] = retrieved
    with pytest.raises(ValueError, match="Q001.*p1"):
        build_summary(strict, hard, retrieval, {})


def test_retrieval_rejects_different_gold_evidence_for_same_question():
    strict, hard = _agent_results()
    retrieval = _retrieval_results()
    retrieval["results"][2]["gold_chunk_ids"] = ["unrelated"]
    with pytest.raises(ValueError, match="gold evidence differs"):
        build_summary(strict, hard, retrieval, {})


def test_check_detects_stale_outputs_without_writing(tmp_path: Path):
    strict, hard = _agent_results()
    summary = build_summary(strict, hard, _retrieval_results(), {"sample": "abc"})
    json_path = tmp_path / "summary.json"
    report_path = tmp_path / "report.md"

    write_or_check(summary, json_path, report_path, check=False)
    original = json_path.read_bytes(), report_path.read_bytes()
    write_or_check(summary, json_path, report_path, check=True)
    json_path.write_text(json.dumps({"stale": True}), encoding="utf-8")
    with pytest.raises(ValueError, match="summary.json"):
        write_or_check(summary, json_path, report_path, check=True)
    assert report_path.read_bytes() == original[1]
    assert json_path.read_text(encoding="utf-8") == '{"stale": true}'
