import hashlib
import json
from pathlib import Path

import pytest

from medrag.benchmark.saved_run import recompute_saved_run


@pytest.fixture
def saved_inputs(tmp_path):
    source = Path(__file__).resolve().parents[1] / "data/benchmark/veritasmed_v1_1/questions.jsonl"
    question = json.loads(source.read_text(encoding="utf-8").splitlines()[0])
    question_path = tmp_path / "questions.jsonl"
    question_path.write_text(json.dumps(question) + "\n", encoding="utf-8")
    text = "Compared with controls, veterans with HIV had greater extracellular volume and lower right-heart function, suggesting fibrosis and subclinical dysfunction [PMID:41791688]."
    digest = hashlib.sha256(text.encode()).hexdigest()
    raw = {
        "split": "development",
        "questions_sha256": hashlib.sha256(question_path.read_bytes()).hexdigest(),
        "answer_source": "production_agent_graph", "model": "fixture", "code_commit": "fixture",
        "working_tree_dirty": False, "runtime_config": {}, "summary": {},
        "results": [{
            "id": question["id"], "query": question["question"], "answer": text,
            "answer_sha256": digest, "retrieved_chunk_ids": ["pubmed:41791688:0"],
            "faithful": True, "latency_seconds": 1,
        }],
    }
    review = {
        "id": question["id"], "answer_sha256": digest, "curator_resolution": "Fixture decision",
        "assessment": {
            "claim_mappings": [{"claim_id": "C1", "citation_chunk_ids": ["pubmed:41791688:0"]}],
            "abstained": False, "boundary_acknowledged": False, "forbidden_claims_present": False,
            "unsupported_material_claim_count": 0, "missing_required_qualifier_count": 0,
        },
    }
    return question_path, tmp_path / "raw.json", tmp_path / "review.jsonl", raw, review


def _write(saved_inputs):
    question_path, raw_path, review_path, raw, review = saved_inputs
    raw_path.write_text(json.dumps(raw), encoding="utf-8")
    review_path.write_text(json.dumps(review) + "\n", encoding="utf-8")
    return question_path, raw_path, review_path


def test_saved_score_needs_no_corpus_or_model(saved_inputs):
    report = recompute_saved_run(*_write(saved_inputs))
    assert report["summary"]["answers"]["strict_pass"] == 1
    assert report["summary"]["retrieval_answerable_at_5"]["all_required_found"] == 1


@pytest.mark.parametrize("defect", ["stale_review", "changed_answer", "missing_retrieval", "missing_row"])
def test_saved_score_rejects_mismatched_inputs(saved_inputs, defect):
    raw, review = saved_inputs[-2:]
    if defect == "stale_review":
        review["answer_sha256"] = "old answer hash"
    elif defect == "changed_answer":
        raw["results"][0]["answer"] += " Unsupported addition."
    elif defect == "missing_retrieval":
        raw["results"][0]["retrieved_chunk_ids"] = []
    elif defect == "missing_row":
        raw["results"] = []
    with pytest.raises(ValueError):
        recompute_saved_run(*_write(saved_inputs))
