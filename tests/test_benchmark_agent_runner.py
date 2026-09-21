from __future__ import annotations

from medrag.benchmark.agent_runner import run_agent_question
from medrag.retrieval.retriever import RetrievedChunk


class FakeAgent:
    def __init__(self) -> None:
        self.state = None
        self.config = None

    def invoke(self, state, *, config):
        self.state = state
        self.config = config
        return {
            **state,
            "answer": "Supported answer [PMID:1].",
            "citations": ["PMID:1"],
            "confidence": 0.8,
            "faithful": True,
            "faithfulness_issues": "",
            "iterations": 1,
            "regen_count": 0,
            "rewritten_queries": ["rewritten query"],
            "retrieved_chunks": [
                RetrievedChunk(
                    chunk_id="pubmed:1:0",
                    text="Passage",
                    score=0.9,
                    payload={"source": "pubmed", "doc_id": "1", "chunk_id": "pubmed:1:0", "text": "Passage"},
                )
            ],
        }


def test_agent_runner_captures_real_graph_result_with_isolated_thread() -> None:
    agent = FakeAgent()

    result = run_agent_question(agent, question_id="VMG-001", query="What was found?")

    assert agent.state["query"] == "What was found?"
    assert agent.config["configurable"]["thread_id"].startswith("benchmark-VMG-001-")
    assert result["answer"] == "Supported answer [PMID:1]."
    assert result["citations"] == ["PMID:1"]
    assert result["retrieved_chunk_ids"] == ["pubmed:1:0"]
    assert result["faithful"] is True
    assert result["iterations"] == 1
