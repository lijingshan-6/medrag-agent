import pytest

from medrag.agent import llms
from medrag.config import ollama_base_url


@pytest.mark.parametrize("value, expected", [
    ("0.0.0.0:11434", "http://127.0.0.1:11434"),
    ("http://0.0.0.0:11434/", "http://127.0.0.1:11434"),
    ("[::]:11434", "http://[::1]:11434"),
    ("localhost:11434", "http://localhost:11434"),
    ("http://ollama:11434/", "http://ollama:11434"),
    ("https://remote.example:443/proxy/", "https://remote.example:443/proxy"),
])
def test_ollama_client_address(monkeypatch, value, expected):
    monkeypatch.setenv("OLLAMA_HOST", value)
    assert ollama_base_url() == expected


def test_ollama_uses_container_host(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434/")
    model = llms.make_llm_fast()
    assert model.base_url == "http://ollama:11434"
    assert model.model == "qwen3.5:9b"
    assert model.client_kwargs["timeout"] == 60.0


def test_ollama_review_uses_bounded_reasoning(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")

    model = llms.make_llm_think(reasoning=True)

    assert model.reasoning is True
    assert model.num_ctx == 8192
    assert model.num_predict == 4096
    assert model.temperature == 1.0


def test_unknown_backend_fails_before_creating_client(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "typo")
    with pytest.raises(ValueError, match="LLM_BACKEND"):
        llms.make_llm_fast()


def test_cloud_request_is_bounded(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "mimo")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://example.invalid/v1")
    monkeypatch.setenv("OPENAI_API_KEY", "unit-test-placeholder")
    model = llms.make_llm_fast()
    assert model.request_timeout == 60.0
    assert model.max_retries == 1


def test_structured_review_requests_json_without_consuming_reasoning_budget(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    model = llms.make_llm_think(structured=True)
    assert model.format == "json"
    assert model.reasoning is False
    assert llms.make_llm_fast().format is None
