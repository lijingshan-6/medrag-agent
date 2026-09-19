import pytest

from medrag.agent import llms


def test_ollama_uses_container_host(monkeypatch):
    monkeypatch.setenv("LLM_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_HOST", "http://ollama:11434/")
    model = llms.make_llm_fast()
    assert model.base_url == "http://ollama:11434"
    assert model.client_kwargs["timeout"] == 60.0


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
