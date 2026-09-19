"""Offline tests never load local secrets or use production checkpoints."""
import os
import sys
import tempfile

import pytest

_test_data = tempfile.TemporaryDirectory(prefix="veritasmed-tests-")
os.environ["MEDRAG_DATA_DIR"] = _test_data.name
os.environ["PYTHON_DOTENV_DISABLED"] = "1"


def pytest_sessionfinish(session, exitstatus):
    graph = sys.modules.get("medrag.agent.graph")
    if graph is not None:
        graph._conn.close()
    _test_data.cleanup()


def pytest_addoption(parser):
    parser.addoption("--run-live", action="store_true", help="Run tests using real Qdrant and LLM services")


def pytest_ignore_collect(collection_path, config):
    if collection_path.name == "test_integration.py" and not config.getoption("--run-live"):
        return True
    return None


def pytest_configure(config):
    if config.getoption("--run-live"):
        os.environ.pop("PYTHON_DOTENV_DISABLED", None)


@pytest.fixture(autouse=True)
def offline_environment(monkeypatch, request):
    if request.config.getoption("--run-live"):
        return
    for key in ("OPENAI_API_KEY", "OPENAI_BASE_URL", "OPENAI_API_BASE", "JUDGE_API_KEY", "JUDGE_BASE_URL"):
        monkeypatch.delenv(key, raising=False)
