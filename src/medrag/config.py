"""Project-wide configuration helpers (env vars, paths)."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from dotenv import load_dotenv

# src/medrag/config.py → repo root is two levels up
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_MODEL = "qwen3.5:9b"
load_dotenv(PROJECT_ROOT / ".env")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "medrag_text")


def load_project_env() -> None:
    """Load .env from repo root (idempotent)."""
    load_dotenv(PROJECT_ROOT / ".env")


def qdrant_url() -> str:
    """Qdrant REST base URL from QDRANT_URL env var."""
    load_project_env()
    return os.getenv("QDRANT_URL", DEFAULT_QDRANT_URL).rstrip("/")


@lru_cache(maxsize=1)
def get_qdrant_client():
    """One client per process; optional embedded mode for a small local demo."""
    from qdrant_client import QdrantClient

    load_project_env()
    path = os.getenv("QDRANT_PATH")
    if path:
        return QdrantClient(path=path, force_disable_check_same_thread=True)
    return QdrantClient(url=qdrant_url(), timeout=10)
