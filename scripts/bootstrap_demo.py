"""Index the bundled demonstration summaries without touching research data."""
from __future__ import annotations

import argparse
from pathlib import Path

import sentence_transformers  # noqa: F401 -- native runtime must load before Qdrant

from medrag.config import get_qdrant_client
from medrag.demo import bootstrap
from medrag.index.embedder import BGEM3Embedder


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    embedder = BGEM3Embedder(device=args.device)
    client = get_qdrant_client()
    try:
        count = bootstrap(client, embedder, Path(__file__).resolve().parent.parent / "data/demo/corpus.jsonl")
        print(f"Ready: medrag_demo contains {count} authored demonstration summaries.")
        print("Start the API with QDRANT_COLLECTION=medrag_demo.")
    finally:
        client.close()


if __name__ == "__main__":
    main()
