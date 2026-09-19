"""BGE-M3 embedder — dense + sparse via FlagEmbedding M3Embedder.

Import path bypasses FlagEmbedding.__init__ to avoid the decoder-only
reranker import chain that triggers STATUS_ACCESS_VIOLATION on Windows:

    from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder

This gives access to BGE-M3's native sparse (SPLADE-style lexical weights)
output, enabling true dense+sparse RRF in HybridRetriever.

sentence_transformers MUST be imported before this module is loaded so that
PyTorch initialises before qdrant_client's gRPC C++ runtime.  app.py handles
this at startup.

Device selection:
  BGEM3Embedder(device="auto")  → cuda if torch.cuda.is_available(), else cpu
  BGEM3Embedder(device="cuda")  → force GPU
  BGEM3Embedder(device="cpu")   → force CPU
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            logger.info("[embedder] CUDA available → gpu")
            return "cuda"
    except ImportError:
        pass
    logger.info("[embedder] CUDA not available → cpu")
    return "cpu"


class BGEM3Embedder:
    def __init__(
        self,
        device: Literal["auto", "cuda", "cpu"] = "cpu",
        use_fp16: bool = False,
    ):
        from FlagEmbedding.inference.embedder.encoder_only.m3 import M3Embedder

        resolved = _resolve_device(device)
        # M3Embedder expects a list of device strings
        devices = ["cuda:0" if resolved == "cuda" else "cpu"]
        self._model = M3Embedder(
            model_name_or_path="BAAI/bge-m3",
            use_fp16=(use_fp16 and resolved == "cuda"),
            devices=devices,
        )
        self._device = resolved
        logger.info("[embedder] BGEM3Embedder loaded on %s (fp16=%s)", resolved, use_fp16)

    def encode(
        self,
        texts: list[str],
        batch_size: int = 12,
        return_sparse: bool = False,
    ) -> dict:
        """Encode texts into dense (and optionally sparse) vectors.

        Returns:
            dict with keys:
              "dense"  — np.ndarray of shape (n, 1024), float32, L2-normalised
              "sparse" — list of {token_id_str: weight} dicts (only if return_sparse=True)
        """
        out = self._model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )

        result: dict = {"dense": np.array(out["dense_vecs"], dtype=np.float32)}

        if return_sparse:
            # lexical_weights is a list of {token_str: float} dicts
            result["sparse"] = out.get("lexical_weights", [{} for _ in texts])

        return result


__all__ = ["BGEM3Embedder"]
