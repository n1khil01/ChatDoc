"""Cross-encoder rerank: cross-encoder/ms-marco-MiniLM-L6-v2, ONNX, CPU.

PROJECT_PLAN.md §5: bge-reranker-base is ~1.1GB fp32 and does not fit; MiniLM-L6 (22.7M params,
~25MB quantized) ships a pre-exported ONNX checkpoint and makes the same "cross-encoder
reranking" claim on hardware this project actually runs on.

fastembed's TextCrossEncoder exposes this exact checkpoint under the Xenova ONNX export
(Xenova/ms-marco-MiniLM-L-6-v2) -- same weights HF's sentence-transformers checkpoint uses,
re-exported to ONNX by Xenova/transformers.js, which is what has an actual downloadable ONNX
artifact (the raw cross-encoder/ms-marco-MiniLM-L6-v2 HF repo ships PyTorch weights only).
"""

from __future__ import annotations

from functools import lru_cache

from fastembed.rerank.cross_encoder import TextCrossEncoder

MODEL_NAME = "Xenova/ms-marco-MiniLM-L-6-v2"


@lru_cache(maxsize=1)
def _model() -> TextCrossEncoder:
    return TextCrossEncoder(model_name=MODEL_NAME, threads=1, providers=["CPUExecutionProvider"])


def rerank(query: str, documents: list[str]) -> list[float]:
    """Return a relevance score per document, same order as input."""
    if not documents:
        return []
    return list(_model().rerank(query, documents))
