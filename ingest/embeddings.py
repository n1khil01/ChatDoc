"""int8 ONNX bge-small-en-v1.5 embeddings via fastembed, CPU, single-threaded.

`intra_op_num_threads=1` per PROJECT_PLAN.md §7 Phase 1 memory rationale: this runs on a
memory-constrained box, so we trade thread parallelism for a predictable, small footprint
rather than importing torch (which alone costs ~250-350MB RSS).

fastembed ships BAAI/bge-small-en-v1.5 as a quantized ONNX export (384-dim) pulled from the
Qdrant HF mirror on first use and cached under ~/.cache/fastembed. No API calls, no torch.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
from fastembed import TextEmbedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"
DIM = 384


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME, threads=1, providers=["CPUExecutionProvider"])


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    vectors = list(_model().embed(texts))
    return [np.asarray(v, dtype=np.float32).tolist() for v in vectors]


def embed_query(text: str) -> list[float]:
    return embed_texts([text])[0]
