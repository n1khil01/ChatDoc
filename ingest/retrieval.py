"""Hybrid retrieval: pgvector halfvec HNSW (dense, cosine) + tsvector/tsquery (sparse),
fused with Reciprocal Rank Fusion, then reranked with a MiniLM cross-encoder.

RRF: score(d) = sum over rankers of 1 / (k + rank_in_that_ranker), k=60 (standard default,
Cormack et al. 2009 -- no dataset-specific tuning justifies deviating from it here).
"""

from __future__ import annotations

from dataclasses import dataclass

from pgvector import HalfVector

from ingest.embeddings import embed_query
from ingest.reranker import rerank as rerank_fn

RRF_K = 60


@dataclass
class RetrievedChunk:
    chunk_id: int
    document_id: int
    page_num: int
    chunk_type: str
    text: str


def dense_search(conn, document_id: int, query: str, top_n: int) -> list[RetrievedChunk]:
    qvec = HalfVector(embed_query(query))
    rows = conn.execute(
        """
        SELECT id, document_id, page_num, chunk_type, text
        FROM chunks
        WHERE document_id = %s
        ORDER BY embedding <=> %s
        LIMIT %s
        """,
        (document_id, qvec, top_n),
    ).fetchall()
    return [RetrievedChunk(*r) for r in rows]


def sparse_search(conn, document_id: int, query: str, top_n: int) -> list[RetrievedChunk]:
    rows = conn.execute(
        """
        SELECT id, document_id, page_num, chunk_type, text
        FROM chunks
        WHERE document_id = %s AND tsv @@ plainto_tsquery('english', %s)
        ORDER BY ts_rank(tsv, plainto_tsquery('english', %s)) DESC
        LIMIT %s
        """,
        (document_id, query, query, top_n),
    ).fetchall()
    return [RetrievedChunk(*r) for r in rows]


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]], k: int = RRF_K
) -> list[tuple[RetrievedChunk, float]]:
    scores: dict[int, float] = {}
    chunks_by_id: dict[int, RetrievedChunk] = {}
    for ranked in ranked_lists:
        for rank, chunk in enumerate(ranked, start=1):
            scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (k + rank)
            chunks_by_id[chunk.chunk_id] = chunk
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(chunks_by_id[cid], score) for cid, score in fused]


def hybrid_retrieve(
    conn,
    document_id: int,
    query: str,
    dense_n: int = 30,
    sparse_n: int = 30,
    rrf_top_n: int = 20,
    rerank_top_k: int = 10,
) -> list[RetrievedChunk]:
    """Dense + sparse search -> RRF fusion -> cross-encoder rerank of the fused top-N,
    returning the top rerank_top_k chunks."""
    dense = dense_search(conn, document_id, query, dense_n)
    sparse = sparse_search(conn, document_id, query, sparse_n)
    fused = rrf_fuse([dense, sparse])[:rrf_top_n]
    if not fused:
        return []

    candidates = [c for c, _ in fused]
    scores = rerank_fn(query, [c.text for c in candidates])
    reranked = sorted(zip(candidates, scores), key=lambda t: -t[1])
    return [c for c, _ in reranked[:rerank_top_k]]
