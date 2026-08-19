"""Retrieval + prompt assembly for the live /query endpoint.

Thin glue over the Phase 1 hybrid retriever and Phase 2 prompt template -- no new
retrieval or gating logic lives here, only the wiring needed to call them per-request.
"""

from __future__ import annotations

from dataclasses import dataclass

from eval.prompt import build_prompt
from ingest.db import get_conn
from ingest.retrieval import RetrievedChunk, hybrid_retrieve_scored


@dataclass
class RetrievalContext:
    scored: list[tuple[RetrievedChunk, float]]

    @property
    def chunks(self) -> list[RetrievedChunk]:
        return [c for c, _ in self.scored]

    @property
    def chunk_ids(self) -> set[int]:
        return {c.chunk_id for c in self.chunks}

    @property
    def text_by_id(self) -> dict[int, str]:
        return {c.chunk_id: c.text for c in self.chunks}

    @property
    def scale_by_id(self) -> dict[int, str | None]:
        return {c.chunk_id: c.unit_scale for c in self.chunks}

    @property
    def top1_top2_scores(self) -> tuple[float, float | None]:
        if not self.scored:
            return (0.0, None)
        top1 = self.scored[0][1]
        top2 = self.scored[1][1] if len(self.scored) > 1 else None
        return (top1, top2)


def retrieve(document_id: int, question: str) -> RetrievalContext:
    with get_conn() as conn:
        scored = hybrid_retrieve_scored(conn, document_id, question)
    return RetrievalContext(scored=scored)


def build_query_prompt(question: str, ctx: RetrievalContext) -> str:
    return build_prompt(question, ctx.chunks)
