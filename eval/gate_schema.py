"""Structured-output schema for the grounding gate (PROJECT_PLAN.md §7 Phase 2, step 1).

The model must fill `sufficient` and `operands` as typed fields, not phrases a parser hopes
to find. Derived answers (growth rates, ratios) are verified through their `operands` rather
than the final `value`, per the plan -- arithmetic is checkable without a second model call.
"""

from __future__ import annotations

from pydantic import BaseModel


class Operand(BaseModel):
    value: float
    citation_chunk_id: int


class Citation(BaseModel):
    chunk_id: int
    quote: str


class GateAnswer(BaseModel):
    sufficient: bool
    value: float | None = None
    unit: str | None = None
    scale: str | None = None
    operands: list[Operand] = []
    answer_text: str | None = None
    citations: list[Citation] = []
