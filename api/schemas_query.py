from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    document_id: int
    question: str = Field(min_length=1, max_length=2000)
