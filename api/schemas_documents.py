from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel

DocumentStatus = Literal["processing", "ready", "failed"]

# Pipeline steps, in the order ingest/pipeline.py emits them. The client renders these as a
# live checklist while status == "processing".
IngestStage = Literal["queued", "reading", "chunking", "embedding", "indexing", "done"]


class DocumentResponse(BaseModel):
    id: int
    doc_name: str
    display_name: str | None
    page_count: int
    status: DocumentStatus
    error: str | None
    stage: IngestStage | None
    stage_current: int
    stage_total: int
    chunk_count: int
    created_at: datetime
