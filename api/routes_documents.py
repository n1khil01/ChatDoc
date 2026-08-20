from __future__ import annotations

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile, status
from fastapi.responses import StreamingResponse

from api.csrf import verify_csrf
from api.deps import get_current_user_id
from api.documents_repo import (
    create_pending_document,
    delete_document,
    get_document,
    list_documents,
)
from api.jobs_repo import MAX_QUEUE_DEPTH, enqueue_job, queue_depth
from api.schemas_documents import DocumentResponse
from api.storage import get_storage
from api.worker import MAX_FILE_SIZE_BYTES

router = APIRouter(prefix="/documents", tags=["documents"])


def _to_response(row) -> DocumentResponse:
    return DocumentResponse(
        id=row.id,
        doc_name=row.doc_name,
        display_name=row.display_name,
        page_count=row.page_count,
        status=row.status,
        error=row.error,
        stage=row.stage,
        stage_current=row.stage_current,
        stage_total=row.stage_total,
        chunk_count=row.chunk_count,
        created_at=row.created_at,
    )


@router.get("", response_model=list[DocumentResponse])
def list_docs(user_id: int = Depends(get_current_user_id)):
    return [_to_response(r) for r in list_documents(user_id)]


@router.get("/{document_id}", response_model=DocumentResponse)
def get_doc(document_id: int, user_id: int = Depends(get_current_user_id)):
    row = get_document(user_id, document_id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    return _to_response(row)


@router.post("", response_model=DocumentResponse, status_code=status.HTTP_202_ACCEPTED)
async def upload_doc(
    request: Request,
    file: UploadFile,
    user_id: int = Depends(get_current_user_id),
):
    verify_csrf(request)

    if file.content_type != "application/pdf":
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "only application/pdf is accepted")

    # Backpressure: the `jobs` table now queues uploads rather than rejecting a second
    # concurrent one outright (Phase 4 crash-safe queue replaces Phase 3's single in-process
    # lock), but an unbounded queue on a free-tier box is its own failure mode.
    if queue_depth() >= MAX_QUEUE_DEPTH:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "ingest queue is full, try again shortly",
        )

    contents = await file.read()
    if len(contents) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"file exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)}MB limit",
        )

    doc_key = f"user{user_id}_{uuid.uuid4().hex[:12]}"
    storage_key = f"{doc_key}.pdf"
    get_storage().save(storage_key, contents)

    display_name = file.filename or "document.pdf"
    document_id = create_pending_document(user_id, doc_key, display_name, storage_key)
    enqueue_job(document_id, doc_key, storage_key)

    row = get_document(user_id, document_id)
    return _to_response(row)


@router.get("/{document_id}/file")
def get_doc_file(document_id: int, user_id: int = Depends(get_current_user_id)):
    row = get_document(user_id, document_id)
    if row is None or row.status != "ready":
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")
    # Proxy-streamed rather than a redirect to a presigned R2 URL: keeps the ownership
    # check above as the only gate a client ever has to satisfy, and keeps pdf.js's fetch
    # (see web/src/lib/api.ts's documentFileUrl) pointed at the API's own origin either way,
    # local disk or R2, with no client-side branching on storage backend.
    #
    # stream() is a generator -- calling it never runs its body, so a missing-key check has
    # to happen explicitly here rather than via try/except around the call, or a StorageError
    # would surface mid-response (inside StreamingResponse's iteration) instead of as a
    # clean 404.
    storage = get_storage()
    if not storage.exists(row.source_path):
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document file missing in storage")
    return StreamingResponse(storage.stream(row.source_path), media_type="application/pdf")


@router.delete("/{document_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_doc(document_id: int, request: Request, user_id: int = Depends(get_current_user_id)):
    verify_csrf(request)
    row = get_document(user_id, document_id)
    deleted = delete_document(user_id, document_id)
    if not deleted:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "document not found")

    # No explicit cancellation needed even if this document is mid-ingest: jobs.document_id
    # cascade-deletes with the row above, so the worker's completion/failure writes just
    # affect zero rows once it gets there -- see api/worker.py's process_job docstring.
    get_storage().delete(row.source_path)
