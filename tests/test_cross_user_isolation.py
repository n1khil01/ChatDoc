"""Proves user B cannot read, list, delete, or query user A's document.

PROJECT_PLAN.md §7 Phase 3 requires this as the actual evidence behind the
`user_id` scoping claim -- a route that merely *looks* scoped (e.g. an
unscoped `WHERE id = %s` that happens to work in manual testing) would pass
every other test in this suite and still leak documents across accounts.

Ingestion itself is not exercised here (that's Phase 1's own test surface);
a document is inserted directly via api.documents_repo and marked 'ready' so
these tests isolate authorization from the PDF pipeline.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from api.documents_repo import create_pending_document, get_document, mark_document_ready


def _make_ready_document(user_id: int) -> int:
    doc_id = create_pending_document(
        user_id, f"isolation-test-{user_id}", "isolation-test.pdf", "/nonexistent/isolation-test.pdf"
    )
    mark_document_ready(doc_id, chunk_count=0)
    return doc_id


def test_user_b_cannot_get_user_a_document(register_user):
    client_a, csrf_a, user_a = register_user()
    client_b, csrf_b, user_b = register_user()
    doc_id = _make_ready_document(user_a)

    # Sanity: the owner can see it.
    own = client_a.get(f"/documents/{doc_id}")
    assert own.status_code == 200

    resp = client_b.get(f"/documents/{doc_id}")
    assert resp.status_code == 404


def test_user_b_document_list_excludes_user_a_document(register_user):
    client_a, csrf_a, user_a = register_user()
    client_b, csrf_b, user_b = register_user()
    doc_id = _make_ready_document(user_a)

    resp = client_b.get("/documents")
    assert resp.status_code == 200
    assert doc_id not in {d["id"] for d in resp.json()}


def test_user_b_cannot_fetch_user_a_document_file(register_user):
    client_a, csrf_a, user_a = register_user()
    client_b, csrf_b, user_b = register_user()
    doc_id = _make_ready_document(user_a)

    resp = client_b.get(f"/documents/{doc_id}/file")
    assert resp.status_code == 404


def test_user_b_cannot_delete_user_a_document(register_user):
    client_a, csrf_a, user_a = register_user()
    client_b, csrf_b, user_b = register_user()
    doc_id = _make_ready_document(user_a)

    resp = client_b.delete(f"/documents/{doc_id}", headers={"x-csrf-token": csrf_b})
    assert resp.status_code == 404

    # The document must still exist and still belong to user A.
    assert get_document(user_a, doc_id) is not None


def test_user_b_cannot_query_user_a_document(register_user):
    """/query looks up the document scoped by user_id *before* touching retrieval or
    Gemini, so cross-user access is rejected the same way as a plain GET -- this
    asserts that check actually runs on the query path too, not just document CRUD."""
    client_a, csrf_a, user_a = register_user()
    client_b, csrf_b, user_b = register_user()
    doc_id = _make_ready_document(user_a)

    resp = client_b.post(
        "/query",
        json={"document_id": doc_id, "question": "What was the revenue?"},
        headers={"x-csrf-token": csrf_b},
    )
    assert resp.status_code == 404


def test_user_a_can_still_query_own_ready_document_past_the_auth_check(register_user):
    """Companion to the negative case above: confirm the 404 in the cross-user tests
    is really about ownership and not some other reason every /query call 404s (e.g.
    a broken route). We don't run this one to completion (that needs Gemini + real
    chunks) -- reaching the retrieval step at all is the point, so the CSRF/ownership
    gate really was the thing blocking user B, not something else entirely."""
    client_a, csrf_a, user_a = register_user()
    doc_id = _make_ready_document(user_a)

    resp = client_a.post(
        "/query",
        json={"document_id": doc_id, "question": "What was the revenue?"},
        headers={"x-csrf-token": csrf_a},
    )
    # No chunks were ingested, so retrieval finds nothing and the endpoint refuses
    # via SSE (200, not 404) -- proof the 404s above were ownership checks, not a
    # blanket failure of the route.
    assert resp.status_code == 200
    assert "refusal" in resp.text
