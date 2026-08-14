"""Phase 1 retrieval eval: table-aware ingestion + hybrid (dense+sparse RRF) + MiniLM rerank,
measured against the same frozen `evidence_page_num` set as the Phase 0 naive baseline.

Zero API calls. Ingests each distinct FinanceBench PDF referenced in eval/data/answerable.jsonl
into the local docker-compose Postgres (see docker-compose.yml, db/schema.sql), then for every
question runs ingest.retrieval.hybrid_retrieve scoped to that question's document and checks
whether the evidence page appears in the top-1/5/10 reranked chunks.

Usage:
    docker compose up -d   # once
    uv run python main.py retrieval-eval-v2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from ingest.db import apply_schema, get_conn, get_ingested_document_id
from ingest.pipeline import ingest_pdf
from ingest.retrieval import hybrid_retrieve

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdfs"
ANSWERABLE_PATH = DATA_DIR / "answerable.jsonl"
OUT_PATH = DATA_DIR / "retrieval_eval_v2.json"
SCHEMA_PATH = ROOT.parent / "db" / "schema.sql"

K_VALUES = (1, 5, 10)
RERANK_TOP_K = 10  # matches max(K_VALUES); ranks beyond this are not distinguished


def evaluate_question(conn, record: dict, document_id: int) -> dict:
    evidence_pages = {e["evidence_page_num"] for e in record["evidence"]}
    ranked = hybrid_retrieve(conn, document_id, record["question"], rerank_top_k=RERANK_TOP_K)

    rank_of_first_hit = None
    for rank, chunk in enumerate(ranked, start=1):
        if chunk.page_num in evidence_pages:
            rank_of_first_hit = rank
            break

    recall_at_k = {k: (rank_of_first_hit is not None and rank_of_first_hit <= k) for k in K_VALUES}
    reciprocal_rank = 1.0 / rank_of_first_hit if rank_of_first_hit else 0.0
    return {
        "financebench_id": record["financebench_id"],
        "rank_of_first_hit": rank_of_first_hit,
        "reciprocal_rank": reciprocal_rank,
        **{f"recall_at_{k}": recall_at_k[k] for k in K_VALUES},
    }


def main() -> None:
    if not ANSWERABLE_PATH.exists():
        print("run `uv run python main.py build-dataset` first", file=sys.stderr)
        raise SystemExit(1)

    with ANSWERABLE_PATH.open(encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    # Sanity check the zero-indexing assumption on a known item before running anything else.
    known = next(r for r in records if r["financebench_id"] == "financebench_id_03029")
    assert known["evidence"][0]["evidence_page_num"] == 59, (
        "evidence_page_num indexing assumption changed -- expected page 59 (0-indexed) for "
        f"financebench_id_03029, got {known['evidence'][0]['evidence_page_num']}"
    )

    with get_conn() as conn:
        apply_schema(conn, SCHEMA_PATH)

        doc_ids: dict[str, int] = {}
        results = []

        for i, record in enumerate(records, 1):
            doc_name = record["doc_name"]
            pdf_path = PDF_DIR / f"{doc_name}.pdf"
            if not pdf_path.exists():
                print(f"  skipping {record['financebench_id']}: {pdf_path.name} not found", file=sys.stderr)
                continue

            if doc_name not in doc_ids:
                existing_id = get_ingested_document_id(conn, doc_name)
                if existing_id is not None:
                    print(f"  skipping {doc_name} (already ingested)", file=sys.stderr)
                    doc_ids[doc_name] = existing_id
                else:
                    print(f"  ingesting {doc_name} ...", file=sys.stderr)
                    doc_ids[doc_name] = ingest_pdf(conn, pdf_path)
            document_id = doc_ids[doc_name]

            print(f"  [{i}/{len(records)}] {record['financebench_id']} ({doc_name})", file=sys.stderr)
            results.append(evaluate_question(conn, record, document_id))

    n = len(results)
    summary = {
        "n_questions": n,
        "retriever": "table-aware chunking + hybrid (pgvector halfvec HNSW dense + tsvector sparse, RRF k=60) + MiniLM cross-encoder rerank",
        "mrr": sum(r["reciprocal_rank"] for r in results) / n if n else 0.0,
        **{
            f"recall_at_{k}": sum(1 for r in results if r[f"recall_at_{k}"]) / n if n else 0.0
            for k in K_VALUES
        },
    }

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(
        json.dumps({"summary": summary, "per_question": results}, indent=2),
        encoding="utf-8",
    )

    print("\n--- Phase 1 retrieval eval ---", file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
