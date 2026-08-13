"""Pull FinanceBench (PatronusAI/financebench, CC-BY-NC-4.0), classify each question
as numeric-gradable or not, and freeze the result to eval/data/answerable.jsonl.

Zero API calls. Source: https://github.com/patronus-ai/financebench

Usage:
    uv run python eval/build_dataset.py
    uv run python eval/build_dataset.py --skip-pdfs   # metadata only, no PDF download
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx
import pymupdf as fitz

from eval.normalize import normalize_numeric_answer

REPO_RAW = "https://raw.githubusercontent.com/patronus-ai/financebench/main"
QUESTIONS_URL = f"{REPO_RAW}/data/financebench_open_source.jsonl"
DOC_INFO_URL = f"{REPO_RAW}/data/financebench_document_information.jsonl"
PDF_URL_TMPL = f"{REPO_RAW}/pdfs/{{doc_name}}.pdf"

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PDF_DIR = ROOT / "pdfs"
OUT_PATH = DATA_DIR / "answerable.jsonl"
MANIFEST_PATH = DATA_DIR / "manifest.json"

KNOWN_OFF_BY_ONE_CHECK_ID = "financebench_id_03029"  # 3M_2018_10K, evidence_page_num=59


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url, follow_redirects=True, timeout=60)
    resp.raise_for_status()
    return resp.content


def load_jsonl(raw: bytes) -> list[dict]:
    return [json.loads(line) for line in raw.decode("utf-8").splitlines() if line.strip()]


def download_pdfs(client: httpx.Client, doc_names: set[str]) -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    for i, doc_name in enumerate(sorted(doc_names), 1):
        dest = PDF_DIR / f"{doc_name}.pdf"
        if dest.exists():
            continue
        url = PDF_URL_TMPL.format(doc_name=doc_name)
        print(f"  [{i}/{len(doc_names)}] downloading {doc_name}.pdf", file=sys.stderr)
        dest.write_bytes(fetch(client, url))


def assert_evidence_page_off_by_one(records: list[dict]) -> None:
    """PROJECT_PLAN.md §15: assert the zero-index/one-index mismatch before trusting
    anything downstream. financebench_id_03029 asks for 3M FY2018 capex; the gold
    evidence_page_num is 59 (zero-indexed) against the cash flow statement. Confirm the
    PDF page at pdf.js's 1-indexed page 60 (fitz's 0-indexed page 59) actually contains
    the expected line item text.
    """
    rec = next((r for r in records if r["financebench_id"] == KNOWN_OFF_BY_ONE_CHECK_ID), None)
    if rec is None:
        raise AssertionError(f"known check item {KNOWN_OFF_BY_ONE_CHECK_ID} not found in dataset")

    evidence = rec["evidence"][0]
    page_num = evidence["evidence_page_num"]
    pdf_path = PDF_DIR / f"{rec['doc_name']}.pdf"
    if not pdf_path.exists():
        print(
            f"  (skipping off-by-one PDF assertion — {pdf_path.name} not downloaded; "
            "run without --skip-pdfs to verify)",
            file=sys.stderr,
        )
        return

    doc = fitz.open(pdf_path)
    try:
        # evidence_page_num is zero-indexed; fitz's .load_page is also zero-indexed,
        # so no adjustment is needed here — the adjustment is only required when
        # driving a 1-indexed viewer such as pdf.js.
        page_text = doc.load_page(page_num).get_text()
    finally:
        doc.close()

    needle = "Purchases of property"
    if needle not in page_text:
        raise AssertionError(
            f"off-by-one check failed: page {page_num} (zero-indexed) of "
            f"{rec['doc_name']}.pdf does not contain {needle!r}. "
            f"evidence_page_num is documented as zero-indexed in PROJECT_PLAN.md §6 — "
            f"verify that assumption before trusting recall@k."
        )
    print(
        f"  off-by-one check OK: evidence_page_num={page_num} (zero-indexed) "
        f"contains the expected line item; pdf.js callers must add 1.",
        file=sys.stderr,
    )


def classify(records: list[dict]) -> tuple[list[dict], list[dict]]:
    gradable, not_gradable = [], []
    for r in records:
        parsed = normalize_numeric_answer(r["answer"])
        if parsed is None:
            not_gradable.append(r)
            continue
        gradable.append(
            {
                **r,
                "gold_value": parsed.value,
                "gold_is_percent": parsed.is_percent,
            }
        )
    return gradable, not_gradable


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-pdfs", action="store_true", help="skip PDF download")
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    with httpx.Client() as client:
        print("fetching financebench_open_source.jsonl ...", file=sys.stderr)
        questions_raw = fetch(client, QUESTIONS_URL)
        print("fetching financebench_document_information.jsonl ...", file=sys.stderr)
        doc_info_raw = fetch(client, DOC_INFO_URL)

        (RAW_DIR / "financebench_open_source.jsonl").write_bytes(questions_raw)
        (RAW_DIR / "financebench_document_information.jsonl").write_bytes(doc_info_raw)

        questions = load_jsonl(questions_raw)
        doc_info = {d["doc_name"]: d for d in load_jsonl(doc_info_raw)}

        doc_names = {r["doc_name"] for r in questions}
        if not args.skip_pdfs:
            print(f"downloading {len(doc_names)} PDFs (cached, skips existing) ...", file=sys.stderr)
            download_pdfs(client, doc_names)
        else:
            print("--skip-pdfs set: not downloading PDFs", file=sys.stderr)

    for r in questions:
        info = doc_info.get(r["doc_name"], {})
        r["doc_period"] = info.get("doc_period")
        r["doc_link"] = info.get("doc_link")

    assert_evidence_page_off_by_one(questions)

    gradable, not_gradable = classify(questions)

    print(
        f"classified {len(gradable)}/{len(questions)} as numeric-gradable "
        f"({len(not_gradable)} not gradable)",
        file=sys.stderr,
    )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for r in gradable:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    not_gradable_path = DATA_DIR / "not_gradable.jsonl"
    with not_gradable_path.open("w", encoding="utf-8") as f:
        for r in not_gradable:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    frozen_bytes = OUT_PATH.read_bytes()
    manifest = {
        "source": "PatronusAI/financebench",
        "source_commit_ref": "main",
        "license": "CC-BY-NC-4.0",
        "built_at": datetime.now(timezone.utc).isoformat(),
        "questions_total": len(questions),
        "numeric_gradable": len(gradable),
        "not_gradable": len(not_gradable),
        "answerable_jsonl_sha256": sha256_bytes(frozen_bytes),
        "questions_source_sha256": sha256_bytes(questions_raw),
        "doc_info_source_sha256": sha256_bytes(doc_info_raw),
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {OUT_PATH} ({len(gradable)} rows)", file=sys.stderr)
    print(f"wrote {not_gradable_path} ({len(not_gradable)} rows)", file=sys.stderr)
    print(f"wrote {MANIFEST_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
