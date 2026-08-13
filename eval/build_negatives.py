"""Build the four-tier negative taxonomy (PROJECT_PLAN.md §7 Phase 0, step 2).

Zero API calls. Built only over the numeric-gradable split (eval/data/answerable.jsonl)
because the leak check needs a single gold figure to search for.

    N1 — same-document evidence ablation (60%): correct filing, evidence page(s) removed.
    N2 — temporal mismatch (25%): a different fiscal year's 10-K, same company.
    N0 — wrong company (sanity floor, reported separately, never the headline).
    N3 — false premise (15%): hand-written, not generated here — see
         eval/data/n3_false_premise.jsonl and the note at the bottom of this file.

Each N1/N2 item is dropped if the gold figure is found anywhere in the document's
remaining text (the "gold-figure leak check") — logged so the drop count is auditable.

Usage:
    uv run python main.py build-negatives
"""

from __future__ import annotations

import hashlib
import json
import random
import re
import sys
from collections import defaultdict
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdfs"
ANSWERABLE_PATH = DATA_DIR / "answerable.jsonl"
DOC_INFO_PATH = DATA_DIR / "raw" / "financebench_document_information.jsonl"
OUT_PATH = DATA_DIR / "negatives.jsonl"
MANIFEST_PATH = DATA_DIR / "negatives_manifest.json"
N3_PATH = DATA_DIR / "n3_false_premise.jsonl"

SEED = 1337


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


MIN_SIGNIFICANT_DIGITS = 3  # below this, a bare number is too generic (years, page refs, etc.)


def gold_figure_variants(gold_value: float) -> list[str]:
    """Deterministic string forms of the gold figure to search for verbatim in text.

    Not exhaustive scale-normalization (that lives in normalize.py for grading) — this
    only needs to catch the obvious case where the same number reappears in prose
    elsewhere in the filing (e.g. the statement figure repeated in MD&A). Variants with
    fewer than MIN_SIGNIFICANT_DIGITS digits are dropped: e.g. a gold value of 12.4
    would otherwise produce the variant "12", which collides with page numbers, years,
    and unrelated dollar figures under naive substring search.
    """
    variants = set()
    for v in (gold_value, round(gold_value), round(gold_value, 2)):
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
        variants.add(s)
        variants.add(s.replace(",", ""))
        s_int = f"{v:,.0f}"
        variants.add(s_int)
        variants.add(s_int.replace(",", ""))
    return [
        v
        for v in variants
        if v and v not in ("0", "-0") and len(re.sub(r"[^0-9]", "", v)) >= MIN_SIGNIFICANT_DIGITS
    ]


def page_texts(pdf_path: Path) -> list[str]:
    doc = pymupdf.open(pdf_path)
    try:
        return [doc.load_page(i).get_text() for i in range(doc.page_count)]
    finally:
        doc.close()


def leaks_elsewhere(gold_value: float, pages: list[str], excluded_pages: set[int]) -> bool:
    variants = gold_figure_variants(gold_value)
    if not variants:
        # Gold figure has too few significant digits to search for reliably (e.g. a
        # small percentage). Cannot rule out a leak, so err toward not dropping —
        # this is the accepted trade-off documented in PROJECT_PLAN.md §7.
        return False
    patterns = [re.compile(rf"(?<!\d){re.escape(v)}(?!\d)") for v in variants]
    for i, text in enumerate(pages):
        if i in excluded_pages:
            continue
        if any(p.search(text) for p in patterns):
            return True
    return False


def build_n1(records: list[dict]) -> tuple[list[dict], int]:
    """Same-document evidence ablation: correct filing, evidence page(s) removed."""
    kept, dropped = [], 0
    for r in records:
        pdf_path = PDF_DIR / f"{r['doc_name']}.pdf"
        if not pdf_path.exists():
            continue
        evidence_pages = {e["evidence_page_num"] for e in r["evidence"]}
        pages = page_texts(pdf_path)
        if leaks_elsewhere(r["gold_value"], pages, evidence_pages):
            dropped += 1
            continue
        kept.append(
            {
                "tier": "N1",
                "financebench_id": r["financebench_id"],
                "question": r["question"],
                "gold_value": r["gold_value"],
                "gold_is_percent": r["gold_is_percent"],
                "served_doc_name": r["doc_name"],
                "ablated_pages": sorted(evidence_pages),
                "expected": "unanswerable",
            }
        )
    return kept, dropped


def build_n2(records: list[dict], doc_info: dict[str, dict]) -> tuple[list[dict], int]:
    """Temporal mismatch: serve a different fiscal year's 10-K of the same company."""
    by_company: dict[str, list[str]] = defaultdict(list)
    for name, info in doc_info.items():
        by_company[info["company"]].append(name)

    rng = random.Random(SEED)
    kept, dropped = [], 0
    for r in records:
        candidates = [
            d for d in by_company.get(r["company"], []) if d != r["doc_name"]
        ]
        if not candidates:
            continue
        served_doc = rng.choice(sorted(candidates))
        pdf_path = PDF_DIR / f"{served_doc}.pdf"
        if not pdf_path.exists():
            continue
        pages = page_texts(pdf_path)
        if leaks_elsewhere(r["gold_value"], pages, excluded_pages=set()):
            dropped += 1
            continue
        kept.append(
            {
                "tier": "N2",
                "financebench_id": r["financebench_id"],
                "question": r["question"],
                "gold_value": r["gold_value"],
                "gold_is_percent": r["gold_is_percent"],
                "served_doc_name": served_doc,
                "correct_doc_name": r["doc_name"],
                "served_doc_period": doc_info[served_doc]["doc_period"],
                "correct_doc_period": doc_info[r["doc_name"]]["doc_period"],
                "expected": "unanswerable",
            }
        )
    return kept, dropped


def build_n0(records: list[dict], doc_info: dict[str, dict]) -> list[dict]:
    """Wrong company — sanity floor, demoted, reported separately from N1-N3."""
    companies = sorted({info["company"] for info in doc_info.values()})
    rng = random.Random(SEED)
    items = []
    for r in records:
        other_companies = [c for c in companies if c != r["company"]]
        wrong_company = rng.choice(other_companies)
        wrong_docs = [n for n, i in doc_info.items() if i["company"] == wrong_company]
        served_doc = rng.choice(sorted(wrong_docs))
        items.append(
            {
                "tier": "N0",
                "financebench_id": r["financebench_id"],
                "question": r["question"],
                "gold_value": r["gold_value"],
                "gold_is_percent": r["gold_is_percent"],
                "served_doc_name": served_doc,
                "correct_doc_name": r["doc_name"],
                "expected": "unanswerable",
                "note": "sanity floor only — measures topic mismatch, not evidence "
                "insufficiency; never report as the headline (PROJECT_PLAN.md §7)",
            }
        )
    return items


def main() -> None:
    if not ANSWERABLE_PATH.exists():
        print("run `uv run python main.py build-dataset` first", file=sys.stderr)
        raise SystemExit(1)

    records = load_jsonl(ANSWERABLE_PATH)
    doc_info = {d["doc_name"]: d for d in load_jsonl(DOC_INFO_PATH)}

    have_pdfs = sum(1 for r in records if (PDF_DIR / f"{r['doc_name']}.pdf").exists())
    print(f"{have_pdfs}/{len(records)} source PDFs present in {PDF_DIR}", file=sys.stderr)
    if have_pdfs == 0:
        print(
            "no PDFs downloaded — run `uv run python main.py build-dataset` "
            "without --skip-pdfs first",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("building N1 (same-document evidence ablation)...", file=sys.stderr)
    n1, n1_dropped = build_n1(records)
    print(f"  N1: {len(n1)} kept, {n1_dropped} dropped (gold figure leaked elsewhere)", file=sys.stderr)

    print("building N2 (temporal mismatch)...", file=sys.stderr)
    n2, n2_dropped = build_n2(records, doc_info)
    print(f"  N2: {len(n2)} kept, {n2_dropped} dropped (gold figure leaked elsewhere)", file=sys.stderr)

    print("building N0 (wrong company, sanity floor)...", file=sys.stderr)
    n0 = build_n0(records, doc_info)
    print(f"  N0: {len(n0)} items", file=sys.stderr)

    n3 = load_jsonl(N3_PATH) if N3_PATH.exists() else []
    if not n3:
        print(
            f"  N3: 0 items — {N3_PATH} does not exist yet. N3 is hand-written "
            "(~30 false-premise questions, PROJECT_PLAN.md §7 step 2) and is not "
            "generated by this script.",
            file=sys.stderr,
        )

    all_negatives = n1 + n2 + n0 + n3
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("w", encoding="utf-8") as f:
        for item in all_negatives:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    manifest = {
        "counts": {"N1": len(n1), "N2": len(n2), "N0": len(n0), "N3": len(n3)},
        "dropped_gold_figure_leak": {"N1": n1_dropped, "N2": n2_dropped},
        "total": len(all_negatives),
        "negatives_jsonl_sha256": hashlib.sha256(OUT_PATH.read_bytes()).hexdigest(),
        "note": "N0 is a sanity floor, reported separately, never the headline metric.",
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print(f"wrote {OUT_PATH} ({len(all_negatives)} total)", file=sys.stderr)
    print(f"wrote {MANIFEST_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
