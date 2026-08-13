"""Naive-baseline retrieval eval: recall@k and MRR against evidence_page_num.

Zero API calls. This is deliberately the *worst* retriever we will ever ship — fixed-size
chunks, dense-only, no table awareness — so that Phase 1's table-aware/hybrid retriever has
a documented before/after (PROJECT_PLAN.md §9 Challenge 2). Embeddings here use a simple
TF-IDF-style bag-of-words cosine score rather than a real embedding model: Phase 0 has no
ONNX pipeline yet (that lands in Phase 1), and recall@k / MRR only need *a* ranking signal
to produce a baseline number, not the final model.

Usage:
    uv run python main.py retrieval-eval
"""

from __future__ import annotations

import json
import math
import re
import sys
from collections import Counter
from pathlib import Path

import pymupdf

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
PDF_DIR = ROOT / "pdfs"
ANSWERABLE_PATH = DATA_DIR / "answerable.jsonl"
OUT_PATH = DATA_DIR / "retrieval_eval_baseline.json"

CHUNK_SIZE_TOKENS = 512
CHUNK_OVERLAP_TOKENS = 64
K_VALUES = (1, 5, 10)

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class Chunk:
    __slots__ = ("text", "page_num", "tokens")

    def __init__(self, text: str, page_num: int):
        self.text = text
        self.page_num = page_num
        self.tokens = tokenize(text)


def chunk_document(pdf_path: Path, excluded_pages: frozenset[int] = frozenset()) -> list[Chunk]:
    """Fixed 512-token chunks with 64-token overlap, naive — no table awareness, no
    section headers, page number carried only as "whichever page the chunk starts on".
    This crude page attribution is itself part of the baseline's weakness.

    `excluded_pages` drops those pages' words before chunking (not after), so an
    ablated page genuinely never appears in any chunk's text — used to build the N1
    same-document evidence-ablation negatives in eval/runner.py.
    """
    doc = pymupdf.open(pdf_path)
    try:
        page_word_spans: list[tuple[int, list[str]]] = []
        for page_num in range(doc.page_count):
            if page_num in excluded_pages:
                continue
            words = doc.load_page(page_num).get_text().split()
            page_word_spans.append((page_num, words))
    finally:
        doc.close()

    all_words: list[tuple[str, int]] = []  # (word, source_page)
    for page_num, words in page_word_spans:
        all_words.extend((w, page_num) for w in words)

    chunks = []
    step = CHUNK_SIZE_TOKENS - CHUNK_OVERLAP_TOKENS
    for start in range(0, len(all_words), step):
        window = all_words[start : start + CHUNK_SIZE_TOKENS]
        if not window:
            continue
        text = " ".join(w for w, _ in window)
        page_num = window[0][1]
        chunks.append(Chunk(text, page_num))
        if start + CHUNK_SIZE_TOKENS >= len(all_words):
            break
    return chunks


def build_idf(chunks: list[Chunk]) -> dict[str, float]:
    df: Counter[str] = Counter()
    for c in chunks:
        df.update(set(c.tokens))
    n = len(chunks)
    return {term: math.log((n + 1) / (freq + 1)) + 1 for term, freq in df.items()}


def score_chunk(query_tokens: list[str], chunk: Chunk, idf: dict[str, float]) -> float:
    chunk_counts = Counter(chunk.tokens)
    query_counts = Counter(query_tokens)
    score = 0.0
    for term, qf in query_counts.items():
        if term in chunk_counts:
            score += qf * chunk_counts[term] * idf.get(term, 0.0)
    norm = math.sqrt(sum(v * v for v in chunk_counts.values())) or 1.0
    return score / norm


def rank_chunks(question: str, chunks: list[Chunk], idf: dict[str, float]) -> list[Chunk]:
    q_tokens = tokenize(question)
    scored = [(score_chunk(q_tokens, c, idf), i, c) for i, c in enumerate(chunks)]
    scored.sort(key=lambda t: (-t[0], t[1]))
    return [c for _, _, c in scored]


def evaluate_question(record: dict, chunks: list[Chunk], idf: dict[str, float]) -> dict:
    evidence_pages = {e["evidence_page_num"] for e in record["evidence"]}
    ranked = rank_chunks(record["question"], chunks, idf)

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

    results = []
    doc_cache: dict[str, tuple[list[Chunk], dict[str, float]]] = {}

    for i, record in enumerate(records, 1):
        doc_name = record["doc_name"]
        pdf_path = PDF_DIR / f"{doc_name}.pdf"
        if not pdf_path.exists():
            print(f"  skipping {record['financebench_id']}: {pdf_path.name} not found", file=sys.stderr)
            continue

        if doc_name not in doc_cache:
            chunks = chunk_document(pdf_path)
            idf = build_idf(chunks)
            doc_cache[doc_name] = (chunks, idf)
        chunks, idf = doc_cache[doc_name]

        print(f"  [{i}/{len(records)}] {record['financebench_id']} ({doc_name}, {len(chunks)} chunks)", file=sys.stderr)
        results.append(evaluate_question(record, chunks, idf))

    n = len(results)
    summary = {
        "n_questions": n,
        "chunking": {"size_tokens": CHUNK_SIZE_TOKENS, "overlap_tokens": CHUNK_OVERLAP_TOKENS},
        "retriever": "naive dense-only baseline (TF-IDF cosine stand-in, no table awareness)",
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

    print("\n--- baseline retrieval eval ---", file=sys.stderr)
    for k, v in summary.items():
        print(f"  {k}: {v}", file=sys.stderr)
    print(f"\nwrote {OUT_PATH}", file=sys.stderr)


if __name__ == "__main__":
    main()
