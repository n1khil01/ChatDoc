"""Table-aware, page-anchored PDF chunking (PROJECT_PLAN.md §7 Phase 1).

Two chunk types come out of every page:
  * "table"  -- one atomic chunk per PyMuPDF-detected table, serialized to Markdown,
               never split across chunks, with scale/unit metadata sniffed from
               nearby footnote text and the table's bbox captured for citations.
  * "prose"  -- everything else on the page, chunked by paragraph with the nearest
               preceding section header prefixed onto each chunk.

Empty text layer (scanned PDF, no OCR) fails loudly per the plan's explicit scope decision --
this project does not silently ingest nothing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

PROSE_CHUNK_CHARS = 1600
PROSE_CHUNK_OVERLAP_CHARS = 200

_SCALE_RE = re.compile(
    r"in\s+(thousands|millions|billions)\b(?:\s*,?\s*except\s+([^.\n]{0,80}))?",
    re.IGNORECASE,
)
_CURRENCY_RE = re.compile(r"\b(USD|US\$|U\.S\.\s*dollars)\b", re.IGNORECASE)

_BOLD_FLAG = 1 << 4  # PyMuPDF span flag bit for bold


class EmptyTextLayerError(Exception):
    """Raised when a PDF page (or whole document) has no extractable text layer.

    Per PROJECT_PLAN.md scope decision: no OCR. A scanned page must fail loudly,
    not be silently ingested as an empty chunk set.
    """


def _clean(text: str) -> str:
    """Strip NUL bytes PyMuPDF occasionally extracts from malformed PDF text streams --
    Postgres text columns reject them outright (psycopg.DataError)."""
    return text.replace("\x00", "")


@dataclass
class Chunk:
    page_num: int  # 0-indexed, matches FinanceBench evidence_page_num
    chunk_type: str  # "table" | "prose"
    text: str
    section_header: str | None = None
    unit_scale: str | None = None
    unit_currency: str | None = None
    unit_note: str | None = None
    bbox: tuple[float, float, float, float] | None = None


@dataclass
class DocumentChunks:
    page_count: int
    chunks: list[Chunk] = field(default_factory=list)


def _detect_scale(text: str) -> tuple[str | None, str | None, str | None]:
    m = _SCALE_RE.search(text)
    scale = m.group(1).lower() if m else None
    currency = "USD" if _CURRENCY_RE.search(text) else None
    note = m.group(0) if m else None
    return scale, currency, note


def _table_to_markdown(table) -> str:
    rows = table.extract()
    if not rows:
        return ""
    lines = []
    header = rows[0]
    lines.append("| " + " | ".join((c or "").strip().replace("\n", " ") for c in header) + " |")
    lines.append("| " + " | ".join("---" for _ in header) + " |")
    for row in rows[1:]:
        lines.append("| " + " | ".join((c or "").strip().replace("\n", " ") for c in row) + " |")
    return "\n".join(lines)


def _extract_headers(page) -> list[tuple[float, str]]:
    """Return (y0, heading_text) for bold, short spans on the page -- section header heuristic
    tuned against real FinanceBench 10-Ks (headers are bold spans at body font size, distinct
    from the repeated 'Table of Contents' running header and bare page-number footers)."""
    headers = []
    d = page.get_text("dict")
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text or len(text) > 100:
                    continue
                if not (span["flags"] & _BOLD_FLAG):
                    continue
                if text in ("Table of Contents",) or text.isdigit():
                    continue
                headers.append((span["bbox"][1], text))
    return headers


def _nearest_header(headers: list[tuple[float, str]], y: float) -> str | None:
    best = None
    best_y = -1.0
    for hy, htext in headers:
        if hy <= y and hy > best_y:
            best_y = hy
            best = htext
    return best


def _prose_paragraphs(page, table_bboxes: list[tuple[float, float, float, float]]) -> list[tuple[str, float]]:
    """Return (paragraph_text, y0) for text blocks not overlapping a detected table bbox."""
    d = page.get_text("dict")
    paras = []
    for block in d.get("blocks", []):
        if block.get("type") != 0:
            continue
        bx0, by0, bx1, by1 = block["bbox"]
        overlaps = any(
            not (bx1 < tx0 or bx0 > tx1 or by1 < ty0 or by0 > ty1)
            for tx0, ty0, tx1, ty1 in table_bboxes
        )
        if overlaps:
            continue
        text = "\n".join(
            span["text"] for line in block.get("lines", []) for span in line.get("spans", [])
        ).strip()
        if text:
            paras.append((text, by0))
    return paras


def _split_prose(text: str) -> list[str]:
    if len(text) <= PROSE_CHUNK_CHARS:
        return [text]
    chunks = []
    step = PROSE_CHUNK_CHARS - PROSE_CHUNK_OVERLAP_CHARS
    for start in range(0, len(text), step):
        window = text[start : start + PROSE_CHUNK_CHARS]
        if window.strip():
            chunks.append(window)
        if start + PROSE_CHUNK_CHARS >= len(text):
            break
    return chunks


def chunk_pdf(pdf_path: Path, excluded_pages: frozenset[int] = frozenset()) -> DocumentChunks:
    """Table-aware chunking: one atomic markdown chunk per detected table (with scale/unit
    metadata + bbox), plus paragraph-grouped prose chunks carrying the nearest section header.

    `excluded_pages` drops those pages entirely (used for N1 evidence-ablation negatives,
    mirroring eval/run_retrieval_eval.py's naive baseline behavior).
    """
    doc = pymupdf.open(pdf_path)
    try:
        page_count = doc.page_count
        any_text = False
        result = DocumentChunks(page_count=page_count)

        for page_num in range(page_count):
            if page_num in excluded_pages:
                continue
            page = doc.load_page(page_num)
            page_text = page.get_text()
            if page_text.strip():
                any_text = True

            headers = _extract_headers(page)
            table_bboxes: list[tuple[float, float, float, float]] = []

            try:
                found = page.find_tables()
                tables = list(found.tables)
            except Exception:
                tables = []

            for table in tables:
                md = _table_to_markdown(table)
                if not md.strip():
                    continue
                bbox = tuple(table.bbox)
                table_bboxes.append(bbox)
                # Scan the table's own text plus a window of page text just above the table for
                # a scale/unit footnote (e.g. "(Millions)" / "in millions, except per share data").
                y0 = bbox[1]
                context = page_text
                scale, currency, note = _detect_scale(md + "\n" + context)
                nearest_header = _nearest_header(headers, y0)
                result.chunks.append(
                    Chunk(
                        page_num=page_num,
                        chunk_type="table",
                        text=_clean(f"{nearest_header}\n\n{md}" if nearest_header else md),
                        section_header=_clean(nearest_header) if nearest_header else None,
                        unit_scale=scale,
                        unit_currency=currency,
                        unit_note=_clean(note) if note else None,
                        bbox=bbox,
                    )
                )

            paragraphs = _prose_paragraphs(page, table_bboxes)
            # Group consecutive paragraphs under running headers into ~PROSE_CHUNK_CHARS windows.
            buf: list[str] = []
            buf_len = 0
            buf_header: str | None = None
            buf_y0: float | None = None

            def flush():
                nonlocal buf, buf_len, buf_header, buf_y0
                if not buf:
                    return
                joined = "\n".join(buf)
                for piece in _split_prose(joined):
                    prefixed = f"{buf_header}\n\n{piece}" if buf_header else piece
                    result.chunks.append(
                        Chunk(
                            page_num=page_num,
                            chunk_type="prose",
                            text=_clean(prefixed),
                            section_header=_clean(buf_header) if buf_header else None,
                        )
                    )
                buf = []
                buf_len = 0

            for text, y0 in paragraphs:
                header = _nearest_header(headers, y0)
                if buf and buf_len + len(text) > PROSE_CHUNK_CHARS:
                    flush()
                if buf_header is None:
                    buf_header = header
                buf.append(text)
                buf_len += len(text)
                buf_y0 = y0
            flush()

        if not any_text:
            raise EmptyTextLayerError(
                f"{pdf_path}: no extractable text on any non-excluded page "
                "(scanned PDF with no OCR is out of scope per PROJECT_PLAN.md)."
            )

        return result
    finally:
        doc.close()
