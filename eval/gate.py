"""The grounding gate: L1 (retrieval confidence, experimental) / L2 (schema sufficiency +
citation validation) / L3a (numeric + operand provenance) / L3b (prose claim support).
PROJECT_PLAN.md §7 Phase 2.

Every function here is a pure check over an already-parsed eval.gate_schema.GateAnswer plus
retrieval context captured at generation time -- no model calls, no I/O, so
eval/ablation.py can replay every config over a cached run at zero API cost (§8).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from eval.gate_schema import GateAnswer
from eval.normalize import scale_multiplier
from ingest.reranker import rerank as rerank_fn

# The user's exact required refusal string (PROJECT_PLAN.md §7 Phase 2 step 6).
REFUSAL_TEXT = "this file does not contain the necessary information to answer the question"

_NUMBER_RE = re.compile(r"-?\(?\d[\d,]*\.?\d*\)?")


@dataclass(frozen=True)
class GateResult:
    passed: bool
    reason: str | None = None


def _value_found_in_text(value: float, scale: str | None, text: str) -> bool:
    """Search `text` for a numeric substring that matches `value` once un-scaled back to the
    raw figure as it would appear on the page (e.g. value=1_577_000_000, scale='millions' ->
    look for "1,577" or "1577" in the chunk text, not the fully-expanded figure).
    """
    mult = scale_multiplier(scale)
    raw = value / mult if mult else value
    candidates = {raw, round(raw), round(raw, 2)}
    variants: set[str] = set()
    for v in candidates:
        s = f"{v:,.2f}".rstrip("0").rstrip(".")
        variants.add(s)
        variants.add(s.replace(",", ""))
        s_int = f"{v:,.0f}"
        variants.add(s_int)
        variants.add(s_int.replace(",", ""))
    for v in variants:
        digits = re.sub(r"[^0-9]", "", v)
        if len(digits) < 2:
            continue
        pattern = re.compile(rf"(?<!\d){re.escape(v)}(?!\d)")
        if pattern.search(text):
            return True
    return False


def check_l2(answer: GateAnswer, retrieved_chunk_ids: set[int]) -> GateResult:
    """Schema-enforced sufficiency + citation-id validation. Free, catches outright
    fabrication: the model must say it has enough evidence, and every chunk_id it names
    must actually be one of the chunks it was shown.
    """
    if not answer.sufficient:
        return GateResult(False, "model reported insufficient evidence")

    cited_ids = {c.chunk_id for c in answer.citations} | {
        o.citation_chunk_id for o in answer.operands
    }
    unknown = cited_ids - retrieved_chunk_ids
    if unknown:
        return GateResult(False, f"citation references unretrieved chunk_id(s): {sorted(unknown)}")

    return GateResult(True)


def check_l3a_numeric(
    answer: GateAnswer,
    chunk_text_by_id: dict[int, str],
    chunk_scale_by_id: dict[int, str | None],
) -> GateResult:
    """Numeric + operand provenance. For a derived answer (operands present), every operand
    must be findable in its cited chunk -- verifying the inputs rather than the arithmetic
    result, per PROJECT_PLAN.md §7 Phase 2 step 2. For a direct answer (no operands), the
    top-level `value` must be findable in one of its cited chunks.
    """
    if answer.operands:
        for op in answer.operands:
            text = chunk_text_by_id.get(op.citation_chunk_id)
            if text is None:
                return GateResult(False, f"operand cites unknown chunk_id {op.citation_chunk_id}")
            scale = chunk_scale_by_id.get(op.citation_chunk_id)
            if not _value_found_in_text(op.value, scale, text):
                return GateResult(
                    False, f"operand value {op.value} not found in cited chunk {op.citation_chunk_id}"
                )
        return GateResult(True)

    if answer.value is not None:
        for c in answer.citations:
            text = chunk_text_by_id.get(c.chunk_id)
            if text is None:
                continue
            scale = chunk_scale_by_id.get(c.chunk_id) or answer.scale
            if _value_found_in_text(answer.value, scale, text):
                return GateResult(True)
        return GateResult(False, f"value {answer.value} not found in any cited chunk")

    # No numeric value and no operands -- nothing for L3a to check; L3b covers prose claims.
    return GateResult(True)


def check_l3b_prose(
    answer: GateAnswer,
    chunk_text_by_id: dict[int, str],
    threshold: float,
) -> GateResult:
    """Prose claim support: reuse the MiniLM cross-encoder already loaded for retrieval
    reranking to score (claim, cited_chunk) pairs, per PROJECT_PLAN.md §7 Phase 2 step 3 --
    one model, two jobs, no lexical-overlap heuristics on boilerplate-heavy filings.
    """
    if not answer.answer_text:
        return GateResult(True)
    if not answer.citations:
        return GateResult(False, "prose claim has no citations to support it")

    texts = [chunk_text_by_id[c.chunk_id] for c in answer.citations if c.chunk_id in chunk_text_by_id]
    if not texts:
        return GateResult(False, "no cited chunk text available to score")

    scores = rerank_fn(answer.answer_text, texts)
    best = max(scores) if scores else 0.0
    if best < threshold:
        return GateResult(False, f"best claim-support score {best:.4f} below threshold {threshold}")
    return GateResult(True)


def check_l1_margin(top1_score: float, top2_score: float | None, threshold: float) -> GateResult:
    """Retrieval confidence gate, built as an *experiment* (PROJECT_PLAN.md §7 Phase 2 step
    4) -- margin (top1 - top2, more query-invariant than an absolute cross-encoder score)
    rather than a global threshold on an uncalibrated logit.
    """
    margin = top1_score - (top2_score if top2_score is not None else 0.0)
    if margin < threshold:
        return GateResult(False, f"retrieval margin {margin:.4f} below threshold {threshold}")
    return GateResult(True)


def apply_gate(
    answer: GateAnswer,
    *,
    retrieved_chunk_ids: set[int],
    chunk_text_by_id: dict[int, str],
    chunk_scale_by_id: dict[int, str | None],
    layers: frozenset[str],
    l3b_threshold: float = 0.0,
    l1_margin: tuple[float, float | None] | None = None,
    l1_threshold: float = 0.0,
) -> GateResult:
    """Run the requested subset of {'L1','L2','L3'} in order, short-circuiting on first
    failure. 'L3' runs both L3a (numeric) and L3b (prose) -- they are never split
    independently in the ablation table (PROJECT_PLAN.md §7 Phase 2 step 7 lists
    {none}/{L2}/{L2+L3}/{L1+L2+L3}, not L3a/L3b separately).
    """
    if "L1" in layers:
        if l1_margin is None:
            return GateResult(False, "L1 requested but no rerank scores available")
        result = check_l1_margin(l1_margin[0], l1_margin[1], l1_threshold)
        if not result.passed:
            return result

    if "L2" in layers:
        result = check_l2(answer, retrieved_chunk_ids)
        if not result.passed:
            return result

    if "L3" in layers:
        result = check_l3a_numeric(answer, chunk_text_by_id, chunk_scale_by_id)
        if not result.passed:
            return result
        result = check_l3b_prose(answer, chunk_text_by_id, l3b_threshold)
        if not result.passed:
            return result

    return GateResult(True)
