"""Numeric normalization shared by dataset classification and grading.

FinanceBench gold answers are free text ("$1,577.00", "1577", "$1.577 billion", "12.3%").
This is the one normalizer both `build_dataset.py` (classification) and the future
grader (`runner.py`) must use, so a number is graded the same way it was classified.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_SCALE_WORDS = {
    "thousand": 1e3,
    "thousands": 1e3,
    "million": 1e6,
    "millions": 1e6,
    "mm": 1e6,
    "billion": 1e9,
    "billions": 1e9,
    "bn": 1e9,
    "trillion": 1e12,
    "trillions": 1e12,
}

_NUMBER_RE = re.compile(r"-?\(?\d[\d,]*\.?\d*\)?")


@dataclass(frozen=True)
class NormalizedNumber:
    value: float
    is_percent: bool


def normalize_numeric_answer(text: str) -> NormalizedNumber | None:
    """Parse a free-text financial answer into a single float, or None if not gradable.

    Returns None for answers that are not reducible to one number: multi-part answers,
    pure prose, ranges, or answers with no digits at all.
    """
    if not text:
        return None
    stripped = text.strip()

    is_percent = "%" in stripped

    lowered = stripped.lower()
    scale = 1.0
    for word, mult in _SCALE_WORDS.items():
        if re.search(rf"\b{word}\b", lowered):
            scale = mult
            break

    matches = _NUMBER_RE.findall(stripped)
    if not matches:
        return None
    if len(matches) > 1:
        # More than one bare number (e.g. a range or a two-part answer) is not
        # deterministically gradable as a single field.
        return None

    raw = matches[0]
    negative = raw.startswith("(") and raw.endswith(")")
    cleaned = raw.strip("()").replace(",", "")
    try:
        value = float(cleaned)
    except ValueError:
        return None
    if negative:
        value = -value

    if not is_percent:
        value *= scale

    return NormalizedNumber(value=value, is_percent=is_percent)


def numbers_match(a: float, b: float, rel_tol: float = 0.01) -> bool:
    """~1% relative tolerance for rounding, per PROJECT_PLAN.md §6."""
    if a == b:
        return True
    denom = max(abs(a), abs(b), 1e-9)
    return abs(a - b) / denom <= rel_tol
