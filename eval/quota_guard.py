"""Client-side quota guard for the Gemini free tier (PROJECT_PLAN.md §7 Phase 0 step 5;
§8 Cost control; §9 Challenge 3).

Google cut the 2.5-flash free-tier daily request cap in December 2025 and no longer
publishes the number. Treat it as a runtime input, never a constant:

  - `daily_budget` must be passed in explicitly by the caller, read from the live limit
    shown in AI Studio for the key in use — this module does not guess it.
  - `max_calls` is a hard per-run ceiling independent of the daily budget, so a runaway
    loop stops at the cap, not at the quota.
  - `dry_run=True` logs what *would* have been called and sends nothing.

Rate limiting is a token bucket at a conservative ~10 RPM. On a 429, this module does
not trust the SDK's default retry policy — the caller is expected to raise `RateLimited`
with whatever `Retry-After` value the actual HTTP response carried, and this module
backs off from that, with jitter, rather than a blind fixed delay.
"""

from __future__ import annotations

import random
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Callable, TypeVar

from eval.cache import GenerationCache

DEFAULT_RPM = 10.0
MAX_BACKOFF_S = 60.0
MAX_RETRIES = 6

T = TypeVar("T")


class QuotaExceeded(RuntimeError):
    """Daily budget or --max-calls would be crossed. Not a crash: the resumable runner
    is expected to catch this, stop cleanly, and pick up where it left off on the next
    invocation (or the next day, for a daily-budget stop).
    """


class RateLimited(Exception):
    """The wrapped call function should raise this — with the real `Retry-After` value
    read from the 429 response — instead of letting a generic exception propagate.
    """

    def __init__(self, retry_after_s: float | None = None):
        self.retry_after_s = retry_after_s
        super().__init__(f"rate limited, retry_after_s={retry_after_s}")


@dataclass
class _TokenBucket:
    rpm: float
    _tokens: float = field(init=False)
    _last_refill: float = field(init=False, default_factory=time.monotonic)

    def __post_init__(self) -> None:
        self._tokens = self.rpm

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            elapsed = now - self._last_refill
            self._tokens = min(self.rpm, self._tokens + elapsed * (self.rpm / 60.0))
            self._last_refill = now
            if self._tokens >= 1:
                self._tokens -= 1
                return
            time.sleep(min((1 - self._tokens) * (60.0 / self.rpm), 5.0))


@dataclass
class QuotaGuard:
    cache: GenerationCache
    run_id: str
    daily_budget: int
    max_calls: int
    rpm: float = DEFAULT_RPM
    dry_run: bool = False
    max_retries: int = MAX_RETRIES

    _bucket: _TokenBucket = field(init=False)
    _calls_this_run: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        self._bucket = _TokenBucket(self.rpm)

    def _today(self) -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def calls_today(self) -> int:
        return self.cache.count_calls_on_date(self._today())

    def call(self, question_id: str, fn: Callable[[], T]) -> T | None:
        """Run `fn()`, gated by dry-run / hard caps / rate limit / backoff.

        `fn` performs the actual API request and must raise `RateLimited(retry_after_s)`
        on a 429 rather than let the SDK retry silently. Returns `fn()`'s result, or
        `None` if `dry_run` is set (nothing was called).
        """
        if self.dry_run:
            self.cache.log_call(self.run_id, question_id, "dry_run", "planned call, no request sent")
            print(f"[dry-run] would call API for {question_id}", file=sys.stderr)
            return None

        if self._calls_this_run >= self.max_calls:
            self.cache.log_call(self.run_id, question_id, "budget_stop", f"max_calls={self.max_calls}")
            raise QuotaExceeded(
                f"--max-calls={self.max_calls} reached this run; re-run to continue "
                f"(resumable — already-answered questions are skipped)"
            )

        if self.calls_today() >= self.daily_budget:
            self.cache.log_call(self.run_id, question_id, "budget_stop", f"daily_budget={self.daily_budget}")
            raise QuotaExceeded(
                f"daily budget of {self.daily_budget} calls reached; resume tomorrow — "
                f"the resumable runner will pick up where it left off"
            )

        attempt = 0
        while True:
            self._bucket.acquire()
            try:
                result = fn()
            except RateLimited as e:
                attempt += 1
                if attempt > self.max_retries:
                    self.cache.log_call(self.run_id, question_id, "error", f"exhausted retries: {e}")
                    raise
                base = e.retry_after_s if e.retry_after_s is not None else 2**attempt
                sleep_s = min(base + random.uniform(0, base * 0.25), MAX_BACKOFF_S)
                print(
                    f"  429 for {question_id}: backing off {sleep_s:.1f}s "
                    f"(attempt {attempt}/{self.max_retries})",
                    file=sys.stderr,
                )
                time.sleep(sleep_s)
                continue
            except Exception as e:
                self.cache.log_call(self.run_id, question_id, "error", str(e))
                raise
            else:
                self._calls_this_run += 1
                self.cache.log_call(self.run_id, question_id, "api_call", "")
                return result
