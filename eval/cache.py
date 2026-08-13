"""Generation cache + resumable run bookkeeping (PROJECT_PLAN.md §7 Phase 0, step 4;
§8 Cost control).

Two tables, one SQLite file (local-laptop Phase 0 — schema is Postgres-compatible and
moves into Neon unchanged in Phase 4, per §5's "one datastore" decision):

  llm_cache        — keyed on sha256(prompt + model + params). Content-addressed, so the
                      *same* prompt against the *same* model/params never calls the API
                      twice, across runs. This is what makes ablation replays free.
  eval_generations — keyed on (run_id, question_id), unique. One row per question per
                      run. Lets a runner resume after a crash or a quota exhaustion by
                      skipping question_ids it already has a row for, without caring
                      whether the underlying prompt was a cache hit or a fresh call.

Every raw pre-gate generation is written here before any gate (L1/L2/L3) runs, so a gate
ablation is a SQL read over eval_generations, never a new API call.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

DB_PATH = Path(__file__).parent / "data" / "cache.sqlite3"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS llm_cache (
    cache_key       TEXT PRIMARY KEY,
    model           TEXT NOT NULL,
    prompt          TEXT NOT NULL,
    params_json     TEXT NOT NULL,
    raw_json        TEXT NOT NULL,
    latency_ms      REAL NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS eval_generations (
    run_id          TEXT NOT NULL,
    question_id     TEXT NOT NULL,
    prompt_hash     TEXT NOT NULL,
    cache_key       TEXT NOT NULL,
    raw_json        TEXT NOT NULL,
    latency_ms      REAL NOT NULL,
    prompt_tokens   INTEGER,
    completion_tokens INTEGER,
    cache_hit       INTEGER NOT NULL,
    created_at      TEXT NOT NULL,
    PRIMARY KEY (run_id, question_id)
);

CREATE TABLE IF NOT EXISTS call_log (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    question_id     TEXT,
    outcome         TEXT NOT NULL,  -- 'cache_hit' | 'api_call' | 'error' | 'budget_stop'
    detail          TEXT,
    created_at      TEXT NOT NULL
);
"""


def cache_key(prompt: str, model: str, params: dict) -> str:
    params_json = json.dumps(params, sort_keys=True)
    return hashlib.sha256(f"{model}\n{params_json}\n{prompt}".encode("utf-8")).hexdigest()


@contextmanager
def connect(db_path: Path = DB_PATH) -> Iterator[sqlite3.Connection]:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


@dataclass(frozen=True)
class Generation:
    raw: dict
    cache_hit: bool
    latency_ms: float
    prompt_tokens: int | None
    completion_tokens: int | None


class GenerationCache:
    """Read-before-call, write-after-call cache. Callers own the actual API call —
    this class only decides whether one is needed and records the result either way.
    """

    def __init__(self, db_path: Path = DB_PATH):
        self.db_path = db_path

    def lookup(self, prompt: str, model: str, params: dict) -> dict | None:
        key = cache_key(prompt, model, params)
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT raw_json FROM llm_cache WHERE cache_key = ?", (key,)
            ).fetchone()
        return json.loads(row["raw_json"]) if row else None

    def already_has_run_result(self, run_id: str, question_id: str) -> bool:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT 1 FROM eval_generations WHERE run_id = ? AND question_id = ?",
                (run_id, question_id),
            ).fetchone()
        return row is not None

    def get_run_result(self, run_id: str, question_id: str) -> dict | None:
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT raw_json FROM eval_generations WHERE run_id = ? AND question_id = ?",
                (run_id, question_id),
            ).fetchone()
        return json.loads(row["raw_json"]) if row else None

    def record(
        self,
        *,
        run_id: str,
        question_id: str,
        prompt: str,
        model: str,
        params: dict,
        raw: dict,
        cache_hit: bool,
        latency_ms: float,
        prompt_tokens: int | None = None,
        completion_tokens: int | None = None,
    ) -> None:
        key = cache_key(prompt, model, params)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        raw_json = json.dumps(raw, ensure_ascii=False)

        with connect(self.db_path) as conn:
            if not cache_hit:
                conn.execute(
                    "INSERT OR REPLACE INTO llm_cache "
                    "(cache_key, model, prompt, params_json, raw_json, latency_ms, "
                    " prompt_tokens, completion_tokens, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        key,
                        model,
                        prompt,
                        json.dumps(params, sort_keys=True),
                        raw_json,
                        latency_ms,
                        prompt_tokens,
                        completion_tokens,
                        now,
                    ),
                )
            conn.execute(
                "INSERT OR REPLACE INTO eval_generations "
                "(run_id, question_id, prompt_hash, cache_key, raw_json, latency_ms, "
                " prompt_tokens, completion_tokens, cache_hit, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    run_id,
                    question_id,
                    key,
                    key,
                    raw_json,
                    latency_ms,
                    prompt_tokens,
                    completion_tokens,
                    int(cache_hit),
                    now,
                ),
            )

    def log_call(self, run_id: str, question_id: str | None, outcome: str, detail: str = "") -> None:
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO call_log (run_id, question_id, outcome, detail, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, question_id, outcome, detail, now),
            )

    def count_calls_on_date(self, date_str: str, outcome: str = "api_call") -> int:
        """date_str is a UTC 'YYYY-MM-DD' prefix, matched against call_log.created_at.
        Counts across *all* run_ids — the daily budget is per API key, not per run.
        """
        with connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT COUNT(*) as n FROM call_log WHERE outcome = ? AND created_at LIKE ?",
                (outcome, f"{date_str}%"),
            ).fetchone()
        return row["n"]

    def run_stats(self, run_id: str) -> dict:
        with connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT outcome, COUNT(*) as n FROM call_log WHERE run_id = ? GROUP BY outcome",
                (run_id,),
            ).fetchall()
        return {r["outcome"]: r["n"] for r in rows}
