"""Apply db/schema.sql to DATABASE_URL. Idempotent (every statement is IF NOT EXISTS /
ADD COLUMN IF NOT EXISTS) -- safe to run against an already-migrated database, which is
what both local dev and CI (a fresh Postgres service container per run) need.

Usage:
    uv run python scripts/apply_schema.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from ingest.db import apply_schema, get_conn  # noqa: E402


def main() -> None:
    with get_conn() as conn:
        apply_schema(conn, ROOT / "db" / "schema.sql")
    print("schema applied")


if __name__ == "__main__":
    main()
