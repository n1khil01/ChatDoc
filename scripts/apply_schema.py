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

import psycopg  # noqa: E402

from ingest.db import DATABASE_URL, apply_schema, get_conn  # noqa: E402


def main() -> None:
    # get_conn() calls pgvector's register_vector() immediately on connect, which fails if
    # the `vector` type doesn't exist yet -- true on a brand new database (a fresh CI
    # Postgres service container every run) since schema.sql's own `CREATE EXTENSION` is
    # what would normally create it. Create the extension first over a plain connection that
    # doesn't try to register the type, then apply the rest of the schema normally.
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    with get_conn() as conn:
        apply_schema(conn, ROOT / "db" / "schema.sql")
    print("schema applied")


if __name__ == "__main__":
    main()
