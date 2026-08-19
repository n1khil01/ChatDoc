"""Export the FastAPI app's OpenAPI schema to web/openapi.json for openapi-typescript.

Run whenever a route's request/response shape changes:
    uv run python scripts/export_openapi.py
    npm --prefix web run gen:types

Writes to a file (rather than pointing openapi-typescript at a running server) so type
generation doesn't depend on uvicorn being up, and works the same way in CI.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from api.app import app  # noqa: E402

OUTPUT_PATH = Path(__file__).parent.parent / "web" / "openapi.json"


def main() -> None:
    schema = app.openapi()
    OUTPUT_PATH.write_text(json.dumps(schema, indent=2), encoding="utf-8")
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
