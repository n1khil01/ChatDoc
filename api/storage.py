"""PDF blob storage: local disk in dev, Cloudflare R2 (S3-compatible) in production
(PROJECT_PLAN.md §5 "Blob storage" -- Render's free tier has no persistent disk, so an
uploaded PDF written to local disk is gone the moment the container spins down or restarts,
which for a free-tier box spinning down after 15 minutes idle is not a hypothetical).

Both backends are addressed by an opaque `key` string (e.g. "user3_ab12cd.pdf" -- the same
`doc_key` already used to name chunks/jobs) so callers never construct a filesystem path or
an S3 URL themselves. `get_storage()` picks R2 when all four R2_* env vars are set, else
falls back to local disk -- local dev, Docker Compose, and CI never need R2 credentials.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterator
from contextlib import AbstractContextManager, contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from dotenv import load_dotenv

# Explicit rather than relying on some other module (ingest/db.py) happening to import
# first and load .env before this module reads R2_* below -- load_dotenv() is idempotent
# (safe to call more than once; a var already in the environment is never overwritten), so
# this is just insurance against import-order becoming load-bearing.
load_dotenv(Path(__file__).parent.parent / ".env")

LOCAL_UPLOAD_DIR = Path(__file__).parent.parent / "data" / "uploads"

CHUNK_SIZE = 1024 * 1024


class StorageError(Exception):
    """Raised for a missing key -- backend-agnostic, so callers don't need to catch both
    FileNotFoundError (local) and botocore's ClientError (R2)."""


class Storage(Protocol):
    def save(self, key: str, data: bytes) -> None: ...
    def delete(self, key: str) -> None: ...
    def exists(self, key: str) -> bool: ...
    def open_local(self, key: str) -> AbstractContextManager[Path]: ...
    def stream(self, key: str) -> Iterator[bytes]: ...


class LocalStorage:
    """Dev / Docker Compose / CI backend -- plain files under data/uploads/."""

    def __init__(self, root: Path = LOCAL_UPLOAD_DIR):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key

    def save(self, key: str, data: bytes) -> None:
        self._path(key).write_bytes(data)

    def delete(self, key: str) -> None:
        self._path(key).unlink(missing_ok=True)

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    @contextmanager
    def open_local(self, key: str) -> Iterator[Path]:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(key)
        yield path

    def stream(self, key: str) -> Iterator[bytes]:
        path = self._path(key)
        if not path.is_file():
            raise StorageError(key)
        with path.open("rb") as f:
            while chunk := f.read(CHUNK_SIZE):
                yield chunk


class R2Storage:
    """Cloudflare R2 via its S3-compatible API. No egress fees, 10GB free (PROJECT_PLAN.md
    §5) -- what makes PDF storage free-tier-viable at all once Render's ephemeral disk is
    out of the picture."""

    def __init__(self) -> None:
        self._bucket = os.environ["R2_BUCKET_NAME"]
        self._client = boto3.client(
            "s3",
            endpoint_url=f"https://{os.environ['R2_ACCOUNT_ID']}.r2.cloudflarestorage.com",
            aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
            config=Config(signature_version="s3v4"),
            region_name="auto",
        )

    def save(self, key: str, data: bytes) -> None:
        self._client.put_object(
            Bucket=self._bucket, Key=key, Body=data, ContentType="application/pdf"
        )

    def delete(self, key: str) -> None:
        # delete_object is a no-op (not an error) on a missing key, matching
        # LocalStorage.delete's missing_ok=True -- deleting an already-gone file/object is
        # not a failure for either backend.
        self._client.delete_object(Bucket=self._bucket, Key=key)

    def exists(self, key: str) -> bool:
        try:
            self._client.head_object(Bucket=self._bucket, Key=key)
            return True
        except ClientError:
            return False

    @contextmanager
    def open_local(self, key: str) -> Iterator[Path]:
        # pymupdf's chunker wants a real file path; download to a throwaway temp file for
        # the duration of ingestion rather than buffering the whole PDF in memory, since
        # some filings run tens of MB and this runs on a memory-constrained box.
        fd, tmp_name = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            self._client.download_file(self._bucket, key, str(tmp_path))
        except ClientError as exc:
            tmp_path.unlink(missing_ok=True)
            raise StorageError(key) from exc
        try:
            yield tmp_path
        finally:
            tmp_path.unlink(missing_ok=True)

    def stream(self, key: str) -> Iterator[bytes]:
        try:
            obj = self._client.get_object(Bucket=self._bucket, Key=key)
        except ClientError as exc:
            raise StorageError(key) from exc
        body = obj["Body"]
        while chunk := body.read(CHUNK_SIZE):
            yield chunk


_R2_ENV_VARS = ("R2_ACCOUNT_ID", "R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_BUCKET_NAME")


@lru_cache(maxsize=1)
def get_storage() -> Storage:
    if all(os.environ.get(var) for var in _R2_ENV_VARS):
        return R2Storage()
    return LocalStorage()
