"""
SQLite registry of ingested files: doc_id (unique), file_path, file_hash.
Used to avoid re-ingesting the same file (by content hash).
"""
import hashlib
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

# DB path: data/ingestion.db (relative to project or cwd)
DATA_FOLDER = os.environ.get("INGESTION_DATA_FOLDER", "data")
REGISTRY_DB = os.path.join(DATA_FOLDER, "ingestion.db")


def _ensure_db():
    Path(DATA_FOLDER).mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(REGISTRY_DB)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS ingested_files (
            doc_id TEXT PRIMARY KEY,
            file_path TEXT NOT NULL,
            file_hash TEXT NOT NULL UNIQUE,
            ingested_at TEXT NOT NULL
        )
    """)
    conn.commit()
    return conn


def compute_file_hash(file_path: str) -> str:
    """SHA-256 hash of file content. Used to detect already-ingested files."""
    with open(file_path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def get_doc_id_by_hash(file_hash: str) -> str | None:
    """Return doc_id if this file_hash was already ingested, else None."""
    conn = _ensure_db()
    try:
        row = conn.execute(
            "SELECT doc_id FROM ingested_files WHERE file_hash = ?",
            (file_hash,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()


def register_file(doc_id: str, file_path: str, file_hash: str) -> None:
    """Record an ingested file (doc_id, path, hash)."""
    conn = _ensure_db()
    try:
        conn.execute(
            "INSERT INTO ingested_files (doc_id, file_path, file_hash, ingested_at) VALUES (?, ?, ?, ?)",
            (doc_id, os.path.abspath(file_path), file_hash, datetime.now(tz=timezone.utc).isoformat()),
        )
        conn.commit()
    finally:
        conn.close()


def get_file_path_by_doc_id(doc_id: str) -> str | None:
    """Return stored file_path for a doc_id, or None."""
    conn = _ensure_db()
    try:
        row = conn.execute(
            "SELECT file_path FROM ingested_files WHERE doc_id = ?",
            (doc_id,),
        ).fetchone()
        return row[0] if row else None
    finally:
        conn.close()
