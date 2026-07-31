#!/usr/bin/env python3
"""Incrementally index active memory-wiki claims into the local Qdrant-compatible store.

The checkpoint is the persisted claim_id + content_hash in each point payload.
A point is embedded only when it is missing or its source text changed.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import sqlite3
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

HERMES_HOME = Path(os.environ.get("HERMES_HOME", "/root/.hermes"))
MEMORY_DB = Path(os.environ.get(
    "MEMORY_WIKI_DB",
    str(HERMES_HOME / "memory-wiki" / "memory_wiki.sqlite3"),
))
VECTOR_DB = HERMES_HOME / "knowledge_db" / "vectors.db"
QDRANT_URL = os.environ.get("MEMORY_WIKI_QDRANT_URL", "http://127.0.0.1:6333").rstrip("/")
COLLECTION = os.environ.get("MEMORY_WIKI_QDRANT_COLLECTION", "memory_wiki_claims_v6")
MODEL = os.environ.get("MEMORY_WIKI_EMBED_MODEL", "perplexity/pplx-embed-v1-4b")
VECTOR_SIZE = int(os.environ.get("MEMORY_WIKI_VECTOR_SIZE", "2560"))
EMBED_URL = os.environ.get("MEMORY_WIKI_EMBED_URL", "https://openrouter.ai/api/v1").rstrip("/")
API_KEY = os.environ.get("MEMORY_WIKI_EMBED_API_KEY", "")
BATCH_SIZE = max(1, int(os.environ.get("MEMORY_WIKI_REINDEX_BATCH", "10")))
LOCK_PATH = Path(os.environ.get("MEMORY_WIKI_REINDEX_LOCK", "/tmp/reindex_qwen3.lock"))


def log(message: str) -> None:
    print(message, flush=True)


def request_json(
    method: str,
    url: str,
    body: dict[str, Any] | None = None,
    timeout: float = 20.0,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {"Accept": "application/json"}
    if headers:
        request_headers.update(headers)
    data = None
    if body is not None:
        data = json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        request_headers["Content-Type"] = "application/json"
    request = urllib.request.Request(url, data=data, headers=request_headers, method=method)
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read()
    return json.loads(raw) if raw else {}


def wait_for_qdrant(timeout_seconds: int = 90) -> None:
    deadline = time.monotonic() + timeout_seconds
    last_error: Exception | None = None
    while time.monotonic() < deadline:
        try:
            health = request_json("GET", f"{QDRANT_URL}/health", timeout=3)
            if health.get("status") == "ok":
                return
        except Exception as exc:  # service can still be starting
            last_error = exc
        time.sleep(2)
    raise RuntimeError(f"Qdrant is unavailable at {QDRANT_URL}: {last_error}")


def ensure_collection() -> None:
    try:
        current = request_json("GET", f"{QDRANT_URL}/collections/{COLLECTION}", timeout=5)
    except urllib.error.HTTPError as exc:
        if exc.code != 404:
            raise
        current = {}
    vectors = (
        current.get("result", {})
        .get("config", {})
        .get("params", {})
        .get("vectors", {})
    )
    actual_size = vectors.get("size")
    if actual_size is not None and int(actual_size) != VECTOR_SIZE:
        raise RuntimeError(
            f"Collection {COLLECTION} has vector size {actual_size}, expected {VECTOR_SIZE}"
        )
    if actual_size is None:
        request_json(
            "PUT",
            f"{QDRANT_URL}/collections/{COLLECTION}",
            {"vectors": {"size": VECTOR_SIZE, "distance": "Cosine"}},
            timeout=10,
        )


def make_document(row: sqlite3.Row) -> str:
    # Keep the same canonical document format used for the initial v6 build.
    claim = str(row["claim"] or "").strip()
    topic = str(row["topic"] or "").strip()
    return f"{claim} {topic}".strip()


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def load_claims() -> list[dict[str, Any]]:
    connection = sqlite3.connect(f"file:{MEMORY_DB}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    rows = connection.execute(
        """
        SELECT id, claim, topic, type, scope, updated_at
        FROM claims
        WHERE status = 'active'
        ORDER BY id
        """
    ).fetchall()
    connection.close()
    claims: list[dict[str, Any]] = []
    for row in rows:
        document = make_document(row)
        claims.append({
            "id": str(row["id"]),
            "document": document,
            "embedding_text": document[:500],
            "content_hash": content_hash(document),
            "topic": str(row["topic"] or ""),
            "type": str(row["type"] or ""),
            "updated_at": int(row["updated_at"] or 0),
        })
    return claims


def point_id(claim_id: str) -> str:
    value = str(claim_id).strip()
    if value.isdigit() and 0 <= int(value) < 2**64:
        return value
    try:
        return str(uuid.UUID(value))
    except ValueError:
        return str(uuid.uuid5(uuid.NAMESPACE_URL, f"hermes-memory-wiki:{value}"))


def load_existing_hashes() -> dict[str, tuple[str, str]]:
    """Map claim_id to (content_hash, persisted point_id) without loading vector BLOBs."""
    if not VECTOR_DB.exists() or QDRANT_URL not in {
        "http://127.0.0.1:6333",
        "http://localhost:6333",
    }:
        return {}
    connection = sqlite3.connect(f"file:{VECTOR_DB}?mode=ro", uri=True)
    rows = connection.execute(
        "SELECT id, payload FROM points WHERE collection = ?",
        (COLLECTION,),
    ).fetchall()
    connection.close()
    result: dict[str, tuple[str, str]] = {}
    for point_id, payload_json in rows:
        try:
            payload = json.loads(payload_json or "{}")
        except json.JSONDecodeError:
            payload = {}
        claim_id = str(payload.get("claim_id") or point_id)
        result[claim_id] = (str(payload.get("content_hash") or ""), str(point_id))
    return result


def embed_documents(texts: list[str]) -> list[list[float]]:
    payload = {
        "model": MODEL,
        "input": texts,
        "encoding_format": "float",
        "dimensions": VECTOR_SIZE,
        "input_type": "search_document",
    }
    headers = {"Authorization": f"Bearer {API_KEY}"}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            result = request_json(
                "POST",
                f"{EMBED_URL}/embeddings",
                payload,
                timeout=45,
                headers=headers,
            )
            vectors = [item.get("embedding") for item in result.get("data", [])]
            if len(vectors) != len(texts):
                raise RuntimeError(f"Embedding count mismatch: {len(vectors)} != {len(texts)}")
            for vector in vectors:
                if not isinstance(vector, list) or len(vector) != VECTOR_SIZE:
                    actual = len(vector) if isinstance(vector, list) else 0
                    raise RuntimeError(f"Embedding size mismatch: {actual} != {VECTOR_SIZE}")
            return vectors
        except Exception as exc:
            last_error = exc
            if attempt == 3:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Embedding request failed after retries: {last_error}")


def upsert_batch(batch: list[dict[str, Any]], vectors: list[list[float]]) -> None:
    """Upsert one batch with retries; a failed batch remains pending by content_hash."""
    points = []
    for claim, vector in zip(batch, vectors):
        points.append({
            "id": point_id(claim["id"]),
            "vector": vector,
            "payload": {
                "claim_id": claim["id"],
                "content_hash": claim["content_hash"],
                "text": claim["document"][:1000],
                "topic": claim["topic"],
                "type": claim["type"],
                "updated_at": claim["updated_at"],
                "embedding_model": MODEL,
            },
        })
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            result = request_json(
                "PUT",
                f"{QDRANT_URL}/collections/{COLLECTION}/points?wait=true",
                {"points": points},
                timeout=30,
            )
            status = result.get("result", {}).get("status")
            if status not in {"completed", "acknowledged"}:
                raise RuntimeError(f"Qdrant upsert failed: {result}")
            return
        except Exception as exc:
            last_error = exc
            if attempt < 3:
                time.sleep(2 ** attempt)
    raise RuntimeError(f"Qdrant upsert failed after retries: {last_error}")


def delete_points(point_ids: list[str]) -> None:
    for offset in range(0, len(point_ids), 500):
        request_json(
            "POST",
            f"{QDRANT_URL}/collections/{COLLECTION}/points/delete?wait=true",
            {"points": point_ids[offset:offset + 500]},
            timeout=30,
        )


def main() -> int:
    if not API_KEY:
        log("MEMORY_WIKI_EMBED_API_KEY is missing; indexing aborted")
        return 2
    lock_file = LOCK_PATH.open("a+")
    try:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        log("Another reindex process is already running; skipping")
        return 0

    wait_for_qdrant()
    ensure_collection()
    claims = load_claims()
    existing = load_existing_hashes()
    active_ids = {claim["id"] for claim in claims}
    stale_point_ids = [persisted_id for claim_id, (_, persisted_id) in existing.items() if claim_id not in active_ids]
    if stale_point_ids:
        delete_points(stale_point_ids)
        log(f"Removed {len(stale_point_ids)} retired/deleted points")
    pending = [
        claim for claim in claims
        if existing.get(claim["id"], ("", ""))[0] != claim["content_hash"]
    ]
    log(
        f"Index {COLLECTION}: total={len(claims)} current={len(claims) - len(pending)} "
        f"pending={len(pending)} model={MODEL} dims={VECTOR_SIZE}"
    )
    if not pending:
        log("Index is current; no embedding API calls needed")
        return 0

    indexed = 0
    failures: list[str] = []
    started = time.monotonic()
    for offset in range(0, len(pending), BATCH_SIZE):
        batch = pending[offset:offset + BATCH_SIZE]
        try:
            vectors = embed_documents([claim["embedding_text"] for claim in batch])
            upsert_batch(batch, vectors)
            indexed += len(batch)
        except Exception as exc:
            failures.extend(claim["id"] for claim in batch)
            log(f"Batch {offset // BATCH_SIZE + 1} failed: {exc}")
        if indexed and (indexed % 100 == 0 or indexed == len(pending)):
            elapsed = max(time.monotonic() - started, 0.001)
            remaining = len(pending) - indexed - len(failures)
            eta = remaining / (indexed / elapsed) if indexed else 0
            log(f"Progress {indexed}/{len(pending)}; failures={len(failures)}; ETA={eta / 60:.1f}m")

    elapsed = time.monotonic() - started
    log(f"Finished: indexed={indexed} failures={len(failures)} elapsed={elapsed:.1f}s")
    if failures:
        log("Failed claim IDs: " + ",".join(failures[:50]))
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
