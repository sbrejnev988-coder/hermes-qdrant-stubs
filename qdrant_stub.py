#!/usr/bin/env python3
"""Small Qdrant-compatible HTTP service backed by SQLite WAL.

Implemented endpoints:
- GET /health
- GET /collections
- GET|PUT|DELETE /collections/{name}
- PUT /collections/{name}/points
- POST /collections/{name}/points/query
- POST /collections/{name}/points/search
- POST /collections/{name}/points/delete

Vectors are persisted as packed big-endian float32 BLOBs. The service opens no
monolithic in-memory store and updates only affected rows.
"""
from __future__ import annotations

import heapq
import json
import math
import os
import signal
import sqlite3
import struct
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Iterator
from urllib.parse import urlparse

STORAGE_DIR = Path(os.path.expanduser("~/.hermes/knowledge_db"))
DB_PATH = STORAGE_DIR / "vectors.db"
ERROR_LOG = Path("/tmp/qdrant_stub.log")
MAX_REQUEST_BYTES = 64 * 1024 * 1024
MAX_UPSERT_POINTS = 500
MAX_VECTOR_SIZE = 16384
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
_COUNTER_LOCK = threading.Lock()
_UPSERT_COUNT = 0
_SEARCH_COUNT = 0


def stub_log(message: str) -> None:
    """Append one diagnostic line without breaking request processing."""
    try:
        with ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} QDRANT {message}\n")
    except OSError:
        pass


def connect_db() -> sqlite3.Connection:
    """Create an independent WAL connection for one request/operation."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_db() -> None:
    """Create the durable schema and enable WAL once at process start."""
    connection = sqlite3.connect(DB_PATH, timeout=30)
    try:
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS collections (
                    name TEXT PRIMARY KEY,
                    vector_size INTEGER NOT NULL CHECK(vector_size > 0)
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS points (
                    collection TEXT NOT NULL,
                    id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    payload TEXT NOT NULL,
                    PRIMARY KEY (collection, id),
                    FOREIGN KEY (collection) REFERENCES collections(name)
                        ON DELETE CASCADE
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS points_collection_idx ON points(collection)"
            )
    finally:
        connection.close()


def pack_vector(vector: list[float]) -> bytes:
    """Encode a vector as compact float32 bytes."""
    return struct.pack(f">{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    """Decode a packed float32 vector."""
    if len(blob) % 4:
        raise ValueError(f"Invalid vector BLOB length: {len(blob)}")
    return struct.unpack(f">{len(blob) // 4}f", blob)


def cosine_similarity(query: list[float], candidate: tuple[float, ...]) -> float:
    """Return cosine similarity for equal-sized vectors."""
    dot = 0.0
    query_norm = 0.0
    candidate_norm = 0.0
    for left, right in zip(query, candidate):
        dot += left * right
        query_norm += left * left
        candidate_norm += right * right
    if query_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    return dot / math.sqrt(query_norm * candidate_norm)


def collection_size(name: str) -> int | None:
    connection = connect_db()
    try:
        row = connection.execute(
            "SELECT vector_size FROM collections WHERE name = ?", (name,)
        ).fetchone()
        return int(row[0]) if row else None
    finally:
        connection.close()


def iter_collection_points(name: str) -> Iterator[tuple[str, bytes, str]]:
    """Yield one persisted point at a time; vector BLOBs are not fetched all at once."""
    connection = connect_db()
    try:
        cursor = connection.execute(
            "SELECT id, vector, payload FROM points WHERE collection = ?", (name,)
        )
        for point_id, vector, payload in cursor:
            yield str(point_id), vector, str(payload)
    finally:
        connection.close()


def increment_counter(name: str) -> int:
    global _UPSERT_COUNT, _SEARCH_COUNT
    with _COUNTER_LOCK:
        if name == "upsert":
            _UPSERT_COUNT += 1
            return _UPSERT_COUNT
        _SEARCH_COUNT += 1
        return _SEARCH_COUNT


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class QdrantHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _read_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise ValueError("Invalid Content-Length") from exc
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body exceeds {MAX_REQUEST_BYTES} bytes")
        if length == 0:
            return {}
        try:
            value = json.loads(self.rfile.read(length))
        except json.JSONDecodeError as exc:
            raise ValueError("Malformed JSON body") from exc
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def _dispatch(self, callback) -> None:
        try:
            callback()
        except ValueError as exc:
            self._send_json({"status": "error", "error": str(exc)}, 400)
        except sqlite3.Error as exc:
            stub_log(f"SQLITE_ERROR {type(exc).__name__}: {exc}")
            self._send_json({"status": "error", "error": "SQLite operation failed"}, 503)
        except Exception as exc:  # keep the service alive and return a structured failure
            stub_log(f"REQUEST_ERROR {type(exc).__name__}: {exc}")
            self._send_json({"status": "error", "error": "Internal server error"}, 500)

    def do_GET(self) -> None:
        self._dispatch(self._do_get)

    def _do_get(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        if path == "/health":
            connection = connect_db()
            try:
                total = int(connection.execute("SELECT COUNT(*) FROM points").fetchone()[0])
            finally:
                connection.close()
            self._send_json({
                "status": "ok",
                "title": "qdrant-stub-sqlite",
                "version": "2.1",
                "total_points": total,
                "upsert_count": _UPSERT_COUNT,
                "search_count": _SEARCH_COUNT,
                "store_file_size": DB_PATH.stat().st_size if DB_PATH.exists() else 0,
                "store_backend": "sqlite-wal",
            })
            return
        if path == "/collections":
            connection = connect_db()
            try:
                rows = connection.execute("SELECT name FROM collections ORDER BY name").fetchall()
            finally:
                connection.close()
            self._send_json({
                "result": {"collections": [{"name": row[0]} for row in rows]},
                "status": "ok",
            })
            return
        if len(parts) == 2 and parts[0] == "collections":
            name = parts[1]
            size = collection_size(name)
            if size is None:
                self._send_json({"status": "error", "error": "Collection not found"}, 404)
                return
            connection = connect_db()
            try:
                count = int(connection.execute(
                    "SELECT COUNT(*) FROM points WHERE collection = ?", (name,)
                ).fetchone()[0])
            finally:
                connection.close()
            self._send_json({
                "result": {
                    "status": "green",
                    "points_count": count,
                    "indexed_vectors_count": count,
                    "config": {"params": {"vectors": {"size": size, "distance": "Cosine"}}},
                },
                "status": "ok",
            })
            return
        self._send_json({"status": "error", "error": "Not found"}, 404)

    def do_PUT(self) -> None:
        self._dispatch(self._do_put)

    def _do_put(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        body = self._read_body()
        if len(parts) == 2 and parts[0] == "collections":
            vectors = body.get("vectors", {})
            if not isinstance(vectors, dict):
                raise ValueError("vectors must be an object")
            size = int(vectors.get("size", 0))
            if size <= 0 or size > MAX_VECTOR_SIZE:
                raise ValueError(f"vector size must be between 1 and {MAX_VECTOR_SIZE}")
            connection = connect_db()
            try:
                with connection:
                    connection.execute(
                        "INSERT OR IGNORE INTO collections(name, vector_size) VALUES(?, ?)",
                        (parts[1], size),
                    )
                actual = collection_size(parts[1])
                if actual != size:
                    raise ValueError(f"Collection vector size is {actual}, requested {size}")
            finally:
                connection.close()
            self._send_json({"result": True, "status": "ok"})
            return
        if len(parts) == 3 and parts[0] == "collections" and parts[2] == "points":
            name = parts[1]
            size = collection_size(name)
            if size is None:
                self._send_json({"status": "error", "error": "Collection not found"}, 404)
                return
            points = body.get("points", [])
            if not isinstance(points, list):
                raise ValueError("points must be a list")
            if len(points) > MAX_UPSERT_POINTS:
                raise ValueError(f"At most {MAX_UPSERT_POINTS} points per request")
            rows: list[tuple[str, str, bytes, str]] = []
            for point in points:
                if not isinstance(point, dict):
                    raise ValueError("Each point must be an object")
                point_id = str(point.get("id", "")).strip()
                vector = point.get("vector")
                payload = point.get("payload", {})
                if not point_id:
                    raise ValueError("Point id must not be empty")
                if not isinstance(vector, list) or len(vector) != size:
                    actual = len(vector) if isinstance(vector, list) else 0
                    raise ValueError(f"Vector size mismatch: expected {size}, got {actual}")
                if not isinstance(payload, dict):
                    raise ValueError("Point payload must be an object")
                payload_json = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
                if len(payload_json.encode("utf-8")) > MAX_PAYLOAD_BYTES:
                    raise ValueError(f"Point payload exceeds {MAX_PAYLOAD_BYTES} bytes")
                rows.append((
                    name,
                    point_id,
                    pack_vector([float(item) for item in vector]),
                    payload_json,
                ))
            connection = connect_db()
            try:
                with connection:
                    connection.executemany(
                        "INSERT OR REPLACE INTO points(collection, id, vector, payload) VALUES(?, ?, ?, ?)",
                        rows,
                    )
            finally:
                connection.close()
            operation_id = increment_counter("upsert")
            self._send_json({
                "result": {"operation_id": operation_id, "status": "completed"},
                "status": "ok",
            })
            return
        self._send_json({"status": "error", "error": "Not found"}, 404)

    def do_POST(self) -> None:
        self._dispatch(self._do_post)

    def _do_post(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        body = self._read_body()
        if (len(parts) == 4 and parts[0] == "collections" and
                parts[2] == "points" and parts[3] == "delete"):
            point_ids = body.get("points", [])
            if not isinstance(point_ids, list):
                raise ValueError("points must be a list")
            connection = connect_db()
            try:
                with connection:
                    connection.executemany(
                        "DELETE FROM points WHERE collection = ? AND id = ?",
                        [(parts[1], str(point_id)) for point_id in point_ids],
                    )
            finally:
                connection.close()
            self._send_json({
                "result": {"operation_id": 0, "status": "completed"},
                "status": "ok",
            })
            return
        if (len(parts) == 4 and parts[0] == "collections" and
                parts[2] == "points" and parts[3] in {"query", "search"}):
            name = parts[1]
            size = collection_size(name)
            if size is None:
                self._send_json({"status": "error", "error": "Collection not found"}, 404)
                return
            key = "query" if parts[3] == "query" else "vector"
            raw_query = body.get(key, [])
            if not isinstance(raw_query, list) or len(raw_query) != size:
                actual = len(raw_query) if isinstance(raw_query, list) else 0
                raise ValueError(f"Vector size mismatch: expected {size}, got {actual}")
            query = [float(item) for item in raw_query]
            limit = max(1, min(int(body.get("limit", 10)), 1000))
            heap: list[tuple[float, int, str, dict[str, Any]]] = []
            sequence = 0
            for point_id, vector_blob, payload_json in iter_collection_points(name):
                candidate = unpack_vector(vector_blob)
                if len(candidate) != size:
                    stub_log(f"CORRUPT_VECTOR collection={name} id={point_id} size={len(candidate)}")
                    continue
                score = cosine_similarity(query, candidate)
                try:
                    payload = json.loads(payload_json) if payload_json else {}
                except json.JSONDecodeError:
                    payload = {}
                item = (score, sequence, point_id, payload)
                sequence += 1
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)
            points = [
                {"id": point_id, "score": score, "payload": payload}
                for score, _sequence, point_id, payload in sorted(heap, reverse=True)
            ]
            increment_counter("search")
            if parts[3] == "query":
                self._send_json({"result": {"points": points}, "status": "ok"})
            else:
                self._send_json({"result": points, "status": "ok"})
            return
        self._send_json({"status": "error", "error": "Not found"}, 404)

    def do_DELETE(self) -> None:
        self._dispatch(self._do_delete)

    def _do_delete(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "collections":
            connection = connect_db()
            try:
                with connection:
                    connection.execute("DELETE FROM points WHERE collection = ?", (parts[1],))
                    connection.execute("DELETE FROM collections WHERE name = ?", (parts[1],))
            finally:
                connection.close()
            self._send_json({"result": True, "status": "ok"})
            return
        self._send_json({"status": "error", "error": "Not found"}, 404)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def run(port: int = 6333) -> None:
    initialize_db()
    server = ThreadingHTTPServer(("127.0.0.1", port), QdrantHandler)

    def stop(_signum, _frame) -> None:
        server.server_close()
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stub_log(f"START port={port} store={DB_PATH.stat().st_size if DB_PATH.exists() else 0}b")
    try:
        server.serve_forever(poll_interval=0.25)
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stub_log("STOP")


if __name__ == "__main__":
    run()
