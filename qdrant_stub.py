#!/usr/bin/env python3
"""Qdrant-compatible SQLite stub for Hermes Memory Wiki.

r9 hardening:
- HERMES_HOME-aware storage
- aliases API and atomic alias updates
- paginated points/scroll used by Memory Wiki reindex reconciliation
- strict vector/payload/name validation; NaN/Inf rejected
- SQLite foreign keys enabled on every connection
- bounded request, upsert, delete, query, and scroll operations
- deterministic, strict JSON responses
"""
from __future__ import annotations

import hashlib
import heapq
import json
import math
import os
import signal
import sqlite3
import struct
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any, Iterator
from urllib.parse import urlparse

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
STORAGE_DIR = Path(os.environ.get("QDRANT_STUB_STORAGE_DIR", str(HERMES_HOME / "knowledge_db"))).expanduser()
DB_PATH = Path(os.environ.get("QDRANT_STUB_DB", str(STORAGE_DIR / "vectors.db"))).expanduser()
ERROR_LOG = Path(os.environ.get("QDRANT_STUB_LOG", "/tmp/qdrant_stub.log"))
BIND_HOST = os.environ.get("QDRANT_STUB_HOST", "127.0.0.1")
DEFAULT_PORT = int(os.environ.get("QDRANT_STUB_PORT", "6333"))
API_KEY = os.environ.get("QDRANT_STUB_API_KEY", "") or os.environ.get("MEMORY_WIKI_QDRANT_API_KEY", "")

MAX_REQUEST_BYTES = int(os.environ.get("QDRANT_STUB_MAX_REQUEST_BYTES", str(64 * 1024 * 1024)))
MAX_UPSERT_POINTS = int(os.environ.get("QDRANT_STUB_MAX_UPSERT_POINTS", "500"))
MAX_DELETE_POINTS = int(os.environ.get("QDRANT_STUB_MAX_DELETE_POINTS", "5000"))
MAX_VECTOR_SIZE = int(os.environ.get("QDRANT_STUB_MAX_VECTOR_SIZE", "65536"))
MAX_PAYLOAD_BYTES = int(os.environ.get("QDRANT_STUB_MAX_PAYLOAD_BYTES", str(2 * 1024 * 1024)))
MAX_COLLECTION_NAME = 200
MAX_ALIAS_NAME = 200
MAX_POINT_ID = 512
MAX_SCROLL_LIMIT = 2048
MAX_QUERY_LIMIT = 1000

STORAGE_DIR.mkdir(parents=True, exist_ok=True)
_COUNTER_LOCK = threading.Lock()
_UPSERT_COUNT = 0
_SEARCH_COUNT = 0
_SCROLL_COUNT = 0


class HttpProblem(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = int(status)
        self.message = str(message)


def stub_log(message: str) -> None:
    try:
        ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
        with ERROR_LOG.open("a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} QDRANT {message}\n")
    except OSError:
        pass


def connect_db() -> sqlite3.Connection:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA synchronous=NORMAL")
    connection.execute("PRAGMA busy_timeout=30000")
    return connection


def initialize_db() -> None:
    connection = sqlite3.connect(DB_PATH, timeout=30)
    try:
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        connection.execute("PRAGMA synchronous=NORMAL")
        connection.execute("PRAGMA busy_timeout=30000")
        with connection:
            connection.execute(
                """CREATE TABLE IF NOT EXISTS collections(
                    name TEXT PRIMARY KEY,
                    vector_size INTEGER NOT NULL CHECK(vector_size > 0),
                    distance TEXT NOT NULL DEFAULT 'Cosine',
                    created_at INTEGER NOT NULL DEFAULT 0
                )"""
            )
            # Migration from the older two-column schema.
            columns = {row[1] for row in connection.execute("PRAGMA table_info(collections)")}
            if "distance" not in columns:
                connection.execute("ALTER TABLE collections ADD COLUMN distance TEXT NOT NULL DEFAULT 'Cosine'")
            if "created_at" not in columns:
                connection.execute("ALTER TABLE collections ADD COLUMN created_at INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS points(
                    collection TEXT NOT NULL,
                    id TEXT NOT NULL,
                    vector BLOB NOT NULL,
                    payload TEXT NOT NULL,
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(collection,id),
                    FOREIGN KEY(collection) REFERENCES collections(name) ON DELETE CASCADE
                )"""
            )
            point_columns = {row[1] for row in connection.execute("PRAGMA table_info(points)")}
            if "updated_at" not in point_columns:
                connection.execute("ALTER TABLE points ADD COLUMN updated_at INTEGER NOT NULL DEFAULT 0")
            connection.execute(
                """CREATE TABLE IF NOT EXISTS aliases(
                    alias_name TEXT PRIMARY KEY,
                    collection_name TEXT NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT 0,
                    FOREIGN KEY(collection_name) REFERENCES collections(name) ON DELETE CASCADE
                )"""
            )
            connection.execute("CREATE INDEX IF NOT EXISTS points_collection_idx ON points(collection,id)")
            connection.execute("CREATE INDEX IF NOT EXISTS aliases_collection_idx ON aliases(collection_name)")
            bootstrap_memory_wiki_alias(connection)
    finally:
        connection.close()


def bootstrap_memory_wiki_alias(connection: sqlite3.Connection) -> None:
    """Create the configured Memory Wiki alias for an existing manifest collection.

    This closes the upgrade gap when an old alias-less stub is replaced while an
    r8 Gateway is still running: alias capability becomes visible immediately,
    so the mapping must exist before the first online read/outbox write.
    """
    if os.environ.get("QDRANT_STUB_AUTO_MEMORY_WIKI_ALIAS", "1").lower() in {"0", "false", "no", "off"}:
        return
    alias = os.environ.get("MEMORY_WIKI_QDRANT_ALIAS", "memory_wiki_claims_active").strip()
    prefix = os.environ.get("MEMORY_WIKI_QDRANT_COLLECTION", "memory_wiki_claims").strip()
    if not alias or not prefix:
        return
    if connection.execute("SELECT 1 FROM aliases WHERE alias_name=?", (alias,)).fetchone():
        return
    manifest_path = HERMES_HOME / "memory-wiki" / "embedding_manifest.json"
    if not manifest_path.is_file():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(manifest, dict):
            return
        physical_manifest = {
            key: value for key, value in manifest.items()
            if key not in {"query_instruction_hash"}
        }
        digest = hashlib.sha256(
            json.dumps(physical_manifest, sort_keys=True, ensure_ascii=True).encode("utf-8")
        ).hexdigest()[:12]
        candidate = f"{prefix}_{digest}"
        if not connection.execute("SELECT 1 FROM collections WHERE name=?", (candidate,)).fetchone():
            return
        connection.execute(
            "INSERT INTO aliases(alias_name,collection_name,created_at) VALUES(?,?,?)",
            (alias, candidate, int(time.time())),
        )
        stub_log(f"AUTO_ALIAS {alias} -> {candidate}")
    except Exception as exc:
        stub_log(f"AUTO_ALIAS_FAILED {type(exc).__name__}: {exc}")


def _valid_name(value: Any, *, kind: str, max_len: int) -> str:
    text = str(value or "").strip()
    if not text or len(text) > max_len or any(ch in text for ch in "/\\\x00"):
        raise HttpProblem(400, f"Invalid {kind} name")
    return text


def _valid_point_id(value: Any) -> str:
    text = str(value or "").strip()
    if not text or len(text) > MAX_POINT_ID or "\x00" in text:
        raise HttpProblem(400, "Invalid point id")
    return text


def _strict_float_vector(value: Any, expected: int) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        actual = len(value) if isinstance(value, list) else 0
        raise HttpProblem(400, f"Vector size mismatch: expected {expected}, got {actual}")
    try:
        vector = [float(item) for item in value]
    except (TypeError, ValueError, OverflowError) as exc:
        raise HttpProblem(400, "Vector contains a non-numeric value") from exc
    if not all(math.isfinite(item) for item in vector):
        raise HttpProblem(400, "Vector values must be finite")
    return vector


def _strict_payload(value: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(value, dict):
        raise HttpProblem(400, "Point payload must be an object")
    try:
        encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    except (TypeError, ValueError) as exc:
        raise HttpProblem(400, "Point payload is not strict-JSON serializable") from exc
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise HttpProblem(400, f"Point payload exceeds {MAX_PAYLOAD_BYTES} bytes")
    return value, encoded


def pack_vector(vector: list[float]) -> bytes:
    return struct.pack(f">{len(vector)}f", *vector)


def unpack_vector(blob: bytes) -> tuple[float, ...]:
    if len(blob) % 4:
        raise ValueError(f"Invalid vector BLOB length: {len(blob)}")
    return struct.unpack(f">{len(blob) // 4}f", blob)


def cosine_similarity(query: list[float], candidate: tuple[float, ...]) -> float:
    if len(query) != len(candidate):
        raise ValueError("Vector dimension mismatch")
    dot = sum(left * right for left, right in zip(query, candidate))
    query_norm = sum(item * item for item in query)
    candidate_norm = sum(item * item for item in candidate)
    if query_norm == 0.0 or candidate_norm == 0.0:
        return 0.0
    score = dot / math.sqrt(query_norm * candidate_norm)
    return score if math.isfinite(score) else 0.0


def _payload_condition_matches(payload: dict[str, Any], condition: Any) -> bool:
    if not isinstance(condition, dict):
        return False
    key = condition.get("key")
    if not isinstance(key, str) or not key:
        return False
    value: Any = payload
    for part in key.split("."):
        if not isinstance(value, dict) or part not in value:
            return False
        value = value[part]
    match = condition.get("match")
    if not isinstance(match, dict):
        return False
    if "value" in match:
        return value == match.get("value")
    if "any" in match and isinstance(match.get("any"), list):
        return value in match["any"]
    if "except" in match and isinstance(match.get("except"), list):
        return value not in match["except"]
    return False


def _payload_matches_filter(payload: dict[str, Any], filter_obj: Any) -> bool:
    if not filter_obj:
        return True
    if not isinstance(filter_obj, dict):
        raise HttpProblem(400, "filter must be an object")
    must = filter_obj.get("must", [])
    must_not = filter_obj.get("must_not", [])
    should = filter_obj.get("should", [])
    if must and (not isinstance(must, list) or not all(_payload_condition_matches(payload, item) for item in must)):
        return False
    if must_not and (not isinstance(must_not, list) or any(_payload_condition_matches(payload, item) for item in must_not)):
        return False
    if should:
        if not isinstance(should, list):
            raise HttpProblem(400, "filter.should must be a list")
        minimum = int(filter_obj.get("min_should", 1) or 1)
        if sum(1 for item in should if _payload_condition_matches(payload, item)) < minimum:
            return False
    return True


def resolve_collection(name: str, connection: sqlite3.Connection | None = None) -> str | None:
    own = connection is None
    db = connection or connect_db()
    try:
        row = db.execute("SELECT name FROM collections WHERE name=?", (name,)).fetchone()
        if row:
            return str(row[0])
        alias = db.execute("SELECT collection_name FROM aliases WHERE alias_name=?", (name,)).fetchone()
        return str(alias[0]) if alias else None
    finally:
        if own:
            db.close()


def collection_info(name: str) -> sqlite3.Row | None:
    db = connect_db()
    try:
        resolved = resolve_collection(name, db)
        if resolved is None:
            return None
        return db.execute(
            "SELECT name,vector_size,distance,created_at FROM collections WHERE name=?", (resolved,)
        ).fetchone()
    finally:
        db.close()


def iter_collection_points(name: str) -> Iterator[tuple[str, bytes, str]]:
    db = connect_db()
    try:
        cursor = db.execute("SELECT id,vector,payload FROM points WHERE collection=? ORDER BY id", (name,))
        for point_id, vector, payload in cursor:
            yield str(point_id), bytes(vector), str(payload)
    finally:
        db.close()


def increment_counter(name: str) -> int:
    global _UPSERT_COUNT, _SEARCH_COUNT, _SCROLL_COUNT
    with _COUNTER_LOCK:
        if name == "upsert":
            _UPSERT_COUNT += 1
            return _UPSERT_COUNT
        if name == "scroll":
            _SCROLL_COUNT += 1
            return _SCROLL_COUNT
        _SEARCH_COUNT += 1
        return _SEARCH_COUNT


def storage_size() -> dict[str, int]:
    result: dict[str, int] = {}
    total = 0
    for label, path in (
        ("db", DB_PATH),
        ("wal", Path(str(DB_PATH) + "-wal")),
        ("shm", Path(str(DB_PATH) + "-shm")),
    ):
        size = path.stat().st_size if path.exists() else 0
        result[label] = size
        total += size
    result["total"] = total
    return result


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class QdrantHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _require_auth(self) -> None:
        if not API_KEY:
            return
        supplied = self.headers.get("api-key", "") or self.headers.get("Authorization", "").removeprefix("Bearer ")
        if supplied != API_KEY:
            raise HttpProblem(401, "Unauthorized")

    def _read_body(self) -> dict[str, Any]:
        raw_length = self.headers.get("Content-Length", "0")
        try:
            length = int(raw_length)
        except ValueError as exc:
            raise HttpProblem(400, "Invalid Content-Length") from exc
        if length < 0:
            raise HttpProblem(400, "Invalid Content-Length")
        if length > MAX_REQUEST_BYTES:
            raise HttpProblem(413, f"Request body exceeds {MAX_REQUEST_BYTES} bytes")
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HttpProblem(400, "Malformed JSON body") from exc
        if not isinstance(value, dict):
            raise HttpProblem(400, "JSON body must be an object")
        return value

    def _send_json(self, value: dict[str, Any], status: int = 200) -> None:
        try:
            body = json.dumps(
                value, ensure_ascii=False, allow_nan=False, separators=(",", ":")
            ).encode("utf-8")
        except (TypeError, ValueError):
            status = 500
            body = b'{"status":"error","error":"Response serialization failed"}'
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _dispatch(self, callback) -> None:
        try:
            self._require_auth()
            callback()
        except HttpProblem as exc:
            self._send_json({"status": "error", "error": exc.message}, exc.status)
        except sqlite3.Error as exc:
            stub_log(f"SQLITE_ERROR {type(exc).__name__}: {exc}")
            self._send_json({"status": "error", "error": "SQLite operation failed"}, 503)
        except Exception as exc:
            stub_log(f"REQUEST_ERROR {type(exc).__name__}: {exc}")
            self._send_json({"status": "error", "error": "Internal server error"}, 500)

    def do_GET(self) -> None:
        self._dispatch(self._do_get)

    def _do_get(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        if path == "/health":
            db = connect_db()
            try:
                total = int(db.execute("SELECT COUNT(*) FROM points").fetchone()[0])
                collections = int(db.execute("SELECT COUNT(*) FROM collections").fetchone()[0])
                aliases = int(db.execute("SELECT COUNT(*) FROM aliases").fetchone()[0])
            finally:
                db.close()
            self._send_json({
                "status": "ok",
                "title": "qdrant-stub-sqlite",
                "version": "2.2-r9",
                "capabilities": ["collections", "aliases", "upsert", "delete", "query", "search", "scroll"],
                "total_points": total,
                "collections": collections,
                "aliases": aliases,
                "upsert_count": _UPSERT_COUNT,
                "search_count": _SEARCH_COUNT,
                "scroll_count": _SCROLL_COUNT,
                "store_sizes": storage_size(),
                "store_backend": "sqlite-wal",
                "db_path": str(DB_PATH),
            })
            return
        if path == "/collections":
            db = connect_db()
            try:
                rows = db.execute("SELECT name FROM collections ORDER BY name").fetchall()
            finally:
                db.close()
            self._send_json({"result": {"collections": [{"name": row[0]} for row in rows]}, "status": "ok"})
            return
        if path == "/aliases":
            db = connect_db()
            try:
                rows = db.execute("SELECT alias_name,collection_name FROM aliases ORDER BY alias_name").fetchall()
            finally:
                db.close()
            self._send_json({
                "result": {"aliases": [
                    {"alias_name": row[0], "collection_name": row[1]} for row in rows
                ]},
                "status": "ok",
            })
            return
        if len(parts) == 2 and parts[0] == "collections":
            requested = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            info = collection_info(requested)
            if info is None:
                raise HttpProblem(404, "Collection not found")
            db = connect_db()
            try:
                count = int(db.execute("SELECT COUNT(*) FROM points WHERE collection=?", (info["name"],)).fetchone()[0])
            finally:
                db.close()
            self._send_json({
                "result": {
                    "status": "green",
                    "points_count": count,
                    "indexed_vectors_count": count,
                    "config": {"params": {"vectors": {
                        "size": int(info["vector_size"]), "distance": str(info["distance"])
                    }}},
                },
                "status": "ok",
            })
            return
        raise HttpProblem(404, "Not found")

    def do_PUT(self) -> None:
        self._dispatch(self._do_put)

    def _do_put(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        body = self._read_body()
        if len(parts) == 2 and parts[0] == "collections":
            name = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            vectors = body.get("vectors", {})
            if not isinstance(vectors, dict):
                raise HttpProblem(400, "vectors must be an object")
            try:
                size = int(vectors.get("size", 0))
            except (TypeError, ValueError) as exc:
                raise HttpProblem(400, "vector size must be an integer") from exc
            if size <= 0 or size > MAX_VECTOR_SIZE:
                raise HttpProblem(400, f"vector size must be between 1 and {MAX_VECTOR_SIZE}")
            distance = str(vectors.get("distance", "Cosine") or "Cosine")
            if distance.lower() != "cosine":
                raise HttpProblem(400, "Only Cosine distance is supported")
            db = connect_db()
            try:
                with db:
                    alias_collision = db.execute("SELECT 1 FROM aliases WHERE alias_name=?", (name,)).fetchone()
                    if alias_collision:
                        raise HttpProblem(409, "Name is already used by an alias")
                    existing = db.execute("SELECT vector_size,distance FROM collections WHERE name=?", (name,)).fetchone()
                    if existing and (int(existing[0]) != size or str(existing[1]).lower() != "cosine"):
                        raise HttpProblem(409, "Collection already exists with a different vector contract")
                    db.execute(
                        "INSERT OR IGNORE INTO collections(name,vector_size,distance,created_at) VALUES(?,?,?,?)",
                        (name, size, "Cosine", int(time.time())),
                    )
            finally:
                db.close()
            self._send_json({"result": True, "status": "ok"})
            return
        if len(parts) == 3 and parts[0] == "collections" and parts[2] == "points":
            requested = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            db = connect_db()
            try:
                resolved = resolve_collection(requested, db)
                if resolved is None:
                    raise HttpProblem(404, "Collection not found")
                size_row = db.execute("SELECT vector_size FROM collections WHERE name=?", (resolved,)).fetchone()
                size = int(size_row[0])
                points = body.get("points", [])
                if not isinstance(points, list):
                    raise HttpProblem(400, "points must be a list")
                if len(points) > MAX_UPSERT_POINTS:
                    raise HttpProblem(400, f"At most {MAX_UPSERT_POINTS} points per request")
                rows: list[tuple[str, str, bytes, str, int]] = []
                for point in points:
                    if not isinstance(point, dict):
                        raise HttpProblem(400, "Each point must be an object")
                    point_id = _valid_point_id(point.get("id"))
                    vector = _strict_float_vector(point.get("vector"), size)
                    _payload, payload_json = _strict_payload(point.get("payload", {}))
                    rows.append((resolved, point_id, pack_vector(vector), payload_json, int(time.time())))
                with db:
                    db.executemany(
                        "INSERT OR REPLACE INTO points(collection,id,vector,payload,updated_at) VALUES(?,?,?,?,?)",
                        rows,
                    )
            finally:
                db.close()
            operation_id = increment_counter("upsert")
            self._send_json({
                "result": {"operation_id": operation_id, "status": "completed"}, "status": "ok"
            })
            return
        raise HttpProblem(404, "Not found")

    def do_POST(self) -> None:
        self._dispatch(self._do_post)

    def _do_post(self) -> None:
        path = urlparse(self.path).path
        parts = path.strip("/").split("/")
        body = self._read_body()
        if path == "/collections/aliases":
            actions = body.get("actions", [])
            if not isinstance(actions, list) or len(actions) > 100:
                raise HttpProblem(400, "actions must be a list with at most 100 items")
            db = connect_db()
            try:
                with db:
                    # Validate all actions first so the update is atomic.
                    normalized: list[tuple[str, str, str]] = []
                    for action in actions:
                        if not isinstance(action, dict) or len(action) != 1:
                            raise HttpProblem(400, "Each alias action must be a one-key object")
                        kind, spec = next(iter(action.items()))
                        if not isinstance(spec, dict):
                            raise HttpProblem(400, "Alias action payload must be an object")
                        if kind == "create_alias":
                            alias = _valid_name(spec.get("alias_name"), kind="alias", max_len=MAX_ALIAS_NAME)
                            collection = _valid_name(spec.get("collection_name"), kind="collection", max_len=MAX_COLLECTION_NAME)
                            if not db.execute("SELECT 1 FROM collections WHERE name=?", (collection,)).fetchone():
                                raise HttpProblem(404, f"Collection not found: {collection}")
                            if db.execute("SELECT 1 FROM collections WHERE name=?", (alias,)).fetchone():
                                raise HttpProblem(409, "Alias collides with a collection name")
                            normalized.append((kind, alias, collection))
                        elif kind == "delete_alias":
                            alias = _valid_name(spec.get("alias_name"), kind="alias", max_len=MAX_ALIAS_NAME)
                            normalized.append((kind, alias, ""))
                        elif kind == "rename_alias":
                            old = _valid_name(spec.get("old_alias_name"), kind="alias", max_len=MAX_ALIAS_NAME)
                            new = _valid_name(spec.get("new_alias_name"), kind="alias", max_len=MAX_ALIAS_NAME)
                            if db.execute("SELECT 1 FROM collections WHERE name=?", (new,)).fetchone():
                                raise HttpProblem(409, "Alias collides with a collection name")
                            normalized.append((kind, old, new))
                        else:
                            raise HttpProblem(400, f"Unsupported alias action: {kind}")
                    for kind, left, right in normalized:
                        if kind == "create_alias":
                            db.execute(
                                "INSERT INTO aliases(alias_name,collection_name,created_at) VALUES(?,?,?) "
                                "ON CONFLICT(alias_name) DO UPDATE SET collection_name=excluded.collection_name",
                                (left, right, int(time.time())),
                            )
                        elif kind == "delete_alias":
                            db.execute("DELETE FROM aliases WHERE alias_name=?", (left,))
                        else:
                            row = db.execute("SELECT collection_name FROM aliases WHERE alias_name=?", (left,)).fetchone()
                            if not row:
                                raise HttpProblem(404, f"Alias not found: {left}")
                            db.execute("DELETE FROM aliases WHERE alias_name=?", (left,))
                            db.execute(
                                "INSERT INTO aliases(alias_name,collection_name,created_at) VALUES(?,?,?)",
                                (right, row[0], int(time.time())),
                            )
            finally:
                db.close()
            self._send_json({"result": True, "status": "ok"})
            return
        if len(parts) == 4 and parts[0] == "collections" and parts[2] == "points" and parts[3] == "delete":
            requested = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            point_ids = body.get("points", [])
            if not isinstance(point_ids, list):
                raise HttpProblem(400, "points must be a list")
            if len(point_ids) > MAX_DELETE_POINTS:
                raise HttpProblem(400, f"At most {MAX_DELETE_POINTS} point ids per request")
            db = connect_db()
            try:
                resolved = resolve_collection(requested, db)
                if resolved is None:
                    raise HttpProblem(404, "Collection not found")
                ids = [_valid_point_id(item) for item in point_ids]
                with db:
                    db.executemany(
                        "DELETE FROM points WHERE collection=? AND id=?",
                        [(resolved, point_id) for point_id in ids],
                    )
            finally:
                db.close()
            self._send_json({"result": {"operation_id": 0, "status": "completed"}, "status": "ok"})
            return
        if len(parts) == 4 and parts[0] == "collections" and parts[2] == "points" and parts[3] == "scroll":
            requested = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            try:
                limit = max(1, min(int(body.get("limit", 10)), MAX_SCROLL_LIMIT))
            except (TypeError, ValueError) as exc:
                raise HttpProblem(400, "limit must be an integer") from exc
            offset = body.get("offset")
            offset_text = str(offset) if offset is not None else None
            with_payload = body.get("with_payload", True)
            with_vector = bool(body.get("with_vector", False))
            db = connect_db()
            try:
                resolved = resolve_collection(requested, db)
                if resolved is None:
                    raise HttpProblem(404, "Collection not found")
                params: list[Any] = [resolved]
                where = "collection=?"
                if offset_text is not None:
                    where += " AND id>?"
                    params.append(offset_text)
                params.append(limit + 1)
                rows = db.execute(
                    f"SELECT id,vector,payload FROM points WHERE {where} ORDER BY id LIMIT ?", params
                ).fetchall()
            finally:
                db.close()
            has_more = len(rows) > limit
            page_rows = rows[:limit]
            selected_payload_keys: set[str] | None = None
            if isinstance(with_payload, list):
                selected_payload_keys = {str(item) for item in with_payload}
            points: list[dict[str, Any]] = []
            for row in page_rows:
                try:
                    payload = json.loads(row["payload"] or "{}")
                except json.JSONDecodeError:
                    payload = {}
                if with_payload is False:
                    payload = None
                elif selected_payload_keys is not None:
                    payload = {key: payload.get(key) for key in selected_payload_keys if key in payload}
                point: dict[str, Any] = {"id": str(row["id"])}
                if payload is not None:
                    point["payload"] = payload
                if with_vector:
                    point["vector"] = list(unpack_vector(row["vector"]))
                points.append(point)
            next_offset = str(page_rows[-1]["id"]) if has_more and page_rows else None
            increment_counter("scroll")
            self._send_json({
                "result": {"points": points, "next_page_offset": next_offset}, "status": "ok"
            })
            return
        if len(parts) == 4 and parts[0] == "collections" and parts[2] == "points" and parts[3] in {"query", "search"}:
            requested = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            db = connect_db()
            try:
                resolved = resolve_collection(requested, db)
                if resolved is None:
                    raise HttpProblem(404, "Collection not found")
                size = int(db.execute("SELECT vector_size FROM collections WHERE name=?", (resolved,)).fetchone()[0])
            finally:
                db.close()
            key = "query" if parts[3] == "query" else "vector"
            query = _strict_float_vector(body.get(key, []), size)
            try:
                limit = max(1, min(int(body.get("limit", 10)), MAX_QUERY_LIMIT))
            except (TypeError, ValueError) as exc:
                raise HttpProblem(400, "limit must be an integer") from exc
            with_payload = body.get("with_payload", True)
            payload_filter = body.get("filter")
            score_threshold_raw = body.get("score_threshold")
            try:
                score_threshold = None if score_threshold_raw is None else float(score_threshold_raw)
            except (TypeError, ValueError) as exc:
                raise HttpProblem(400, "score_threshold must be numeric") from exc
            selected_payload_keys: set[str] | None = None
            if isinstance(with_payload, list):
                selected_payload_keys = {str(item) for item in with_payload}
            heap: list[tuple[float, int, str, dict[str, Any]]] = []
            sequence = 0
            for point_id, vector_blob, payload_json in iter_collection_points(resolved):
                # R12: reject non-matching tenant/model/context payloads before
                # unpacking a 4096-dimensional vector and computing cosine.
                # This preserves exact Qdrant semantics while avoiding the most
                # expensive work for points excluded by the request filter.
                try:
                    payload = json.loads(payload_json) if payload_json else {}
                except json.JSONDecodeError:
                    payload = {}
                if not _payload_matches_filter(payload, payload_filter):
                    continue
                try:
                    candidate = unpack_vector(vector_blob)
                    if len(candidate) != size:
                        raise ValueError("dimension mismatch")
                    score = cosine_similarity(query, candidate)
                except Exception as exc:
                    stub_log(f"CORRUPT_VECTOR collection={resolved} id={point_id} error={exc}")
                    continue
                if score_threshold is not None and score < score_threshold:
                    continue
                if with_payload is False:
                    payload = {}
                elif selected_payload_keys is not None:
                    payload = {key_: payload.get(key_) for key_ in selected_payload_keys if key_ in payload}
                item = (score, sequence, point_id, payload)
                sequence += 1
                if len(heap) < limit:
                    heapq.heappush(heap, item)
                elif score > heap[0][0]:
                    heapq.heapreplace(heap, item)
            points = [
                {"id": point_id, "score": score, "payload": payload}
                for score, _seq, point_id, payload in sorted(heap, reverse=True)
            ]
            increment_counter("search")
            if parts[3] == "query":
                self._send_json({"result": {"points": points}, "status": "ok"})
            else:
                self._send_json({"result": points, "status": "ok"})
            return
        raise HttpProblem(404, "Not found")

    def do_DELETE(self) -> None:
        self._dispatch(self._do_delete)

    def _do_delete(self) -> None:
        parts = urlparse(self.path).path.strip("/").split("/")
        if len(parts) == 2 and parts[0] == "collections":
            name = _valid_name(parts[1], kind="collection", max_len=MAX_COLLECTION_NAME)
            db = connect_db()
            try:
                # Deleting through an alias is intentionally rejected.
                if db.execute("SELECT 1 FROM aliases WHERE alias_name=?", (name,)).fetchone():
                    raise HttpProblem(400, "Delete the alias via /collections/aliases")
                if not db.execute("SELECT 1 FROM collections WHERE name=?", (name,)).fetchone():
                    raise HttpProblem(404, "Collection not found")
                with db:
                    db.execute("DELETE FROM collections WHERE name=?", (name,))
            finally:
                db.close()
            self._send_json({"result": True, "status": "ok"})
            return
        raise HttpProblem(404, "Not found")

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def run(port: int = DEFAULT_PORT) -> None:
    initialize_db()
    server = ThreadingHTTPServer((BIND_HOST, port), QdrantHandler)
    stop_event = threading.Event()

    def stop(_signum, _frame) -> None:
        if stop_event.is_set():
            return
        stop_event.set()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    stub_log(f"START host={BIND_HOST} port={port} db={DB_PATH} size={storage_size()['total']}b")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        stub_log("STOP")


if __name__ == "__main__":
    run()
