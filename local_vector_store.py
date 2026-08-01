#!/usr/bin/env python3
"""Strict stdlib-only vector store shared by Hermes maintenance scripts.

This module is not used by the HTTP stubs at runtime, but its contracts now match
embed_stub.py and Memory Wiki: configurable 2560 dimensions, no silent vector
truncation, finite-value validation, HERMES_HOME-aware storage, and bounded
streaming search.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import sqlite3
import struct
import threading
import time
from collections import Counter
from contextlib import closing
from pathlib import Path
from typing import Any, Iterator

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser()
VECTOR_SIZE = max(8, min(int(os.environ.get("LOCAL_VECTOR_SIZE", os.environ.get("MEMORY_WIKI_VECTOR_SIZE", "2560"))), 65536))
NGRAM_SIZES = (2, 3, 4)
DEFAULT_DB = Path(os.environ.get("LOCAL_VECTOR_DB", str(HERMES_HOME / "knowledge_db" / "local_vectors.db"))).expanduser()
JSON_STORE = Path(os.environ.get("LOCAL_VECTOR_JSON", str(HERMES_HOME / "knowledge_db" / "vectors.json"))).expanduser()
MAX_PAYLOAD_BYTES = 2 * 1024 * 1024
MAX_POINT_ID = 512
MAX_COLLECTION_NAME = 200


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def extract_ngrams(word: str, n: int) -> list[str]:
    return [word] if len(word) < n else [word[i:i+n] for i in range(len(word) - n + 1)]


def hash_ngram(ngram: str, dimensions: int = VECTOR_SIZE) -> int:
    return int.from_bytes(hashlib.md5(ngram.encode("utf-8")).digest()[:4], "big") % int(dimensions)


def text_to_vector(text: str, dimensions: int = VECTOR_SIZE) -> list[float]:
    words = tokenize(str(text)[:12000])
    if not words:
        return [0.0] * VECTOR_SIZE
    counts: Counter[str] = Counter()
    for word in words:
        for size in NGRAM_SIZES:
            for ngram in extract_ngrams(word, size):
                counts[ngram] += 1
    dimensions = int(dimensions)
    if dimensions < 1:
        raise ValueError("Embedding dimensions must be positive")
    vector = [0.0] * dimensions
    maximum = max(counts.values(), default=1)
    for ngram, count in counts.items():
        vector[hash_ngram(ngram, dimensions)] += count / maximum
    norm = math.sqrt(sum(value * value for value in vector))
    return [value / norm for value in vector] if norm else vector


def strict_vector(value: Any, expected: int = VECTOR_SIZE) -> list[float]:
    if not isinstance(value, list) or len(value) != expected:
        actual = len(value) if isinstance(value, list) else 0
        raise ValueError(f"Vector size must be {expected}; got {actual}")
    vector = [float(item) for item in value]
    if not all(math.isfinite(item) for item in vector):
        raise ValueError("Vector values must be finite")
    return vector


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if len(left) != len(right):
        raise ValueError(f"Vector dimensions differ: {len(left)} != {len(right)}")
    dot = sum(a * b for a, b in zip(left, right))
    norm_left = math.sqrt(sum(item * item for item in left))
    norm_right = math.sqrt(sum(item * item for item in right))
    if norm_left == 0 or norm_right == 0:
        return 0.0
    score = dot / (norm_left * norm_right)
    return score if math.isfinite(score) else 0.0


def valid_name(value: Any, max_len: int, label: str) -> str:
    text = str(value or "").strip()
    if not text or len(text) > max_len or any(ch in text for ch in "/\\\x00"):
        raise ValueError(f"Invalid {label}")
    return text


def payload_json(value: Any) -> str:
    if not isinstance(value, dict):
        raise ValueError("Point payload must be an object")
    encoded = json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_PAYLOAD_BYTES:
        raise ValueError("Point payload exceeds 2 MiB")
    return encoded


class LocalVectorStore:
    def __init__(self, db_path: str | os.PathLike[str] = DEFAULT_DB, vector_size: int = VECTOR_SIZE):
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.vector_size = int(vector_size)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=30000")
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn, conn:
            conn.execute(
                """CREATE TABLE IF NOT EXISTS collections(
                    name TEXT PRIMARY KEY,
                    vector_size INTEGER NOT NULL,
                    created_at INTEGER NOT NULL DEFAULT 0
                )"""
            )
            conn.execute(
                """CREATE TABLE IF NOT EXISTS vectors(
                    collection_name TEXT NOT NULL,
                    point_id TEXT NOT NULL,
                    vec BLOB NOT NULL,
                    dims INTEGER NOT NULL,
                    payload TEXT NOT NULL DEFAULT '{}',
                    updated_at INTEGER NOT NULL DEFAULT 0,
                    PRIMARY KEY(collection_name,point_id),
                    FOREIGN KEY(collection_name) REFERENCES collections(name) ON DELETE CASCADE
                )"""
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection_name,point_id)")

    def health(self) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            collections = [dict(row) for row in conn.execute("SELECT name,vector_size FROM collections ORDER BY name")]
            total = int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])
        sizes = {}
        for label, path in (("db", self.db_path), ("wal", Path(str(self.db_path)+"-wal")), ("shm", Path(str(self.db_path)+"-shm"))):
            sizes[label] = path.stat().st_size if path.exists() else 0
        sizes["total"] = sum(sizes.values())
        return {
            "status": "ok", "version": "2.0-r9", "vector_size": self.vector_size,
            "collections": collections, "total_vectors": total, "store_sizes": sizes,
            "db_path": str(self.db_path), "stdlib_only": True,
        }

    def embed(self, text: str | list[str]) -> list[float] | list[list[float]]:
        return (
            [text_to_vector(str(item), self.vector_size) for item in text]
            if isinstance(text, list)
            else text_to_vector(str(text), self.vector_size)
        )

    def create_collection(self, name: str, vector_size: int | None = None) -> bool:
        name = valid_name(name, MAX_COLLECTION_NAME, "collection name")
        size = int(vector_size or self.vector_size)
        with self._lock, closing(self._connect()) as conn, conn:
            existing = conn.execute("SELECT vector_size FROM collections WHERE name=?", (name,)).fetchone()
            if existing and int(existing[0]) != size:
                raise ValueError(f"Collection vector size is {existing[0]}, requested {size}")
            conn.execute(
                "INSERT OR IGNORE INTO collections(name,vector_size,created_at) VALUES(?,?,?)",
                (name, size, int(time.time())),
            )
        return True

    def _prepare_point(self, collection: str, point: dict[str, Any], size: int) -> tuple[Any, ...]:
        point_id = valid_name(point.get("id"), MAX_POINT_ID, "point id")
        vector = strict_vector(point.get("vector"), size)
        encoded = payload_json(point.get("payload", {}))
        return collection, point_id, struct.pack(f">{size}f", *vector), size, encoded, int(time.time())

    def upsert(self, collection: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        collection = valid_name(collection, MAX_COLLECTION_NAME, "collection name")
        if len(points) > 500:
            raise ValueError("At most 500 points per upsert")
        with self._lock, closing(self._connect()) as conn:
            existing = conn.execute("SELECT vector_size FROM collections WHERE name=?", (collection,)).fetchone()
            size = int(existing[0]) if existing else self.vector_size
            rows = [self._prepare_point(collection, point, size) for point in points]
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO collections(name,vector_size,created_at) VALUES(?,?,?)",
                    (collection, size, int(time.time())),
                )
                conn.executemany(
                    "INSERT OR REPLACE INTO vectors(collection_name,point_id,vec,dims,payload,updated_at) VALUES(?,?,?,?,?,?)",
                    rows,
                )
        return {"status": "completed", "points_upserted": len(rows)}

    def replace_collection(self, collection: str, points: list[dict[str, Any]]) -> dict[str, Any]:
        collection = valid_name(collection, MAX_COLLECTION_NAME, "collection name")
        rows = [self._prepare_point(collection, point, self.vector_size) for point in points]
        with self._lock, closing(self._connect()) as conn:
            existing = conn.execute("SELECT vector_size FROM collections WHERE name=?", (collection,)).fetchone()
            if existing and int(existing[0]) != self.vector_size:
                raise ValueError(f"Collection vector size is {existing[0]}, expected {self.vector_size}")
            with conn:
                conn.execute(
                    "INSERT OR IGNORE INTO collections(name,vector_size,created_at) VALUES(?,?,?)",
                    (collection, self.vector_size, int(time.time())),
                )
                conn.execute("DELETE FROM vectors WHERE collection_name=?", (collection,))
                conn.executemany(
                    "INSERT INTO vectors(collection_name,point_id,vec,dims,payload,updated_at) VALUES(?,?,?,?,?,?)",
                    rows,
                )
        return {"status": "completed", "points_replaced": len(rows)}

    def _iter_rows(self, collection: str) -> Iterator[sqlite3.Row]:
        conn = self._connect()
        try:
            cursor = conn.execute(
                "SELECT point_id,vec,dims,payload FROM vectors WHERE collection_name=? ORDER BY point_id",
                (collection,),
            )
            yield from cursor
        finally:
            conn.close()

    def search(self, collection: str, query_vector: list[float], limit: int = 5, filter_type: str | None = None) -> list[dict[str, Any]]:
        collection = valid_name(collection, MAX_COLLECTION_NAME, "collection name")
        query = strict_vector(query_vector, self.vector_size)
        limit = max(1, min(int(limit), 100))
        results: list[dict[str, Any]] = []
        for row in self._iter_rows(collection):
            try:
                dims = int(row["dims"])
                if dims != self.vector_size:
                    continue
                candidate = list(struct.unpack(f">{dims}f", row["vec"]))
                payload = json.loads(row["payload"] or "{}")
                if filter_type and payload.get("type") != filter_type:
                    continue
                results.append({
                    "id": str(row["point_id"]),
                    "score": cosine_similarity(query, candidate),
                    "payload": payload,
                })
            except (ValueError, struct.error, json.JSONDecodeError):
                continue
        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:limit]

    def scroll(self, collection: str, limit: int = 500, offset: str | None = None) -> dict[str, Any]:
        collection = valid_name(collection, MAX_COLLECTION_NAME, "collection name")
        limit = max(1, min(int(limit), 2000))
        with closing(self._connect()) as conn:
            if offset is None:
                rows = conn.execute(
                    "SELECT point_id,payload FROM vectors WHERE collection_name=? ORDER BY point_id LIMIT ?",
                    (collection, limit + 1),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT point_id,payload FROM vectors WHERE collection_name=? AND point_id>? ORDER BY point_id LIMIT ?",
                    (collection, str(offset), limit + 1),
                ).fetchall()
        has_more = len(rows) > limit
        rows = rows[:limit]
        return {
            "points": [
                {"id": str(row["point_id"]), "payload": json.loads(row["payload"] or "{}")}
                for row in rows
            ],
            "next_page_offset": str(rows[-1]["point_id"]) if has_more and rows else None,
        }

    def get_collection_info(self, name: str) -> dict[str, Any]:
        with closing(self._connect()) as conn:
            row = conn.execute("SELECT vector_size FROM collections WHERE name=?", (name,)).fetchone()
            if not row:
                return {"error": "Collection not found"}
            count = int(conn.execute("SELECT COUNT(*) FROM vectors WHERE collection_name=?", (name,)).fetchone()[0])
        return {
            "status": "green", "points_count": count, "indexed_vectors_count": count,
            "config": {"params": {"vectors": {"size": int(row[0]), "distance": "Cosine"}}},
        }


_STORES: dict[str, LocalVectorStore] = {}
_STORE_LOCK = threading.Lock()


def get_store(db_path: str | os.PathLike[str] | None = None) -> LocalVectorStore:
    path = str(Path(db_path or DEFAULT_DB).expanduser().resolve())
    with _STORE_LOCK:
        if path not in _STORES:
            _STORES[path] = LocalVectorStore(path)
        return _STORES[path]


if __name__ == "__main__":
    import sys
    store = get_store()
    command = sys.argv[1] if len(sys.argv) > 1 else "health"
    if command == "health":
        print(json.dumps(store.health(), indent=2, ensure_ascii=False))
    elif command == "embed":
        vector = store.embed(" ".join(sys.argv[2:]) or "test")
        assert isinstance(vector, list)
        print(f"Vector {len(vector)}d, norm={math.sqrt(sum(v*v for v in vector)):.4f}")
    else:
        raise SystemExit("Commands: health, embed <text>")
