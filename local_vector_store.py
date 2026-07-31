#!/usr/bin/env python3
"""
local_vector_store.py — Локальное хранилище векторов (stdlib-only).
Заменяет embed_stub (:4000) + qdrant_stub (:6333) прямыми вызовами.

v1.0 — 2026-06-27:
  + Character n-gram MD5 hashing (768 dim) — ИДЕНТИЧНО embed_stub.py
  + SQLite-хранение (WAL-журнал)
  + Cosine similarity поиск
  + Миграция из vectors.json → SQLite
"""
import hashlib, json, math, os, re, sqlite3, struct, threading, time
from collections import Counter

VECTOR_SIZE = 768
NGRAM_SIZES = [2, 3, 4]
DEFAULT_DB = os.path.expanduser("~/.hermes/knowledge_db/local_vectors.db")
JSON_STORE = os.path.expanduser("~/.hermes/knowledge_db/vectors.json")

def tokenize(text): return re.findall(r'\w+', text.lower())
def extract_ngrams(word, n): return [word] if len(word) < n else [word[i:i+n] for i in range(len(word) - n + 1)]
def hash_ngram(ngram): return int.from_bytes(hashlib.md5(ngram.encode()).digest()[:4], 'big') % VECTOR_SIZE

def text_to_vector(text, idf=None):
    words = tokenize(text)
    if not words: return [0.0] * VECTOR_SIZE
    ngram_counts = Counter()
    for word in words:
        for n in NGRAM_SIZES:
            for ng in extract_ngrams(word, n): ngram_counts[ng] += 1
    if not ngram_counts: return [0.0] * VECTOR_SIZE
    vector = [0.0] * VECTOR_SIZE; max_tf = max(ngram_counts.values())
    for ng, count in ngram_counts.items():
        idx = hash_ngram(ng); tf = count / max_tf
        if idf and ng in idf: tf *= idf[ng]
        vector[idx] += tf
    norm = math.sqrt(sum(v * v for v in vector))
    if norm > 0: vector = [v / norm for v in vector]
    return vector

def cosine_similarity(a, b):
    min_len = min(len(a), len(b)); a, b = a[:min_len], b[:min_len]
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a)); norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0: return 0.0
    return dot / (norm_a * norm_b)

class LocalVectorStore:
    def __init__(self, db_path=DEFAULT_DB):
        self.db_path = db_path; self._lock = threading.Lock(); self._init_db()
    def _connect(self):
        conn = sqlite3.connect(self.db_path); conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL"); conn.execute("PRAGMA synchronous=NORMAL"); conn.execute("PRAGMA busy_timeout=5000")
        return conn
    def _init_db(self):
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute("CREATE TABLE IF NOT EXISTS collections (name TEXT PRIMARY KEY, vector_size INTEGER DEFAULT 768, created_at TEXT DEFAULT (datetime('now')), next_id INTEGER DEFAULT 0)")
                conn.execute("CREATE TABLE IF NOT EXISTS vectors (collection_name TEXT NOT NULL, point_id TEXT NOT NULL, vec BLOB NOT NULL, dims INTEGER NOT NULL, payload TEXT DEFAULT '{}', created_at TEXT DEFAULT (datetime('now')), PRIMARY KEY (collection_name, point_id))")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_vectors_collection ON vectors(collection_name)")
                conn.execute("CREATE TABLE IF NOT EXISTS store_meta (key TEXT PRIMARY KEY, value TEXT)")
            conn.close()
    def health(self):
        try:
            conn = self._connect()
            cols = conn.execute("SELECT name, vector_size FROM collections").fetchall()
            total = conn.execute("SELECT count(*) n FROM vectors").fetchone()['n']
            db_size = os.path.getsize(self.db_path) if os.path.exists(self.db_path) else 0
            conn.close()
            return {"status": "ok", "version": "1.0", "collections": [{"name": r['name'], "vector_size": r['vector_size']} for r in cols], "total_vectors": total, "db_size": db_size, "stdlib_only": True}
        except Exception as e: return {"status": "error", "error": str(e)}
    def embed(self, text):
        if isinstance(text, list): return [text_to_vector(str(t)[:8000]) for t in text]
        return text_to_vector(str(text)[:8000])
    def create_collection(self, name, vector_size=768):
        with self._lock:
            conn = self._connect()
            with conn: conn.execute("INSERT OR IGNORE INTO collections(name, vector_size) VALUES(?,?)", (name, vector_size))
            conn.close()
        return True
    def upsert(self, collection, points):
        with self._lock:
            conn = self._connect()
            with conn:
                conn.execute("INSERT OR IGNORE INTO collections(name, vector_size) VALUES(?,768)", (collection,))
                for pt in points:
                    pid = str(pt.get("id", "")); vec = pt.get("vector", [])
                    payload = json.dumps(pt.get("payload", {}), ensure_ascii=False)
                    dims = len(vec); packed = struct.pack(f'{dims}f', *vec)
                    conn.execute("INSERT OR REPLACE INTO vectors(collection_name, point_id, vec, dims, payload) VALUES(?,?,?,?,?)", (collection, pid, packed, dims, payload))
            conn.close()
        return {"status": "completed", "points_upserted": len(points)}
    def replace_collection(self, collection, points):
        """Atomically replace a bounded collection in one SQLite transaction."""
        collection = str(collection).strip()
        if not collection or len(collection) > 128:
            raise ValueError("Collection name length must be between 1 and 128")
        prepared = []
        for point in points:
            point_id = str(point.get("id", "")).strip()
            vector = point.get("vector", [])
            payload = point.get("payload", {})
            if not point_id or len(point_id) > 256:
                raise ValueError("Point id length must be between 1 and 256")
            if not isinstance(vector, list) or len(vector) != VECTOR_SIZE:
                raise ValueError(f"Vector size must be {VECTOR_SIZE}")
            normalized_vector = [float(value) for value in vector]
            if not all(math.isfinite(value) for value in normalized_vector):
                raise ValueError("Vector values must be finite")
            if not isinstance(payload, dict):
                raise ValueError("Point payload must be an object")
            payload_json = json.dumps(
                payload,
                ensure_ascii=False,
                allow_nan=False,
                separators=(",", ":"),
            )
            if len(payload_json.encode("utf-8")) > 2 * 1024 * 1024:
                raise ValueError("Point payload exceeds 2 MiB")
            prepared.append((
                collection,
                point_id,
                struct.pack(f"{VECTOR_SIZE}f", *normalized_vector),
                VECTOR_SIZE,
                payload_json,
            ))
        with self._lock:
            conn = self._connect()
            try:
                with conn:
                    existing = conn.execute(
                        "SELECT vector_size FROM collections WHERE name = ?",
                        (collection,),
                    ).fetchone()
                    if existing and int(existing[0]) != VECTOR_SIZE:
                        raise ValueError(
                            f"Collection vector size is {existing[0]}, expected {VECTOR_SIZE}"
                        )
                    conn.execute(
                        "INSERT OR IGNORE INTO collections(name, vector_size) VALUES(?, ?)",
                        (collection, VECTOR_SIZE),
                    )
                    conn.execute(
                        "DELETE FROM vectors WHERE collection_name = ?", (collection,)
                    )
                    conn.executemany(
                        "INSERT INTO vectors(collection_name, point_id, vec, dims, payload) VALUES(?, ?, ?, ?, ?)",
                        prepared,
                    )
            finally:
                conn.close()
        return {"status": "completed", "points_replaced": len(prepared)}

    def search(self, collection, query_vector, limit=5, filter_type=None):
        limit = max(1, min(limit, 100))
        conn = self._connect()
        rows = conn.execute("SELECT point_id, vec, dims, payload FROM vectors WHERE collection_name=?", (collection,)).fetchall()
        conn.close()
        results = []
        for row in rows:
            try:
                payload = json.loads(row['payload']) if row['payload'] else {}
                if filter_type and payload.get("type") != filter_type: continue
                dims = row['dims']; packed = row['vec']
                if isinstance(packed, bytes):
                    r_vec = list(struct.unpack(f'{dimensions}f', packed[:dimensions*4])) if (dimensions := dims) else []
                    if r_vec: results.append({"id": row['point_id'], "score": cosine_similarity(query_vector, r_vec), "payload": payload})
            except Exception: continue
        results.sort(key=lambda x: -x["score"]); return results[:limit]
    def scroll(self, collection, limit=500):
        limit = min(limit, 2000)
        conn = self._connect()
        rows = conn.execute("SELECT point_id, payload FROM vectors WHERE collection_name=? LIMIT ?", (collection, limit)).fetchall()
        conn.close()
        return [{"id": r['point_id'], "payload": json.loads(r['payload']) if r['payload'] else {}} for r in rows]
    def get_collection_info(self, name):
        conn = self._connect()
        col = conn.execute("SELECT * FROM collections WHERE name=?", (name,)).fetchone()
        count = conn.execute("SELECT count(*) n FROM vectors WHERE collection_name=?", (name,)).fetchone()['n']
        conn.close()
        if not col: return {"error": "Collection not found"}
        return {"status": "green", "points_count": count, "indexed_vectors_count": count, "config": {"params": {"vectors": {"size": col["vector_size"]}}}}
    def migrate_from_json(self, json_path=None, collection="evey-knowledge", dry_run=False):
        path = json_path or JSON_STORE
        if not os.path.exists(path): return {"error": f"File not found: {path}"}
        try:
            with open(path) as f: data = json.load(f)
        except Exception as e: return {"error": f"JSON load error: {e}"}
        collections = data.get("collections", {})
        stats = {name: {"source_points": len(col["points"]), "migrated": 0, "skipped": 0} for name, col in collections.items()}
        total_points = sum(len(col["points"]) for col in collections.values())
        if dry_run: return {"dry_run": True, "collections": stats, "total_points": total_points}
        migrated = skipped = 0
        with self._lock:
            conn = self._connect()
            with conn:
                for col_name, col_data in collections.items():
                    vector_size = col_data.get("vector_size", 768)
                    conn.execute("INSERT OR IGNORE INTO collections(name, vector_size) VALUES(?,?)", (col_name, vector_size))
                    points = col_data.get("points", {})
                    for pid, pt in points.items():
                        vec = pt.get("vector", []); payload = pt.get("payload", {})
                        if not vec: skipped += 1; continue
                        dims = len(vec); packed = struct.pack(f'{dims}f', *vec)
                        payload_json = json.dumps(payload, ensure_ascii=False)
                        conn.execute("INSERT OR REPLACE INTO vectors(collection_name, point_id, vec, dims, payload) VALUES(?,?,?,?,?)", (col_name, str(pid), packed, dims, payload_json))
                        migrated += 1
                    max_id = max((int(pid) for pid in points.keys() if pid.isdigit()), default=0)
                    conn.execute("UPDATE collections SET next_id=? WHERE name=?", (max_id + 1, col_name))
                conn.execute("INSERT OR REPLACE INTO store_meta(key, value) VALUES(?,?)", ("migration_source", path))
                conn.execute("INSERT OR REPLACE INTO store_meta(key, value) VALUES(?,?)", ("migration_date", time.strftime('%Y-%m-%dT%H:%M:%S')))
            conn.close()
        return {"status": "completed", "migrated": migrated, "skipped": skipped, "total_points": total_points}

_vector_store: LocalVectorStore | None = None
_store_lock = threading.Lock()

def get_store(db_path=None):
    global _vector_store
    if _vector_store is None:
        with _store_lock:
            if _vector_store is None: _vector_store = LocalVectorStore(db_path or DEFAULT_DB)
    return _vector_store

if __name__ == "__main__":
    import sys
    store = get_store()
    if len(sys.argv) < 2:
        print(json.dumps(store.health(), indent=2)); sys.exit(0)
    cmd = sys.argv[1]
    if cmd == "health": print(json.dumps(store.health(), indent=2))
    elif cmd == "embed":
        text = " ".join(sys.argv[2:]) or "test"
        vec = store.embed(text)
        print(f"Vector {len(vec)}d, norm={math.sqrt(sum(v*v for v in vec)):.4f}, non-zero={sum(1 for v in vec if v!=0)}")
    elif cmd == "migrate":
        dry_run = "--dry-run" in sys.argv
        result = store.migrate_from_json(dry_run=dry_run)
        print(json.dumps(result, indent=2))
    elif cmd == "search":
        query = " ".join(sys.argv[2:]) or "hermes config"
        vec = store.embed(query)
        results = store.search("evey-knowledge", vec, limit=5)
        for r in results: print(f"  {r['score']:.3f} | {r['payload'].get('title', r['id'])[:80]}")
    else: print(f"Unknown: {cmd}\nCommands: health, embed <text>, migrate [--dry-run], search <query>")
