#!/usr/bin/env python3
"""Hermes Qdrant-compatible SQLite stub, R17 state-contract edition.

Stdlib-only, Android/proot friendly. Implements the subset used by Memory Wiki
and the DeepSeek semantic cache: collections, atomic aliases, upsert/delete,
query/search, filters, scroll, payload selectors and WAL persistence.
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sqlite3
import struct
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse

VERSION = "2.3-r17-state-contract"
DEFAULT_DIM = int(os.environ.get("MEMORY_WIKI_VECTOR_SIZE") or os.environ.get("MEMORY_WIKI_EMBED_DIMENSIONS") or "4096")
MAX_BODY = int(os.environ.get("HERMES_QDRANT_MAX_BODY_BYTES", str(128 * 1024 * 1024)))


def stable_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pack_vector(vector, expected):
    if not isinstance(vector, list) or len(vector) != expected:
        raise ValueError(f"vector dimension mismatch: expected {expected}, got {len(vector) if isinstance(vector,list) else 'non-list'}")
    vals = []
    for item in vector:
        x = float(item)
        if not math.isfinite(x):
            raise ValueError("vector contains non-finite value")
        vals.append(x)
    return struct.pack(f"<{expected}f", *vals)


def unpack_vector(blob, size):
    return list(struct.unpack(f"<{size}f", bytes(blob)))


def cosine(a, b):
    dot = na = nb = 0.0
    for x, y in zip(a, b):
        dot += x * y; na += x * x; nb += y * y
    if na <= 0.0 or nb <= 0.0:
        return 0.0
    return dot / math.sqrt(na * nb)


def get_path(payload, key):
    cur = payload
    for part in str(key or "").split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


def match_condition(payload, point_id, cond):
    if not isinstance(cond, dict):
        return True
    if "has_id" in cond:
        return str(point_id) in {str(v) for v in (cond.get("has_id") or [])}
    if "is_empty" in cond:
        spec = cond.get("is_empty") or {}; value = get_path(payload, spec.get("key"))
        return value in (None, "", [], {})
    if "is_null" in cond:
        spec = cond.get("is_null") or {}; return get_path(payload, spec.get("key")) is None
    if "nested" in cond:
        spec = cond.get("nested") or {}; value = get_path(payload, spec.get("key"))
        nested_filter = spec.get("filter") or {}
        if isinstance(value, list):
            return any(match_filter(v if isinstance(v, dict) else {"value": v}, point_id, nested_filter) for v in value)
        return isinstance(value, dict) and match_filter(value, point_id, nested_filter)
    key = cond.get("key")
    value = get_path(payload, key)
    if "match" in cond:
        spec = cond.get("match") or {}
        if "value" in spec: return value == spec.get("value")
        if "any" in spec:
            wanted = set(map(str, spec.get("any") or []))
            if isinstance(value, list): return any(str(v) in wanted for v in value)
            return str(value) in wanted
        if "except" in spec:
            blocked = set(map(str, spec.get("except") or []))
            if isinstance(value, list): return all(str(v) not in blocked for v in value)
            return str(value) not in blocked
        if "text" in spec: return str(spec.get("text") or "").lower() in str(value or "").lower()
    if "range" in cond:
        try: x = float(value)
        except Exception: return False
        r = cond.get("range") or {}
        if "gt" in r and not x > float(r["gt"]): return False
        if "gte" in r and not x >= float(r["gte"]): return False
        if "lt" in r and not x < float(r["lt"]): return False
        if "lte" in r and not x <= float(r["lte"]): return False
        return True
    if "values_count" in cond:
        n = len(value) if isinstance(value, (list, dict, str)) else 0
        return match_condition({"n": n}, point_id, {"key":"n", "range":cond.get("values_count")})
    return True


def match_filter(payload, point_id, filt):
    if not filt: return True
    if not isinstance(filt, dict): return False
    must = filt.get("must") or []
    must_not = filt.get("must_not") or []
    should = filt.get("should") or []
    if any(not match_condition(payload, point_id, c) for c in must): return False
    if any(match_condition(payload, point_id, c) for c in must_not): return False
    if should:
        minimum = int((filt.get("min_should") or {}).get("min_count", 1)) if isinstance(filt.get("min_should"), dict) else 1
        if sum(1 for c in should if match_condition(payload, point_id, c)) < minimum: return False
    return True


def select_payload(payload, selector):
    if selector is False or selector is None: return None
    if selector is True: return payload
    if isinstance(selector, list): return {k: get_path(payload, k) for k in selector if get_path(payload, k) is not None}
    if isinstance(selector, dict):
        include = selector.get("include")
        exclude = set(selector.get("exclude") or [])
        out = dict(payload) if not include else {k: get_path(payload, k) for k in include if get_path(payload, k) is not None}
        for k in exclude: out.pop(k, None)
        return out
    return payload


class Store:
    def __init__(self, db_path):
        self.path = str(db_path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=NORMAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.db.execute("PRAGMA busy_timeout=5000")
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS collections(name TEXT PRIMARY KEY,size INTEGER NOT NULL,distance TEXT NOT NULL,created_at INTEGER NOT NULL);
        CREATE TABLE IF NOT EXISTS aliases(alias TEXT PRIMARY KEY,collection_name TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE);
        CREATE INDEX IF NOT EXISTS aliases_collection_idx ON aliases(collection_name);
        CREATE TABLE IF NOT EXISTS points(collection_name TEXT NOT NULL REFERENCES collections(name) ON DELETE CASCADE,id TEXT NOT NULL,vector BLOB NOT NULL,payload_json TEXT NOT NULL DEFAULT '{}',updated_at INTEGER NOT NULL,PRIMARY KEY(collection_name,id));
        CREATE INDEX IF NOT EXISTS points_collection_idx ON points(collection_name,id);
        """)
        try: os.chmod(self.path, 0o600)
        except Exception: pass

    def resolve(self, name):
        row = self.db.execute("SELECT collection_name FROM aliases WHERE alias=?", (name,)).fetchone()
        return str(row[0]) if row else name

    def collection(self, name):
        actual = self.resolve(name)
        row = self.db.execute("SELECT * FROM collections WHERE name=?", (actual,)).fetchone()
        return actual, row

    def bootstrap(self):
        alias = os.environ.get("MEMORY_WIKI_QDRANT_COLLECTION", "").strip()
        if not alias: return
        with self.lock:
            if self.db.execute("SELECT 1 FROM collections WHERE name=?", (alias,)).fetchone(): return
            if self.db.execute("SELECT 1 FROM aliases WHERE alias=?", (alias,)).fetchone(): return
            physical = f"{alias}__bootstrap_{DEFAULT_DIM}d"
            now = int(time.time())
            self.db.execute("BEGIN IMMEDIATE")
            try:
                self.db.execute("INSERT OR IGNORE INTO collections VALUES(?,?,?,?)", (physical, DEFAULT_DIM, "Cosine", now))
                self.db.execute("INSERT OR IGNORE INTO aliases VALUES(?,?)", (alias, physical))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK"); raise


class Handler(BaseHTTPRequestHandler):
    server_version = "HermesQdrantStub/" + VERSION
    def log_message(self, fmt, *args):
        if os.environ.get("HERMES_QDRANT_QUIET", "0") != "1": super().log_message(fmt, *args)
    @property
    def store(self): return self.server.store
    def read_json(self):
        length = int(self.headers.get("content-length") or 0)
        if length > MAX_BODY: raise ValueError("request body too large")
        raw = self.rfile.read(length) if length else b"{}"
        return json.loads(raw.decode("utf-8") or "{}")
    def send_json(self, status, payload):
        raw = json.dumps(payload, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(raw))); self.end_headers(); self.wfile.write(raw)
    def ok(self, result=True, status=200): self.send_json(status, {"result": result, "status":"ok", "time":0})
    def error(self, status, message): self.send_json(status, {"status":{"error":str(message)},"result":None})
    def route(self): return [unquote(p) for p in urlparse(self.path).path.split("/") if p]
    def do_HEAD(self): self.do_GET(head=True)
    def do_GET(self, head=False):
        try:
            parts = self.route()
            if not parts or parts[0] in {"healthz","readyz","livez"}:
                payload={"ok":True,"version":VERSION,"vector_size":DEFAULT_DIM,"sqlite_wal":True}
                if head: self.send_response(200); self.end_headers()
                else: self.send_json(200,payload)
                return
            if parts == ["collections"]:
                rows=self.store.db.execute("SELECT name,size,distance FROM collections ORDER BY name").fetchall()
                return self.ok({"collections":[{"name":r["name"]} for r in rows]})
            if parts == ["aliases"]:
                rows=self.store.db.execute("SELECT alias,collection_name FROM aliases ORDER BY alias").fetchall()
                return self.ok({"aliases":[{"alias_name":r["alias"],"collection_name":r["collection_name"]} for r in rows]})
            if len(parts)==2 and parts[0]=="collections":
                actual,row=self.store.collection(parts[1])
                if not row: return self.error(404,"collection not found")
                count=self.store.db.execute("SELECT COUNT(*) FROM points WHERE collection_name=?",(actual,)).fetchone()[0]
                return self.ok({"status":"green","optimizer_status":"ok","vectors_count":count,"points_count":count,"config":{"params":{"vectors":{"size":row["size"],"distance":row["distance"]}}},"aliases":[]})
            if len(parts)==4 and parts[0]=="collections" and parts[2]=="points":
                actual,row=self.store.collection(parts[1]); point=self.store.db.execute("SELECT * FROM points WHERE collection_name=? AND id=?",(actual,parts[3])).fetchone() if row else None
                if not point:return self.error(404,"point not found")
                return self.ok({"id":parts[3],"payload":json.loads(point["payload_json"]),"vector":unpack_vector(point["vector"],row["size"])})
            return self.error(404,"not found")
        except Exception as e: self.error(400,e)
    def do_PUT(self): self.mutate()
    def do_POST(self): self.mutate()
    def do_DELETE(self):
        try:
            parts=self.route()
            if len(parts)==2 and parts[0]=="collections":
                name=parts[1]
                with self.store.lock:
                    self.store.db.execute("BEGIN IMMEDIATE")
                    try:
                        actual=self.store.resolve(name)
                        self.store.db.execute("DELETE FROM aliases WHERE alias=? OR collection_name=?",(name,actual))
                        cur=self.store.db.execute("DELETE FROM collections WHERE name=?",(actual,))
                        self.store.db.execute("COMMIT")
                    except Exception:self.store.db.execute("ROLLBACK");raise
                return self.ok(bool(cur.rowcount))
            return self.error(404,"not found")
        except Exception as e:self.error(400,e)
    def mutate(self):
        try:
            parts=self.route(); body=self.read_json()
            if parts==["collections","aliases"]:
                actions=body.get("actions") or []
                with self.store.lock:
                    self.store.db.execute("BEGIN IMMEDIATE")
                    try:
                        for action in actions:
                            if "create_alias" in action:
                                a=action["create_alias"]; alias=str(a.get("alias_name") or a.get("alias") or ""); coll=str(a.get("collection_name") or "")
                                if not self.store.db.execute("SELECT 1 FROM collections WHERE name=?",(coll,)).fetchone(): raise ValueError(f"unknown collection: {coll}")
                                existing=self.store.db.execute("SELECT collection_name FROM aliases WHERE alias=?",(alias,)).fetchone()
                                if existing and existing[0]!=coll: raise ValueError(f"alias exists: {alias}")
                                self.store.db.execute("INSERT OR REPLACE INTO aliases VALUES(?,?)",(alias,coll))
                            elif "delete_alias" in action:
                                a=action["delete_alias"]; self.store.db.execute("DELETE FROM aliases WHERE alias=?",(str(a.get("alias_name") or a.get("alias") or ""),))
                            elif "rename_alias" in action:
                                a=action["rename_alias"]; old=str(a.get("old_alias_name") or ""); new=str(a.get("new_alias_name") or "")
                                row=self.store.db.execute("SELECT collection_name FROM aliases WHERE alias=?",(old,)).fetchone()
                                if not row: raise ValueError(f"unknown alias: {old}")
                                if self.store.db.execute("SELECT 1 FROM aliases WHERE alias=?",(new,)).fetchone(): raise ValueError(f"alias exists: {new}")
                                self.store.db.execute("DELETE FROM aliases WHERE alias=?",(old,)); self.store.db.execute("INSERT INTO aliases VALUES(?,?)",(new,row[0]))
                            else: raise ValueError("unsupported alias action")
                        self.store.db.execute("COMMIT")
                    except Exception:self.store.db.execute("ROLLBACK");raise
                return self.ok(True)
            if len(parts)==2 and parts[0]=="collections":
                vectors=body.get("vectors") or ((body.get("config") or {}).get("params") or {}).get("vectors") or {}
                if "default" in vectors: vectors=vectors["default"]
                size=int(vectors.get("size") or 0); distance=str(vectors.get("distance") or "Cosine")
                if size<=0: raise ValueError("vectors.size is required")
                name=parts[1]
                with self.store.lock:
                    self.store.db.execute("BEGIN IMMEDIATE")
                    try:
                        if self.store.db.execute("SELECT 1 FROM aliases WHERE alias=?",(name,)).fetchone():
                            raise ValueError(f"collection name is already an alias: {name}")
                        existing=self.store.db.execute("SELECT size,distance FROM collections WHERE name=?",(name,)).fetchone()
                        if existing:
                            if int(existing["size"]) != size or str(existing["distance"]).lower() != distance.lower():
                                raise ValueError(
                                    f"collection contract mismatch: expected size={existing['size']} distance={existing['distance']}, "
                                    f"requested size={size} distance={distance}"
                                )
                        else:
                            self.store.db.execute("INSERT INTO collections VALUES(?,?,?,?)",(name,size,distance,int(time.time())))
                        self.store.db.execute("COMMIT")
                    except Exception:
                        self.store.db.execute("ROLLBACK")
                        raise
                return self.ok(True)
            if len(parts)>=3 and parts[0]=="collections" and parts[2]=="points":
                actual,row=self.store.collection(parts[1])
                if not row:return self.error(404,"collection not found")
                size=int(row["size"])
                if len(parts)==3:
                    points=body.get("points") or []
                    with self.store.lock:
                        self.store.db.execute("BEGIN IMMEDIATE")
                        try:
                            for p in points:
                                pid=str(p.get("id") if p.get("id") is not None else uuid.uuid4())
                                vector=p.get("vector")
                                if isinstance(vector,dict): vector=vector.get("default") or next(iter(vector.values()))
                                blob=pack_vector(vector,size); payload=p.get("payload") if isinstance(p.get("payload"),dict) else {}
                                self.store.db.execute("INSERT INTO points VALUES(?,?,?,?,?) ON CONFLICT(collection_name,id) DO UPDATE SET vector=excluded.vector,payload_json=excluded.payload_json,updated_at=excluded.updated_at",(actual,pid,blob,stable_json(payload),int(time.time()*1000)))
                            self.store.db.execute("COMMIT")
                        except Exception:self.store.db.execute("ROLLBACK");raise
                    return self.ok({"operation_id":1,"status":"completed"})
                op=parts[3]
                if op=="delete":
                    ids=body.get("points") or body.get("ids") or []
                    filt=body.get("filter")
                    with self.store.lock:
                        self.store.db.execute("BEGIN IMMEDIATE")
                        try:
                            if ids:
                                self.store.db.executemany("DELETE FROM points WHERE collection_name=? AND id=?",[(actual,str(i)) for i in ids])
                            elif filt:
                                rows=self.store.db.execute("SELECT id,payload_json FROM points WHERE collection_name=?",(actual,)).fetchall()
                                self.store.db.executemany("DELETE FROM points WHERE collection_name=? AND id=?",[(actual,r["id"]) for r in rows if match_filter(json.loads(r["payload_json"]),r["id"],filt)])
                            self.store.db.execute("COMMIT")
                        except Exception:self.store.db.execute("ROLLBACK");raise
                    return self.ok({"operation_id":1,"status":"completed"})
                if op in {"query","search"}:
                    vector=body.get("query") if op=="query" else body.get("vector")
                    if isinstance(vector,dict): vector=vector.get("nearest") or vector.get("default") or next((v for v in vector.values() if isinstance(v,list)),None)
                    query=unpack_vector(pack_vector(vector,size),size)
                    limit=max(1,min(int(body.get("limit") or 10),10000)); threshold=body.get("score_threshold")
                    candidates=[]
                    for r in self.store.db.execute("SELECT id,vector,payload_json FROM points WHERE collection_name=?",(actual,)):
                        payload=json.loads(r["payload_json"])
                        if not match_filter(payload,r["id"],body.get("filter")):continue
                        score=cosine(query,unpack_vector(r["vector"],size))
                        if threshold is not None and score<float(threshold):continue
                        item={"id":r["id"],"score":score}
                        selected=select_payload(payload,body.get("with_payload",False))
                        if selected is not None:item["payload"]=selected
                        if body.get("with_vector"):item["vector"]=unpack_vector(r["vector"],size)
                        candidates.append(item)
                    candidates.sort(key=lambda x:(-x["score"],str(x["id"])))
                    result=candidates[:limit]
                    return self.ok({"points":result} if op=="query" else result)
                if op=="scroll":
                    limit=max(1,min(int(body.get("limit") or 10),10000)); offset=body.get("offset")
                    rows=self.store.db.execute("SELECT id,vector,payload_json FROM points WHERE collection_name=? ORDER BY id",(actual,)).fetchall()
                    result=[]
                    for r in rows:
                        if offset is not None and str(r["id"])<=str(offset):continue
                        payload=json.loads(r["payload_json"])
                        if not match_filter(payload,r["id"],body.get("filter")):continue
                        item={"id":r["id"]}; selected=select_payload(payload,body.get("with_payload",True))
                        if selected is not None:item["payload"]=selected
                        if body.get("with_vector"):item["vector"]=unpack_vector(r["vector"],size)
                        result.append(item)
                        if len(result)>=limit:break
                    next_offset=result[-1]["id"] if len(result)==limit else None
                    return self.ok({"points":result,"next_page_offset":next_offset})
                if op=="payload":
                    payload=body.get("payload") or {}; ids=body.get("points") or []
                    for pid in ids:
                        r=self.store.db.execute("SELECT payload_json FROM points WHERE collection_name=? AND id=?",(actual,str(pid))).fetchone()
                        if r:
                            merged=json.loads(r[0]); merged.update(payload); self.store.db.execute("UPDATE points SET payload_json=?,updated_at=? WHERE collection_name=? AND id=?",(stable_json(merged),int(time.time()*1000),actual,str(pid)))
                    return self.ok({"operation_id":1,"status":"completed"})
            return self.error(404,"not found")
        except ValueError as e:self.error(400,e)
        except sqlite3.IntegrityError as e:self.error(409,e)
        except Exception as e:self.error(500,e)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--host",default=os.environ.get("QDRANT_STUB_HOST","127.0.0.1")); ap.add_argument("--port",type=int,default=int(os.environ.get("QDRANT_STUB_PORT","6333"))); ap.add_argument("--db",default=os.environ.get("QDRANT_STUB_DB") or str(Path(os.environ.get("HERMES_HOME") or Path.home()/".hermes")/"knowledge_db"/"qdrant_stub.sqlite3")); args=ap.parse_args()
    store=Store(args.db); store.bootstrap(); server=ThreadingHTTPServer((args.host,args.port),Handler); server.store=store
    print(json.dumps({"ok":True,"version":VERSION,"listen":f"http://{args.host}:{args.port}","db":args.db,"vector_size":DEFAULT_DIM}),flush=True)
    try: server.serve_forever()
    except KeyboardInterrupt: pass
    finally: server.server_close(); store.db.close()
if __name__=="__main__":main()
