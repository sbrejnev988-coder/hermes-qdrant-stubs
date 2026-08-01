#!/usr/bin/env python3
"""OpenAI-compatible deterministic hash embedding fallback.

This is NOT Qwen3-Embedding-8B. It is an offline emergency fallback whose
vector dimension is configurable and defaults to the user's 4096D contract.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import signal
import threading
import time
from collections import Counter
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn


def env_int(name: str, default: int, low: int, high: int) -> int:
    try:
        return max(low, min(int(os.environ.get(name, str(default))), high))
    except (TypeError, ValueError):
        return default


VECTOR_SIZE = env_int(
    "EMBED_STUB_VECTOR_SIZE",
    env_int("MEMORY_WIKI_EMBED_DIMENSIONS", 4096, 8, 16384),
    8,
    16384,
)
PORT = env_int("EMBED_STUB_PORT", 4000, 1, 65535)
MAX_TEXT_LEN = env_int("EMBED_STUB_MAX_TEXT_LEN", 12000, 256, 131072)
MAX_REQUEST_BYTES = env_int("EMBED_STUB_MAX_REQUEST_BYTES", 5_000_000, 1024, 64 * 1024 * 1024)
NGRAM_SIZES = (2, 3, 4)
ERROR_LOG = os.environ.get("EMBED_STUB_LOG", "/tmp/embed_stub.log")
QWEN_QUERY_INSTRUCTION = os.environ.get(
    "QWEN_QUERY_INSTRUCTION",
    os.environ.get(
        "MEMORY_WIKI_QUERY_INSTRUCTION",
        "Retrieve durable personal infrastructure facts, preferences, decisions "
        "and operational context relevant to the user's request.",
    ),
)
QWEN_DOCUMENT_PREFIX = os.environ.get(
    "QWEN_DOCUMENT_PREFIX",
    os.environ.get("MEMORY_WIKI_DOCUMENT_PREFIX", ""),
)


def log(message: str) -> None:
    try:
        with open(ERROR_LOG, "a", encoding="utf-8") as handle:
            handle.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} EMBED {message}\n")
    except OSError:
        pass


def memory_ok() -> bool:
    try:
        values = {}
        with open("/proc/meminfo", encoding="utf-8") as handle:
            for line in handle:
                key, _, tail = line.partition(":")
                if tail:
                    values[key.strip()] = int(tail.strip().split()[0])
        total = values.get("MemTotal", 0)
        available = values.get("MemAvailable", 0) or values.get("MemFree", 0)
        return total == 0 or available / total >= 0.10
    except Exception:
        return True


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower(), flags=re.UNICODE)


def ngrams(word: str, size: int):
    if len(word) < size:
        yield word
        return
    for index in range(len(word) - size + 1):
        yield word[index:index + size]


def bucket(value: str) -> int:
    digest = hashlib.blake2b(value.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % VECTOR_SIZE


def text_to_vector(text: str, *, instruction: str = "", task_type: str = "search_document") -> list[float]:
    if task_type == "search_query" and instruction:
        text = instruction + "\n\n" + text
    elif task_type == "search_document" and QWEN_DOCUMENT_PREFIX:
        text = QWEN_DOCUMENT_PREFIX + text

    counts: Counter[str] = Counter()
    for word in tokenize(text):
        for size in NGRAM_SIZES:
            counts.update(ngrams(word, size))
    if not counts:
        return [0.0] * VECTOR_SIZE

    vector = [0.0] * VECTOR_SIZE
    maximum = max(counts.values())
    for gram, count in counts.items():
        vector[bucket(gram)] += count / maximum
    norm = math.sqrt(sum(value * value for value in vector))
    if norm:
        vector = [value / norm for value in vector]
    return vector


class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    allow_reuse_address = True
    daemon_threads = True


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, _format, *_args):
        return

    def send_json(self, value, status=200):
        body = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)

    def read_json(self):
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError:
            raise ValueError("Invalid Content-Length")
        if length < 0 or length > MAX_REQUEST_BYTES:
            raise ValueError(f"Request body exceeds {MAX_REQUEST_BYTES} bytes")
        if not length:
            return {}
        value = json.loads(self.rfile.read(length))
        if not isinstance(value, dict):
            raise ValueError("JSON body must be an object")
        return value

    def do_GET(self):
        if self.path == "/health":
            self.send_json({
                "status": "ok",
                "version": "2.0-4096-contract",
                "backend": "hash-ngram-fallback",
                "vector_size": VECTOR_SIZE,
                "model": f"hash-ngram-{VECTOR_SIZE}",
                "warning": "offline fallback; not qwen/qwen3-embedding-8b",
            })
            return
        self.send_json({"error": "Not found"}, 404)

    def do_POST(self):
        if self.path not in ("/v1/embeddings", "/embeddings"):
            self.send_json({"error": "Not found"}, 404)
            return
        try:
            if not memory_ok():
                self.send_json({"error": "Server under memory pressure"}, 503)
                return
            body = self.read_json()
            raw_input = body.get("input", "")
            texts = raw_input if isinstance(raw_input, list) else [raw_input]
            texts = [str(item)[:MAX_TEXT_LEN] for item in texts]
            task_type = str(body.get("task_type", body.get("input_type", "search_document")))
            instruction = str(body.get("instruction") or "")
            if task_type == "search_query" and not instruction:
                instruction = QWEN_QUERY_INSTRUCTION
            embeddings = [
                text_to_vector(text, instruction=instruction, task_type=task_type)
                for text in texts
            ]
            token_estimate = sum(len(tokenize(text)) for text in texts)
            self.send_json({
                "object": "list",
                "data": [
                    {"object": "embedding", "index": index, "embedding": embedding}
                    for index, embedding in enumerate(embeddings)
                ],
                "model": body.get("model") or f"hash-ngram-{VECTOR_SIZE}",
                "usage": {"prompt_tokens": token_estimate, "total_tokens": token_estimate},
            })
        except ValueError as exc:
            log(f"BAD_REQUEST {exc}")
            self.send_json({"error": str(exc)}, 400)
        except Exception as exc:
            log(f"ERROR {type(exc).__name__}: {exc}")
            self.send_json({"error": "Internal server error"}, 500)


def main() -> int:
    server = ThreadingHTTPServer(("127.0.0.1", PORT), Handler)

    def stop(_signum, _frame):
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    log(f"START port={PORT} vector_size={VECTOR_SIZE}")
    try:
        server.serve_forever(poll_interval=0.25)
    finally:
        server.server_close()
        log("STOP")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
