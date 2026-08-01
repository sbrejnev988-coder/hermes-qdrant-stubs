#!/usr/bin/env python3
"""Safe deterministic indexer for the auxiliary Hermes knowledge collection.

r9 changes:
- HERMES_HOME-aware paths; no /root hard-coding
- stable hash point IDs instead of order-dependent integer IDs
- secret redaction before both embedding and payload persistence
- vault/.env/credential files excluded
- bounded reads and explicit optional OpenClaw roots
- uses the strict local_vector_store vector contract
"""
from __future__ import annotations

import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable

HERMES_HOME = Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))).expanduser().resolve()
KNOWLEDGE_DIR = Path(os.environ.get("QDRANT_STUB_DIR", str(HERMES_HOME / "knowledge_db"))).expanduser().resolve()
if str(KNOWLEDGE_DIR) not in sys.path:
    sys.path.insert(0, str(KNOWLEDGE_DIR))

from local_vector_store import get_store  # noqa: E402

COLLECTION = os.environ.get("KNOWLEDGE_INDEX_COLLECTION", "evey-knowledge")
MAX_FILE_BYTES = max(4096, min(int(os.environ.get("KNOWLEDGE_INDEX_MAX_FILE_BYTES", str(2 * 1024 * 1024))), 16 * 1024 * 1024))
MAX_EMBED_CHARS = max(512, min(int(os.environ.get("KNOWLEDGE_INDEX_MAX_EMBED_CHARS", "6000")), 24000))
MAX_PAYLOAD_CHARS = max(512, min(int(os.environ.get("KNOWLEDGE_INDEX_MAX_PAYLOAD_CHARS", "8000")), 32000))
INCLUDE_CONFIG = os.environ.get("KNOWLEDGE_INDEX_INCLUDE_CONFIG", "1").lower() not in {"0", "false", "no", "off"}
store = get_store()

PRIVATE_KEY_RE = re.compile(
    r"-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----.*?-----END (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----",
    re.IGNORECASE | re.DOTALL,
)
URL_CREDENTIAL_RE = re.compile(r"(?i)\b([a-z][a-z0-9+.-]*://)([^\s/@:]+):([^\s/@]+)@")
SENSITIVE_LINE_RE = re.compile(
    r"(?im)^(?P<indent>\s*)(?P<key>[\w.-]*(?:pass(?:word|wd)?|token|api[_-]?key|client[_-]?secret|secret|access[_-]?key|private[_-]?key|authorization|cookie|session|totp|seed)[\w.-]*)"
    r"(?P<sep>\s*[:=]\s*)(?P<value>[^\r\n#]+)"
)
SENSITIVE_JSON_RE = re.compile(
    r"(?ix)([\"\'](?:password|passwd|token|api[_-]?key|client[_-]?secret|secret|access[_-]?key|private[_-]?key|authorization|cookie|session|totp|seed)[\"\']\s*:\s*)([\"\'])(.*?)(\2)"
)

TOKEN_RULES = [
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bgh[opsu]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{8,}\b"),
]
EXCLUDED_PARTS = {
    "vault", "secrets", "secret", ".env", ".git", "node_modules", "__pycache__",
    "credentials", "credential", "cookies", "sessions",
}
EXCLUDED_NAMES = {
    "secrets_registry.json", "credentials.json", "service-account.json", ".env",
    "id_rsa", "id_ed25519", "known_hosts",
}


def redact_secrets(text: str) -> str:
    value = PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", str(text or ""))
    value = URL_CREDENTIAL_RE.sub(r"\1[REDACTED]:[REDACTED]@", value)
    value = SENSITIVE_LINE_RE.sub(lambda m: f"{m.group('indent')}{m.group('key')}{m.group('sep')}[REDACTED]", value)
    value = SENSITIVE_JSON_RE.sub(lambda m: f'{m.group(1)}"[REDACTED]"', value)
    for rule in TOKEN_RULES:
        value = rule.sub("[REDACTED_TOKEN]", value)
    return value


def safe_path(path: Path, *, allow_config: bool = False) -> bool:
    try:
        resolved = path.expanduser().resolve(strict=True)
    except (OSError, RuntimeError):
        return False
    if not resolved.is_file() or resolved.stat().st_size > MAX_FILE_BYTES:
        return False
    lower_parts = {part.lower() for part in resolved.parts}
    if resolved.name.lower() in EXCLUDED_NAMES:
        return False
    if lower_parts & EXCLUDED_PARTS:
        return False
    if resolved.suffix.lower() in {".key", ".pem", ".p12", ".pfx", ".kdbx"}:
        return False
    if resolved.name == "config.yaml" and allow_config:
        return True
    return True


def read_redacted(path: Path, *, allow_config: bool = False) -> str:
    if not safe_path(path, allow_config=allow_config):
        return ""
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return redact_secrets(raw)


def stable_id(source: str) -> str:
    return "k_" + hashlib.sha256(source.encode("utf-8", "ignore")).hexdigest()[:24]


def point(source: str, kind: str, title: str, description: str, content: str) -> dict:
    clean = redact_secrets(content)[:MAX_PAYLOAD_CHARS]
    search_text = redact_secrets(f"{title}: {description}\n{clean}")[:MAX_EMBED_CHARS]
    vector = store.embed(search_text)
    return {
        "id": stable_id(source),
        "vector": vector,
        "payload": {
            "source": source,
            "type": kind,
            "title": title,
            "description": redact_secrets(description)[:1000],
            "content": clean,
            "security": "secret_redacted_r9",
        },
    }


def index_skills() -> list[dict]:
    result: list[dict] = []
    root = HERMES_HOME / "skills"
    if not root.is_dir():
        return result
    for directory in sorted(root.iterdir(), key=lambda p: p.name):
        path = directory / "SKILL.md"
        content = read_redacted(path)
        if not content:
            continue
        title = directory.name.replace("-", " ").title()
        description = title
        for line in content.splitlines()[:40]:
            if line.startswith("# "):
                title = line[2:].strip() or title
            if line.lower().startswith("description:"):
                description = line.split(":", 1)[1].strip().strip('"') or description
        result.append(point(f"skills/{directory.name}", "skill", title, description, content))
    return result


def index_plugins() -> list[dict]:
    result: list[dict] = []
    root = HERMES_HOME / "plugins"
    if not root.is_dir():
        return result
    for directory in sorted(root.iterdir(), key=lambda p: p.name):
        if not directory.is_dir():
            continue
        readme = directory / "README.md"
        init = directory / "__init__.py"
        content = read_redacted(readme) or read_redacted(init)
        if not content:
            continue
        title = directory.name
        description = title
        for line in content.splitlines()[:60]:
            if line.startswith("# "):
                description = line[2:].strip() or description
                break
        result.append(point(f"plugins/{directory.name}", "plugin", title, description, content))
    return result


def configured_openclaw_roots() -> Iterable[Path]:
    configured = os.environ.get("KNOWLEDGE_INDEX_OPENCLAW_ROOTS", "").strip()
    if configured:
        for item in configured.split(os.pathsep):
            if item.strip():
                yield Path(item.strip()).expanduser()
        return
    default = Path.home() / ".openclaw" / "workspace"
    if default.is_dir():
        yield default


def index_memory() -> list[dict]:
    result: list[dict] = []
    names = ("SOUL.md", "USER.md", "AGENTS.md", "MEMORY.md", "BOOTSTRAP.md", "HEARTBEAT.md", "TOOLS.md")
    roots = list(configured_openclaw_roots())
    for root in roots:
        for name in names:
            path = root / name
            content = read_redacted(path)
            if content:
                source = f"workspace/{root.name}/{name}"
                result.append(point(source, "memory", name.removesuffix(".md"), f"{name} ({root})", content))
    hermes_agents = HERMES_HOME / "AGENTS.md"
    content = read_redacted(hermes_agents)
    if content:
        result.append(point("docs/AGENTS.md", "docs", "AGENTS", "Hermes agent instructions", content))
    return result


def index_configs() -> list[dict]:
    if not INCLUDE_CONFIG:
        return []
    path = HERMES_HOME / "config.yaml"
    content = read_redacted(path, allow_config=True)
    if not content:
        return []
    result: list[dict] = []
    sections = re.split(r"\n(?=[A-Za-z_][\w.-]*\s*:)", content)
    for index, section in enumerate(sections[:100]):
        section = section.strip()
        if len(section) < 30:
            continue
        section_name = section.splitlines()[0].strip().rstrip(":")[:120]
        source = f"config.yaml#{index}:{section_name}"
        result.append(point(source, "config", f"Config: {section_name}", f"Redacted config section: {section_name}", section))
    return result


def main() -> int:
    print("=== Safe Knowledge Base index r9 ===")
    groups = [
        ("skills", index_skills()),
        ("plugins", index_plugins()),
        ("memory", index_memory()),
        ("configs", index_configs()),
    ]
    all_points = [entry for _name, entries in groups for entry in entries]
    for name, entries in groups:
        print(f"{name}: {len(entries)}")
    if not all_points:
        print("Nothing indexed")
        return 0
    result = store.replace_collection(COLLECTION, all_points)
    print(f"Atomic replace: {result.get('points_replaced', 0)} vectors")
    print(store.get_collection_info(COLLECTION))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
