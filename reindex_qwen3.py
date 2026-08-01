#!/usr/bin/env python3
"""Compatibility guard for the obsolete standalone reindexer.

The old script uses a fixed v6 collection name and a different canonical document
format from current Memory Wiki. Running it can create a parallel incompatible
index. Use the plugin's memory_wiki_reindex tool, which owns the manifest,
checkpoint, reconciliation and alias switch.
"""
import os
import sys

MESSAGE = """Standalone reindex_qwen3.py is disabled by the r9 compatibility guard.
Use Hermes tool:
  memory_wiki_reindex({\"force\": false})
The legacy script used a stale collection name and incompatible vector text.
The original file was backed up by install.sh as reindex_qwen3.py.pre-r9.
"""

if os.environ.get("MEMORY_WIKI_ALLOW_LEGACY_REINDEX", "0").lower() in {"1", "true", "yes"}:
    legacy = os.path.join(os.path.dirname(__file__), "reindex_qwen3.py.pre-r9")
    if not os.path.exists(legacy):
        raise SystemExit("Legacy reindex file not found: " + legacy)
    os.execv(sys.executable, [sys.executable, legacy, *sys.argv[1:]])

print(MESSAGE, file=sys.stderr)
raise SystemExit(2)
