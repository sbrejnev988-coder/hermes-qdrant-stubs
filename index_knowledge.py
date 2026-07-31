#!/usr/bin/env python3
"""
index_knowledge.py — Индексатор для knowledge base.
Сканирует: скиллы, плагины, конфиги, память, сессии.
Использует local_vector_store (прямые вызовы, без HTTP-стабов).

v2.0 — 2026-06-27: migrated from HTTP stubs to local_vector_store (stdlib-only)
"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.expanduser("~/.hermes/knowledge_db"))
from local_vector_store import get_store

COLLECTION = "evey-knowledge"
HERMES_HOME = os.path.expanduser("~/.hermes")
store = get_store()

def index_skills():
    points = []
    skills_dir = os.path.join(HERMES_HOME, "skills")
    if not os.path.isdir(skills_dir): return points
    for dname in sorted(os.listdir(skills_dir)):
        dpath = os.path.join(skills_dir, dname)
        skill_file = os.path.join(dpath, "SKILL.md")
        if not os.path.isdir(dpath) or not os.path.exists(skill_file): continue
        try:
            with open(skill_file) as f: content = f.read()
        except Exception: continue
        title = dname.replace("-", " ").title(); desc = ""
        for line in content.split("\n")[:30]:
            if line.startswith("# ") and not title: title = line[2:].strip()
            if line.startswith("description:") or line.lower().startswith("description:"):
                desc = line.split(":", 1)[-1].strip().strip('"')
        search_text = f"{title}: {desc}\n{content[:2000]}"
        try:
            vec = store.embed(search_text)
            points.append({"id": len(points), "vector": vec, "payload": {"source": f"skills/{dname}", "type": "skill", "title": title, "description": desc or title, "content": content[:3000]}})
            print(f"  ✓ skill: {title}")
        except Exception as e: print(f"  ✗ skill {dname}: {e}")
    return points

def index_plugins():
    points = []
    plugins_dir = os.path.join(HERMES_HOME, "plugins")
    if not os.path.isdir(plugins_dir): return points
    for pname in os.listdir(plugins_dir):
        pdir = os.path.join(plugins_dir, pname)
        if not os.path.isdir(pdir): continue
        init_file = os.path.join(pdir, "__init__.py"); readme_file = os.path.join(pdir, "README.md")
        content = ""; desc = pname
        if os.path.exists(readme_file):
            try:
                with open(readme_file) as f: content = f.read()
                for line in content.split("\n"):
                    if line.startswith("# "): desc = line[2:].strip(); break
            except Exception: pass
        if not content and os.path.exists(init_file):
            try:
                with open(init_file) as f: content = f.read()
                m = re.search(r'"""(.*?)"""', content, re.DOTALL)
                if m: desc = m.group(1).strip().split("\n")[0]
            except Exception: pass
        if content:
            try:
                search_text = f"{desc}\n{content[:2000]}"
                vec = store.embed(search_text)
                points.append({"id": len(points) + 10000, "vector": vec, "payload": {"source": f"plugins/{pname}", "type": "plugin", "title": pname, "description": desc, "content": content[:3000]}})
                print(f"  ✓ plugin: {pname}")
            except Exception as e: print(f"  ✗ plugin {pname}: {e}")
    return points

def index_memory():
    points = []
    for ws_dir in ["/root/.openclaw/workspace", "/root/.openclaw/workspace-trading"]:
        if not os.path.isdir(ws_dir): continue
        for fname in ["SOUL.md", "USER.md", "AGENTS.md", "MEMORY.md", "BOOTSTRAP.md", "HEARTBEAT.md"]:
            fpath = os.path.join(ws_dir, fname)
            if os.path.exists(fpath):
                try:
                    with open(fpath) as f: content = f.read()
                    vec = store.embed(content[:3000])
                    points.append({"id": len(points) + 20000, "vector": vec, "payload": {"source": f"workspace/{fname}", "type": "memory", "title": fname.replace('.md', ''), "description": f"{fname.replace('.md', '')} ({os.path.basename(ws_dir)})", "content": content[:3000]}})
                    print(f"  ✓ memory: {fname} ({os.path.basename(ws_dir)})")
                except Exception as e: print(f"  ✗ memory {fpath}: {e}")
    for root_file in ["/root/.openclaw/workspace/TOOLS.md", "/root/.hermes/AGENTS.md"]:
        if os.path.exists(root_file):
            try:
                with open(root_file) as f: content = f.read()
                fname = os.path.basename(root_file)
                vec = store.embed(content[:3000])
                points.append({"id": len(points) + 20000, "vector": vec, "payload": {"source": f"docs/{fname}", "type": "docs", "title": fname.replace(".md", ""), "description": fname.replace(".md", ""), "content": content[:3000]}})
                print(f"  ✓ docs: {fname}")
            except Exception as e: print(f"  ✗ docs {root_file}: {e}")
    return points

def index_configs():
    points = []
    config_file = os.path.join(HERMES_HOME, "config.yaml")
    if os.path.exists(config_file):
        try:
            with open(config_file) as f: content = f.read()
            sections = re.split(r'\n(?=\w+:)', content)
            for i, section in enumerate(sections[:20]):
                if len(section.strip()) < 50: continue
                try:
                    vec = store.embed(section[:3000])
                    section_name = section.split("\n")[0].strip().rstrip(":")
                    points.append({"id": len(points) + 30000 + i, "vector": vec, "payload": {"source": "config.yaml", "type": "config", "title": f"Config: {section_name}", "description": f"Config section: {section_name}", "content": section[:3000]}})
                except Exception: pass
            print(f"  ✓ config: {len(points)} sections")
        except Exception as e: print(f"  ✗ config: {e}")
    return points

def main():
    print("=== Индексация Knowledge Base v2.0 (local_vector_store) ===\n")
    health = store.health()
    print(f"Store: {health.get('total_vectors', 0)} vectors, {len(health.get('collections', []))} collections, db={health.get('db_size', 0)}b\n")
    all_points = []
    print("📚 Скиллы:"); all_points.extend(index_skills())
    print("\n🔌 Плагины:"); all_points.extend(index_plugins())
    print("\n🧠 Память:"); all_points.extend(index_memory())
    print("\n⚙️ Конфиги:"); all_points.extend(index_configs())
    total = len(all_points)
    if total == 0: print("\n⚠️ Ничего не проиндексировано"); return
    result = store.replace_collection(COLLECTION, all_points)
    print(f"\n✅ Atomic replace: {result.get('points_replaced', 0)} vectors")
    info = store.get_collection_info(COLLECTION)
    print(f"Collection {COLLECTION}: {info.get('points_count', 0)} points")

if __name__ == "__main__": main()
