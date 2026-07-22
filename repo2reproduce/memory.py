"""Persistent, cross-run memory for Repo2Reproduce.

Unlike CLAUDE.md (static instructions you write) or session ``resume`` (continue ONE
prior run), this is an *accumulating* knowledge base: agents append durable learnings
during a run, and every future run recalls them at startup.

Storage: a JSONL file at ``R2R_MEMORY_PATH`` (default ``~/.r2r/memory.jsonl``), living
OUTSIDE any per-run ``runs/`` dir so it survives across runs and repos. Each line is one
learning; we de-duplicate on normalized text and support tag/scope filtering.

A learning looks like:
    {"text": "...", "tags": ["odl","json"], "scope": "global|<repo_url>",
     "source_run": "runs/kvzip", "ts": 1719800000}
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any, Optional


def memory_path() -> Path:
    p = os.environ.get("R2R_MEMORY_PATH")
    if p:
        return Path(p)
    return Path.home() / ".r2r" / "memory.jsonl"


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()


def load_memories() -> list[dict[str, Any]]:
    """Return all stored learnings (newest last). Tolerates a missing/corrupt file."""
    path = memory_path()
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def add_memory(
    text: str,
    tags: Optional[list[str]] = None,
    scope: str = "global",
    source_run: Optional[str] = None,
) -> dict[str, Any]:
    """Append a learning if it's not a near-duplicate of an existing one.

    Returns ``{"added": bool, "reason": str, "count": int}``.
    """
    text = text.strip()
    if len(text) < 8:
        return {"added": False, "reason": "too short to be a useful learning", "count": 0}

    existing = load_memories()
    ntext = _normalize(text)
    for m in existing:
        if _normalize(m.get("text", "")) == ntext:
            return {"added": False, "reason": "duplicate", "count": len(existing)}

    entry = {
        "text": text,
        "tags": [t.strip() for t in (tags or []) if t.strip()],
        "scope": scope or "global",
        "source_run": source_run,
        "ts": int(time.time()),
    }
    path = memory_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return {"added": True, "reason": "stored", "count": len(existing) + 1}


def recall(
    query: Optional[str] = None,
    tags: Optional[list[str]] = None,
    scope: Optional[str] = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return relevant learnings, filtered by optional query/tags/scope.

    ``scope`` matches entries whose scope equals it OR equals "global" (global learnings
    always apply). ``query`` is a whitespace-insensitive substring match on the text.
    """
    mems = load_memories()
    nq = _normalize(query) if query else None
    tagset = {t.strip().lower() for t in (tags or []) if t.strip()}

    def keep(m: dict[str, Any]) -> bool:
        if scope and m.get("scope") not in (scope, "global"):
            return False
        if tagset and not (tagset & {t.lower() for t in m.get("tags", [])}):
            return False
        if nq and nq not in _normalize(m.get("text", "")):
            return False
        return True

    hits = [m for m in mems if keep(m)]
    return hits[-limit:]


def format_for_prompt(scope: Optional[str] = None, limit: int = 40) -> str:
    """Render recalled learnings as a compact bullet list for a system-prompt block."""
    hits = recall(scope=scope, limit=limit)
    if not hits:
        return ""
    lines = ["## Learned memory (from previous runs — apply when relevant)"]
    for m in hits:
        tags = f" [{', '.join(m['tags'])}]" if m.get("tags") else ""
        lines.append(f"- {m['text']}{tags}")
    return "\n".join(lines)
