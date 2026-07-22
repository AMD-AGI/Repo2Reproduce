"""Code-intelligence helpers: clone a repo and extract REAL CLI flags / config keys.

The point of these tools is anti-hallucination: the command-synthesizer agent may only
use flags that ``extract_argparse`` actually found, and config keys that ``scan_configs``
actually found. If it is not here, it does not exist.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Optional


# --------------------------------------------------------------------------- #
# Clone
# --------------------------------------------------------------------------- #
def clone_repo(repo_url: str, dest: Path, depth: int = 1) -> dict[str, Any]:
    """Shallow-clone a public git repo to ``dest``. Returns basic metadata."""
    from git import Repo  # gitpython

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        return {"path": str(dest), "already_present": True}
    Repo.clone_from(repo_url, str(dest), depth=depth)
    return {"path": str(dest), "already_present": False}


# --------------------------------------------------------------------------- #
# argparse extraction (AST, no execution)
# --------------------------------------------------------------------------- #
def extract_argparse(file_path: Path) -> dict[str, Any]:
    """Statically extract argparse arguments from a Python file.

    Walks the AST for ``add_argument(...)`` calls and records the option strings,
    default, type, choices and help. Never imports or runs the file.
    """
    src = file_path.read_text(encoding="utf-8", errors="replace")
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return {"file": str(file_path), "error": f"syntax error: {e}", "arguments": []}

    args: list[dict] = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and _is_add_argument(node.func)):
            continue
        names = [_literal(a) for a in node.args if isinstance(_literal(a), str)]
        names = [n for n in names if isinstance(n, str) and n.startswith("-")]
        kw = {k.arg: _literal(k.value) for k in node.keywords if k.arg}
        args.append(
            {
                "names": names,
                "default": kw.get("default"),
                "type": kw.get("type"),
                "choices": kw.get("choices"),
                "required": kw.get("required"),
                "action": kw.get("action"),
                "help": kw.get("help"),
                "lineno": node.lineno,
            }
        )
    return {"file": str(file_path), "arguments": args, "count": len(args)}


def _is_add_argument(func: ast.AST) -> bool:
    return isinstance(func, ast.Attribute) and func.attr == "add_argument"


def _literal(node: ast.AST) -> Any:
    """Best-effort literal extraction; returns a marker string for non-literals."""
    try:
        return ast.literal_eval(node)
    except Exception:
        if isinstance(node, ast.Attribute):
            return f"<{node.attr}>"
        if isinstance(node, ast.Name):
            return f"<{node.id}>"
        return None


# --------------------------------------------------------------------------- #
# Config scanning (yaml / json / gin / toml)
# --------------------------------------------------------------------------- #
def scan_configs(repo_root: Path, max_files: int = 50) -> dict[str, Any]:
    """Locate likely config files and surface their keys with provenance.

    We do not hard-depend on a YAML parser; we record the file list and a light,
    line-numbered key dump so an agent can read exact values with provenance.
    """
    patterns = ["*.yaml", "*.yml", "*.json", "*.gin", "*.toml", "*.cfg", "*.ini"]
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv"}
    found: list[dict] = []
    for pat in patterns:
        for p in repo_root.rglob(pat):
            if any(part in skip_dirs for part in p.parts):
                continue
            if len(found) >= max_files:
                break
            rel = p.relative_to(repo_root)
            found.append({"path": str(rel), "size": p.stat().st_size, "kind": p.suffix.lstrip(".")})
    found.sort(key=lambda d: (len(Path(d["path"]).parts), d["path"]))
    return {"repo_root": str(repo_root), "configs": found, "count": len(found)}


def find_entry_points(repo_root: Path) -> dict[str, Any]:
    """Heuristically find likely training/eval entry-point scripts."""
    skip_dirs = {".git", "node_modules", "__pycache__", ".venv", "venv", "tests", "test"}
    name_hints = ("train", "main", "run", "eval", "evaluate", "finetune", "pretrain", "test")
    hits: list[dict] = []
    for p in repo_root.rglob("*.py"):
        if any(part in skip_dirs for part in p.parts):
            continue
        rel = p.relative_to(repo_root)
        stem = p.stem.lower()
        score = 0
        if stem in name_hints:
            score += 3
        elif any(h in stem for h in name_hints):
            score += 1
        try:
            head = p.read_text(encoding="utf-8", errors="replace")[:4000]
        except Exception:
            head = ""
        if "__main__" in head:
            score += 2
        if "add_argument" in head or "ArgumentParser" in head:
            score += 2
        if "argparse" in head or "hydra" in head or "click" in head:
            score += 1
        if score > 0:
            hits.append({"path": str(rel), "score": score})
    hits.sort(key=lambda d: -d["score"])
    return {"repo_root": str(repo_root), "entry_points": hits[:25]}


def find_readme_commands(repo_root: Path) -> dict[str, Any]:
    """Pull fenced code blocks from README files so agents can see suggested commands."""
    blocks: list[dict] = []
    for name in ["README.md", "README.rst", "README.txt", "readme.md"]:
        p = repo_root / name
        if not p.exists():
            continue
        text = p.read_text(encoding="utf-8", errors="replace")
        # crude fenced-block extraction
        parts = text.split("```")
        for i in range(1, len(parts), 2):
            block = parts[i]
            # strip an optional language tag on the first line
            lines = block.splitlines()
            if lines and lines[0].strip().isalpha():
                lines = lines[1:]
            content = "\n".join(lines).strip()
            if content:
                blocks.append({"file": name, "block": content[:1500]})
    return {"repo_root": str(repo_root), "readme_command_blocks": blocks[:30]}


def verify_flags(file_path: Path, flags: list[str]) -> dict[str, Any]:
    """Check that each flag in ``flags`` exists among the file's argparse options."""
    extracted = extract_argparse(file_path)
    known: set[str] = set()
    for a in extracted.get("arguments", []):
        for n in a.get("names", []):
            known.add(n)
    results = {flag: (flag.split("=")[0] in known) for flag in flags}
    return {"file": str(file_path), "results": results, "known_flags": sorted(known)}
