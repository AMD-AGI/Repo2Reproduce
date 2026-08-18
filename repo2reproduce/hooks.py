"""Hooks for safety, observability, and cost tracking.

* PreToolUse  -> block destructive Bash, and (in plan-only mode) block commands that would
  actually run training or do mass downloads on the host. Also log every tool call.
* SubagentStart / SubagentStop -> trace which phase is running.

Two tiers of blocking:
* ALWAYS blocked: destructive / host-mutating ops (rm -rf, git push, sudo, fork bomb,
  shutdown) — blocked in every mode, including --execute.
* PLAN-ONLY blocked: launching training/eval and shell mass-downloads. In --execute mode
  these are allowed, because real execution happens INSIDE the container via the
  mcp__r2r__docker_run tool (sandboxed), not on the host.
"""

from __future__ import annotations

import re
from typing import Any

from claude_agent_sdk import HookMatcher

# Destructive / host-mutating — blocked in EVERY mode.
_ALWAYS_BLOCK = [
    r"\brm\s+-rf\b",
    r"\bgit\s+push\b",
    r"\bgit\s+reset\s+--hard\b",
    r"\bsudo\b",
    r"\bmkfs\b",
    r":\(\)\s*\{",  # fork bomb
    r"\bshutdown\b|\breboot\b",
    r"\bdocker\s+(rmi|system\s+prune|volume\s+rm)\b",  # don't nuke images/volumes
]

# Allowed only in --execute mode (otherwise the system is plan-only).
_PLAN_ONLY_BLOCK = [
    r"\bpython[0-9.]*\s+.*\b(train|pretrain|finetune|main)\.py\b",
    r"\btorchrun\b|\baccelerate\s+launch\b|\bdeepspeed\b",
    r"\bwget\b|\bcurl\b\s+.*\b(http|https)://",
]


def _make_bash_guard(execute: bool):
    async def guard_bash(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict:
        if input_data.get("tool_name") != "Bash":
            return {}
        command = str(input_data.get("tool_input", {}).get("command", ""))
        patterns = list(_ALWAYS_BLOCK) + ([] if execute else list(_PLAN_ONLY_BLOCK))
        for pat in patterns:
            if re.search(pat, command):
                reason = f"Blocked by Repo2Reproduce safety policy (pattern: {pat}). " + (
                    "Destructive/host-mutating commands are never allowed."
                    if pat in _ALWAYS_BLOCK
                    else "This is plan-only mode; run experiments inside the container "
                    "via docker_run, or re-run with --execute."
                )
                return {
                    "hookSpecificOutput": {
                        "hookEventName": "PreToolUse",
                        "permissionDecision": "deny",
                        "permissionDecisionReason": reason,
                    }
                }
        return {}

    return guard_bash


def _make_logger(label: str):
    async def _log(input_data: dict[str, Any], tool_use_id: str | None, context: Any) -> dict:
        name = input_data.get("tool_name") or input_data.get("hook_event_name") or label
        agent = input_data.get("agent_type")
        prefix = f"[{agent}] " if agent else ""
        print(f"  · {prefix}{label}: {name}")
        return {}

    return _log


def build_hooks(execute: bool = False, amd: bool = False) -> dict:
    """Assemble safety and observability hooks only.

    Pipeline order/completion is owned by Python in ``orchestrator.py``; hooks are no
    longer used as an orchestration mechanism. ``amd`` remains in the signature for API
    compatibility with callers.
    """
    del amd
    return {
        "PreToolUse": [
            HookMatcher(matcher="Bash", hooks=[_make_bash_guard(execute)], timeout=30),
            HookMatcher(hooks=[_make_logger("pre-tool")]),
        ],
        "SubagentStart": [HookMatcher(hooks=[_make_logger("subagent-start")])],
        "SubagentStop": [HookMatcher(hooks=[_make_logger("subagent-stop")])],
    }
