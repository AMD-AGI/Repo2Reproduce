#!/usr/bin/env python3
"""Claude Code PreToolUse hook enforcing Docker-only mutation/execution."""

from __future__ import annotations

import datetime as dt
import json
import os
import pathlib
import re
import sys

SAFE_HOST_COMMAND = re.compile(
    r"^(?:pwd|whoami|hostname|nproc|"
    r"uname(?:\s+-[a-zA-Z]+)?|"
    r"ls(?:\s+[-a-zA-Z0-9_./]+)*|"
    r"df(?:\s+[-a-zA-Z0-9_./]+)*|"
    r"du(?:\s+[-a-zA-Z0-9_./]+)*|"
    r"free(?:\s+[-a-zA-Z0-9_./]+)*|"
    r"stat(?:\s+[-a-zA-Z0-9_./]+)+|"
    r"rocm-smi(?:\s+[-a-zA-Z0-9_=.,/]+)*)$"
)
# Installed only for --rocm-porting runs. One command, no shell syntax.
PORTING_COMMAND = re.compile(r"^(?:port-scan|port-hipify)(?:\s+[A-Za-z0-9_./:@+-]+)*$")


def decision(value: str, reason: str) -> dict:
    return {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": value,
            "permissionDecisionReason": reason,
        }
    }


def main() -> int:
    payload = json.load(sys.stdin)
    command = str(payload.get("tool_input", {}).get("command", "")).strip()
    run_dir = pathlib.Path(os.environ["RUN_DIR"])
    log_path = run_dir / "logs" / "bash_tool_calls.jsonl"

    if command == "docker" or command.startswith("docker "):
        result = decision(
            "allow",
            "Docker command allowed; the benchmark Docker wrapper enforces GPU and mounts.",
        )
    elif SAFE_HOST_COMMAND.fullmatch(command):
        result = decision("allow", "Read-only host inspection allowed.")
    elif PORTING_COMMAND.fullmatch(command):
        result = decision(
            "allow",
            "CUDA-to-ROCm porting command allowed. port-scan only reads; port-hipify writes a "
            "HIPIFY-translated copy inside the workspace and never edits the clone.",
        )
    else:
        result = decision(
            "deny",
            "Host-side execution is disabled. Use file tools for workspace inspection and "
            "run installation, downloads, cloning, Python, smoke tests, and evaluation "
            "through a command beginning with `docker`.",
        )

    event = {
        "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
        "command": command,
        "decision": result["hookSpecificOutput"]["permissionDecision"],
        "reason": result["hookSpecificOutput"]["permissionDecisionReason"],
    }
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event) + "\n")

    print(json.dumps(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
