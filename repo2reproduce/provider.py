"""Claude Code provider and runtime-path wiring."""

from __future__ import annotations

import os
from pathlib import Path

from . import config

provider_env = config.provider_env
reasoning_model = config.reasoning_model


def ensure_runtime_path() -> None:
    """Ensure the dir holding the ``claude`` binary is on PATH.

    Defaults to ``~/.local/bin`` (where the installer puts it); override with
    ``R2R_CLI_BIN_DIR`` if the binary lives elsewhere.
    """
    extra = config.runtime_extra_path() or str(Path.home() / ".local" / "bin")
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if extra not in parts:
        os.environ["PATH"] = f"{extra}{os.pathsep}{os.environ.get('PATH', '')}"
