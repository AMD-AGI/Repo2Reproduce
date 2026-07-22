"""Provider wiring (compatibility shim).

All configuration now lives in :mod:`repo2reproduce.config`, which resolves settings
generically from environment variables (via ``.env``) and/or a YAML/JSON config file.
"""

from __future__ import annotations

import os
from pathlib import Path

from . import config

# Re-export the provider helpers so existing imports keep working.
provider_env = config.provider_env
reasoning_model = config.reasoning_model
mapping_model = config.mapping_model


def ensure_runtime_path() -> None:
    """Ensure the dir holding the ``claude`` binary is on PATH.

    Defaults to ``~/.local/bin`` (where the installer puts it); override with
    ``R2R_CLI_BIN_DIR`` if the binary lives elsewhere.
    """
    extra = config.runtime_extra_path() or str(Path.home() / ".local" / "bin")
    parts = os.environ.get("PATH", "").split(os.pathsep)
    if extra not in parts:
        os.environ["PATH"] = f"{extra}{os.pathsep}{os.environ.get('PATH', '')}"


def using_amd() -> bool:
    """True when the active provider is named ``amd`` (kept for callers/tests)."""
    return config.active_provider_name().lower() == "amd"


def amd_user() -> str:
    return os.environ.get("R2R_AUTH_USER") or os.environ.get("USER") or "unknown"
