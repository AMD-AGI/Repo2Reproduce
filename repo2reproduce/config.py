"""Central configuration — a SINGLE source of truth: environment variables.

Every key, secret, and setting comes from the environment, which you populate from one
place: a ``.env`` file (loaded by ``python-dotenv`` at CLI startup). There is no YAML/JSON
config file and no separate key file — one file, one source.

Built-in *fallback defaults* (public Anthropic API, model aliases, run limits) live in
this module purely so the tool runs out-of-the-box; anything you set in ``.env`` overrides
them.

No vendor-specific value (gateway URL, custom header names, model ids) is hardcoded here;
those are supplied via ``.env`` (see ``.env.example``).

The LLM is reached through the Claude Agent SDK, which drives the Claude Code CLI. The CLI
(not our Python) makes the HTTP calls, so a provider is expressed as environment overrides
the CLI understands:

* ``ANTHROPIC_BASE_URL``        -> the provider's Anthropic-format endpoint (if any)
* ``ANTHROPIC_CUSTOM_HEADERS``  -> gateway auth headers (for ``custom_headers`` auth)
* ``ANTHROPIC_AUTH_TOKEN``      -> placeholder token so the CLI does not fall back to a
                                   saved login when a gateway uses header auth
* ``ENABLE_TOOL_SEARCH``        -> keep our ``mcp__r2r__*`` tools working behind a
                                   non-Anthropic base URL
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# --------------------------------------------------------------------------- #
# Built-in fallback defaults (vendor-neutral; all overridable via .env).
# --------------------------------------------------------------------------- #
_DEFAULT_PROVIDER = "anthropic"
_DEFAULT_API_KEY_ENV = "ANTHROPIC_API_KEY"
_DEFAULT_REASONING_MODEL = "opus"
_DEFAULT_MAPPING_MODEL = "sonnet"
_DEFAULT_MAX_BUDGET_USD = 5.0
_DEFAULT_MAX_TURNS = 200
_DEFAULT_AUTH_TOKEN = "gateway"


# --------------------------------------------------------------------------- #
# Small env helpers
# --------------------------------------------------------------------------- #
def _env(*names: str) -> Optional[str]:
    """Return the first non-empty environment variable among ``names``."""
    for n in names:
        v = os.environ.get(n)
        if v:
            return v
    return None


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in {"1", "true", "yes", "on"}


def _parse_headers(spec: str) -> dict[str, str]:
    """Parse ``Name: value`` / ``Name=value`` pairs separated by newlines or ``;``."""
    headers: dict[str, str] = {}
    parts = [p for chunk in spec.split("\n") for p in chunk.split(";")]
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if ":" in part:
            name, _, value = part.partition(":")
        elif "=" in part:
            name, _, value = part.partition("=")
        else:
            continue
        headers[name.strip()] = value.strip()
    return headers


# --------------------------------------------------------------------------- #
# Provider resolution (entirely from environment variables)
# --------------------------------------------------------------------------- #
@dataclass
class Provider:
    name: str
    auth_mode: str  # "api_key" | "custom_headers"
    base_url: Optional[str]
    api_key_env: str
    key: Optional[str]  # resolved secret (custom_headers mode)
    headers: dict[str, str] = field(default_factory=dict)
    auth_token: str = _DEFAULT_AUTH_TOKEN
    enable_tool_search: bool = True
    reasoning_model: str = _DEFAULT_REASONING_MODEL
    mapping_model: str = _DEFAULT_MAPPING_MODEL


def active_provider_name() -> str:
    """The selected provider label (just a name; behavior is driven by R2R_AUTH_MODE)."""
    return (os.environ.get("R2R_PROVIDER") or _DEFAULT_PROVIDER).strip()


def get_provider() -> Provider:
    """Assemble the active provider purely from environment variables."""
    name = active_provider_name()
    auth_mode = (os.environ.get("R2R_AUTH_MODE") or "api_key").strip()
    base_url = _env("R2R_BASE_URL")
    api_key_env = os.environ.get("R2R_API_KEY_ENV") or _DEFAULT_API_KEY_ENV

    hdr_spec = os.environ.get("R2R_AUTH_HEADERS")
    headers = _parse_headers(hdr_spec) if hdr_spec else {}

    user = _env("R2R_AUTH_USER", "USER") or "unknown"
    key = _env("R2R_AUTH_KEY")
    rendered_headers = {
        hname: hval.replace("{key}", key or "").replace("{user}", user)
        for hname, hval in headers.items()
    }

    reasoning = os.environ.get("R2R_MODEL_REASONING") or _DEFAULT_REASONING_MODEL
    mapping = os.environ.get("R2R_MODEL_MAPPING") or _DEFAULT_MAPPING_MODEL

    return Provider(
        name=name,
        auth_mode=auth_mode,
        base_url=base_url,
        api_key_env=api_key_env,
        key=key,
        headers=rendered_headers,
        auth_token=os.environ.get("R2R_AUTH_TOKEN") or _DEFAULT_AUTH_TOKEN,
        enable_tool_search=_env_bool("R2R_ENABLE_TOOL_SEARCH", True),
        reasoning_model=reasoning,
        mapping_model=mapping,
    )


def provider_env() -> dict[str, str]:
    """Environment overrides to pass to ``ClaudeAgentOptions.env`` for the CLI.

    * ``api_key`` mode: the CLI reads the key from its own env var (default
      ``ANTHROPIC_API_KEY``); we only inject a base URL if one is configured.
    * ``custom_headers`` mode: inject the gateway base URL + rendered auth headers.
    """
    p = get_provider()
    env: dict[str, str] = {}

    if p.auth_mode == "custom_headers":
        if not p.base_url:
            raise RuntimeError(
                f"Provider '{p.name}' uses custom_headers auth but R2R_BASE_URL is not set."
            )
        if not p.headers:
            raise RuntimeError(
                f"Provider '{p.name}' uses custom_headers auth but R2R_AUTH_HEADERS is not set."
            )
        if not p.key:
            raise RuntimeError(
                f"Provider '{p.name}' needs a gateway key but R2R_AUTH_KEY is not set."
            )
        env["ANTHROPIC_BASE_URL"] = p.base_url
        env["ANTHROPIC_CUSTOM_HEADERS"] = "\n".join(f"{k}: {v}" for k, v in p.headers.items())
        env["ANTHROPIC_AUTH_TOKEN"] = p.auth_token
        if p.enable_tool_search:
            env["ENABLE_TOOL_SEARCH"] = "true"
    else:  # api_key
        if p.base_url:
            env["ANTHROPIC_BASE_URL"] = p.base_url
            if p.enable_tool_search:
                env["ENABLE_TOOL_SEARCH"] = "true"

    return env


def reasoning_model() -> str:
    return get_provider().reasoning_model


def mapping_model() -> str:
    return get_provider().mapping_model


# --------------------------------------------------------------------------- #
# Run guardrails
# --------------------------------------------------------------------------- #
def max_budget_usd() -> float:
    env = os.environ.get("R2R_MAX_BUDGET_USD")
    if env:
        try:
            return float(env)
        except ValueError:
            pass
    return _DEFAULT_MAX_BUDGET_USD


def max_turns() -> int:
    env = os.environ.get("R2R_MAX_TURNS")
    if env:
        try:
            return int(env)
        except ValueError:
            pass
    return _DEFAULT_MAX_TURNS


def runtime_extra_path() -> Optional[str]:
    """Optional extra dir to prepend to PATH so the SDK finds the ``claude`` binary."""
    return os.environ.get("R2R_CLI_BIN_DIR")
