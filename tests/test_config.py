"""Tests for the env-only config layer (single source of truth: environment vars)."""

from __future__ import annotations

import importlib

import pytest


@pytest.fixture()
def cfg(monkeypatch):
    """A fresh config module with all provider/config env vars cleared."""
    from repo2reproduce import config

    for var in [
        "R2R_PROVIDER",
        "R2R_BASE_URL",
        "R2R_AUTH_MODE",
        "R2R_AUTH_HEADERS",
        "R2R_AUTH_KEY",
        "R2R_AUTH_USER",
        "R2R_AUTH_TOKEN",
        "R2R_API_KEY_ENV",
        "R2R_ENABLE_TOOL_SEARCH",
        "R2R_MODEL_REASONING",
        "R2R_MAX_BUDGET_USD",
        "R2R_MAX_TURNS",
        "ANTHROPIC_API_KEY",
    ]:
        monkeypatch.delenv(var, raising=False)
    return importlib.reload(config)


def test_default_provider_is_anthropic(cfg):
    assert cfg.active_provider_name() == "anthropic"
    # api_key mode with no base_url -> no CLI env overrides needed.
    assert cfg.provider_env() == {}
    assert cfg.reasoning_model() == "opus"


def test_api_key_mode_with_base_url(cfg, monkeypatch):
    monkeypatch.setenv("R2R_BASE_URL", "https://proxy.example/Anthropic")
    env = cfg.provider_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://proxy.example/Anthropic"
    assert env["ENABLE_TOOL_SEARCH"] == "true"


def test_env_defines_custom_headers_gateway(cfg, monkeypatch):
    monkeypatch.setenv("R2R_PROVIDER", "amd")
    monkeypatch.setenv("R2R_AUTH_MODE", "custom_headers")
    monkeypatch.setenv("R2R_BASE_URL", "https://gw.example/Anthropic")
    monkeypatch.setenv("R2R_AUTH_HEADERS", "X-Key={key};user={user}")
    monkeypatch.setenv("R2R_AUTH_KEY", "secret123")
    monkeypatch.setenv("R2R_AUTH_USER", "alice")
    monkeypatch.setenv("R2R_MODEL_REASONING", "claude-opus-4-1")

    env = cfg.provider_env()
    assert env["ANTHROPIC_BASE_URL"] == "https://gw.example/Anthropic"
    assert env["ANTHROPIC_CUSTOM_HEADERS"] == "X-Key: secret123\nuser: alice"
    assert env["ENABLE_TOOL_SEARCH"] == "true"
    assert env["ANTHROPIC_AUTH_TOKEN"] == "gateway"
    assert cfg.reasoning_model() == "claude-opus-4-1"
    assert cfg.active_provider_name() == "amd"


def test_gateway_missing_key_raises(cfg, monkeypatch):
    monkeypatch.setenv("R2R_PROVIDER", "amd")
    monkeypatch.setenv("R2R_AUTH_MODE", "custom_headers")
    monkeypatch.setenv("R2R_BASE_URL", "https://gw.example/Anthropic")
    monkeypatch.setenv("R2R_AUTH_HEADERS", "X-Key={key}")
    with pytest.raises(RuntimeError):
        cfg.provider_env()


def test_gateway_missing_base_url_raises(cfg, monkeypatch):
    monkeypatch.setenv("R2R_AUTH_MODE", "custom_headers")
    monkeypatch.setenv("R2R_AUTH_HEADERS", "X-Key={key}")
    monkeypatch.setenv("R2R_AUTH_KEY", "abc")
    with pytest.raises(RuntimeError):
        cfg.provider_env()


def test_run_guardrail_overrides(cfg, monkeypatch):
    assert cfg.max_turns() == 200
    assert cfg.max_budget_usd() == 100.0
    monkeypatch.setenv("R2R_MAX_TURNS", "7")
    monkeypatch.setenv("R2R_MAX_BUDGET_USD", "12.5")
    assert cfg.max_turns() == 7
    assert cfg.max_budget_usd() == 12.5
