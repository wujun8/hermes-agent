"""Behavioral tests for the optional Codex app-server whole-turn timeout config."""

from __future__ import annotations

import logging
from pathlib import Path
from types import SimpleNamespace

import pytest


_SETTING = "codex_app_server_turn_timeout_seconds"
_MISSING = object()


def _apply_agent_config(raw=_MISSING):
    """Apply just the agent section without constructing a live model client."""
    from agent.agent_init import _apply_agent_section

    section = {"environment_probe": False}
    if raw is not _MISSING:
        section[_SETTING] = raw
    agent = SimpleNamespace(run_budget_seconds=None)
    _apply_agent_section(agent, {"agent": section})
    return agent


def _make_real_agent(home: Path, monkeypatch, raw):
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / ".env").write_text("", encoding="utf-8")
    (home / "config.yaml").write_text(
        "agent:\n"
        "  environment_probe: false\n"
        f"  {_SETTING}: {raw}\n",
        encoding="utf-8",
    )
    from run_agent import AIAgent
    return AIAgent(
        model="gpt-5.6-sol",
        provider="custom",
        api_key="test-key",
        base_url="https://example.invalid/v1",
        api_mode="codex_app_server",
        quiet_mode=True,
        skip_context_files=True,
        skip_memory=True,
        platform="cli",
    )


def test_timeout_defaults_to_none_and_accepts_positive_finite_values(monkeypatch):
    from hermes_cli.config_defaults import DEFAULT_CONFIG

    assert DEFAULT_CONFIG["agent"][_SETTING] is None
    # The ordinary API timeout must not become a whole-turn app-server default.
    monkeypatch.setenv("HERMES_API_TIMEOUT", "123")
    assert getattr(_apply_agent_config(), _SETTING) is None
    assert getattr(_apply_agent_config(900), _SETTING) == 900.0
    assert getattr(_apply_agent_config("0.5"), _SETTING) == 0.5


def test_real_agent_loads_timeout_from_profile_config(tmp_path, monkeypatch):
    agent = _make_real_agent(tmp_path, monkeypatch, 7200)
    assert agent.codex_app_server_turn_timeout_seconds == 7200.0


@pytest.mark.parametrize("raw", [None, 0, -1, 0.0, "0", "-4.5"])
def test_null_and_non_positive_values_disable_timeout(raw):
    assert getattr(_apply_agent_config(raw), _SETTING) is None


@pytest.mark.parametrize(
    "raw",
    [True, False, "not-a-number", float("nan"), float("inf"), float("-inf")],
)
def test_invalid_and_non_finite_values_disable_timeout_with_warning(raw, caplog):
    with caplog.at_level(logging.WARNING, logger="run_agent"):
        agent = _apply_agent_config(raw)

    assert getattr(agent, _SETTING) is None
    assert any(_SETTING in record.getMessage() for record in caplog.records)
