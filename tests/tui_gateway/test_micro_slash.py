"""TUI parity tests for the session-scoped ``/micro`` command."""

from __future__ import annotations

import importlib
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import yaml

from hermes_state import SessionDB


class RecordingEngine:
    def __init__(self, *, supported: bool = True) -> None:
        self.supported = supported
        self.enabled = None
        self.calls: list[bool] = []

    def set_micro_compact_enabled(self, enabled: bool) -> None:
        if not self.supported:
            raise AttributeError("runtime switch unsupported")
        self.enabled = enabled
        self.calls.append(enabled)


class LiveAgent:
    def __init__(self, db: SessionDB, session_id: str, *, supported: bool = True) -> None:
        self.session_id = session_id
        self.model = "test-model"
        self.provider = "test-provider"
        self.context_compressor = RecordingEngine(supported=supported)
        self._session_init_model_config: dict = {}
        self.micro_compact_enabled = False
        self.micro_compact_override = None
        self.micro_compact_source = "inherit"
        self.micro_compact_runtime_supported = False
        self._micro_compact_global_value = False
        self._db = db
        self.ensure_calls = 0
        self.run_conversation_calls = 0

    def _ensure_db_session(self) -> None:
        self.ensure_calls += 1
        if self._db.get_session(self.session_id) is None:
            self._db.create_session(self.session_id, source="tui", model=self.model)

    def run_conversation(self, *_args, **_kwargs):
        self.run_conversation_calls += 1


class RecordingWorker:
    instances: list["RecordingWorker"] = []

    def __init__(self, *_args, **_kwargs) -> None:
        self.commands: list[str] = []
        self.closed = False
        self.__class__.instances.append(self)

    def run(self, command: str) -> str:
        self.commands.append(command)
        return "worker handled /micro"

    def close(self) -> None:
        self.closed = True


@pytest.fixture()
def gateway(tmp_path, monkeypatch):
    home = tmp_path / "launch-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    (home / "config.yaml").write_text(
        yaml.safe_dump({"compression": {"micro_compact": False}}),
        encoding="utf-8",
    )

    server = importlib.import_module("tui_gateway.server")
    db = SessionDB(db_path=home / "state.db")
    old_methods = dict(server._methods)
    old_sessions = dict(server._sessions)
    old_db = server._db
    old_db_error = server._db_error
    server._methods.clear()
    server._methods.update(old_methods)
    server._sessions.clear()
    server._db = db
    server._db_error = None
    RecordingWorker.instances.clear()
    try:
        yield server, db, home
    finally:
        for sid in list(server._sessions):
            session = server._sessions.pop(sid)
            worker = session.get("slash_worker")
            if worker is not None:
                worker.close()
        server._methods.clear()
        server._methods.update(old_methods)
        server._sessions.clear()
        server._db = old_db
        server._db_error = old_db_error
        db.close()


def _session(
    server,
    db,
    sid: str,
    key: str,
    agent,
    *,
    profile_home: Path | None = None,
    row_db: SessionDB | None = None,
):
    (row_db or db).create_session(key, source="tui", model="test-model")
    server._sessions[sid] = {
        "session_key": key,
        "agent": agent,
        "history": [],
        "history_lock": threading.Lock(),
        "running": False,
        "profile_home": str(profile_home) if profile_home else None,
        "slash_worker": None,
        "model": "test-model",
    }
    return server._sessions[sid]


def _call(server, method: str, **params) -> dict:
    return server._methods[method]("rid", params)


def _result(response: dict) -> dict:
    assert "result" in response, response
    return response["result"]


def test_command_resolve_exposes_micro_tui_metadata(gateway):
    server, _db, _home = gateway

    result = _result(_call(server, "command.resolve", name="/micro"))

    assert result["canonical"] == "micro"
    assert result["args_hint"] == "[on|off|inherit|status]"
    assert result["subcommands"] == ["on", "off", "inherit", "status"]
    assert result["busy_policy"] == "reject"


def test_live_micro_on_off_inherit_status_updates_exact_row_and_engine(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "live-key")
    session = _session(server, db, "sid-live", "live-key", agent)

    on = _result(_call(server, "slash.exec", session_id="sid-live", command="/micro on"))
    assert "override saved: ON" in on["output"]
    assert db.session_micro_compact_override(db.get_session("live-key")) is True
    assert agent.micro_compact_enabled is True
    assert agent.context_compressor.enabled is True
    assert agent.ensure_calls == 1

    status = _result(_call(server, "slash.exec", session_id="sid-live", command="micro status"))
    assert "Micro-compaction: ON" in status["output"]
    assert "Source: session" in status["output"]
    assert agent.ensure_calls == 1

    off = _result(_call(server, "slash.exec", session_id="sid-live", command="micro off"))
    assert "override saved: OFF" in off["output"]
    assert db.session_micro_compact_override(db.get_session("live-key")) is False
    assert agent.micro_compact_enabled is False
    assert agent.context_compressor.enabled is False

    inherit = _result(_call(server, "slash.exec", session_id="sid-live", command="micro inherit"))
    assert "override saved: OFF" in inherit["output"]
    assert db.session_micro_compact_override(db.get_session("live-key")) is None
    assert agent.micro_compact_override is None
    assert agent.run_conversation_calls == 0
    assert session["history"] == []


def test_live_status_is_read_only_and_does_not_ensure_or_set_engine(gateway):
    server, db, _home = gateway
    key = "status-key"
    db.create_session(key, source="tui", model="test-model")
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-status", key, agent)
    calls_before = list(agent.context_compressor.calls)

    result = _result(_call(server, "slash.exec", session_id="sid-status", command="micro status"))

    assert "Micro-compaction: OFF" in result["output"]
    assert "Source: global (inherited)" in result["output"]
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == calls_before
    assert db.session_micro_compact_override(db.get_session(key)) is None


def test_live_invalid_micro_command_has_no_mutation(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "invalid-key")
    _session(server, db, "sid-invalid", "invalid-key", agent)

    response = _call(server, "slash.exec", session_id="sid-invalid", command="micro maybe")

    assert "result" in response
    assert "on|off|inherit|status" in _result(response)["output"]
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []
    assert db.session_micro_compact_override(db.get_session("invalid-key")) is None


def test_live_plugin_engine_saves_and_reports_unsupported(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "plugin-key", supported=False)
    _session(server, db, "sid-plugin", "plugin-key", agent)

    result = _result(_call(server, "slash.exec", session_id="sid-plugin", command="micro on"))

    assert "saved: ON" in result["output"]
    assert "does not support switching micro-compaction" in result["output"]
    assert db.session_micro_compact_override(db.get_session("plugin-key")) is True
    assert agent.micro_compact_enabled is True


def test_two_live_sessions_are_isolated(gateway):
    server, db, _home = gateway
    agent_a = LiveAgent(db, "session-a")
    agent_b = LiveAgent(db, "session-b")
    _session(server, db, "sid-a", "session-a", agent_a)
    _session(server, db, "sid-b", "session-b", agent_b)

    _result(_call(server, "slash.exec", session_id="sid-a", command="micro on"))
    _result(_call(server, "slash.exec", session_id="sid-b", command="micro off"))

    assert db.session_micro_compact_override(db.get_session("session-a")) is True
    assert db.session_micro_compact_override(db.get_session("session-b")) is False
    assert agent_a.micro_compact_enabled is True
    assert agent_b.micro_compact_enabled is False
    assert agent_a.context_compressor.calls == [True]
    assert agent_b.context_compressor.calls == [False]


def test_busy_micro_rejects_before_db_or_live_mutation(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "busy-key")
    session = _session(server, db, "sid-busy", "busy-key", agent)
    session["running"] = True
    before = db.get_session("busy-key")

    response = _call(server, "slash.exec", session_id="sid-busy", command="micro on")

    assert "error" in response
    assert response["error"]["code"] == 4009
    assert "busy" in response["error"]["message"].lower()
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []
    assert db.get_session("busy-key") == before


def test_no_live_agent_uses_worker_and_does_not_touch_other_live_agent(gateway, monkeypatch):
    server, db, _home = gateway
    other = LiveAgent(db, "other-key")
    _session(server, db, "sid-other", "other-key", other)
    _session(server, db, "sid-cold", "cold-key", None)
    monkeypatch.setattr(server, "_SlashWorker", RecordingWorker)

    result = _result(_call(server, "slash.exec", session_id="sid-cold", command="/micro on"))

    assert result["output"] == "worker handled /micro"
    assert RecordingWorker.instances[-1].commands == ["/micro on"]
    assert other.ensure_calls == 0
    assert other.context_compressor.calls == []
    assert db.session_micro_compact_override(db.get_session("other-key")) is None


def test_profile_session_uses_profile_db_and_readonly_config(gateway, tmp_path):
    server, launch_db, launch_home = gateway
    profile_home = tmp_path / "profile-a"
    profile_home.mkdir()
    (profile_home / "config.yaml").write_text(
        yaml.safe_dump({"compression": {"micro_compact": True}}),
        encoding="utf-8",
    )
    profile_db = SessionDB(db_path=profile_home / "state.db")
    try:
        key = "profile-key"
        profile_db.create_session(key, source="tui", model="test-model")
        agent = LiveAgent(profile_db, key)
        _session(
            server,
            launch_db,
            "sid-profile",
            key,
            agent,
            profile_home=profile_home,
            row_db=profile_db,
        )

        result = _result(
            _call(server, "slash.exec", session_id="sid-profile", command="micro status")
        )

        assert "Global: ON" in result["output"]
        assert "Source: global (inherited)" in result["output"]
        assert launch_db.get_session(key) is None
        assert profile_db.session_micro_compact_override(profile_db.get_session(key)) is None
        assert launch_home.joinpath("config.yaml").read_text(encoding="utf-8")
    finally:
        profile_db.close()


def test_worker_state_hydrates_later_live_agent(gateway, monkeypatch):
    server, db, _home = gateway
    _session(server, db, "sid-cold-hydrate", "cold-hydrate", None)

    class DurableWorker(RecordingWorker):
        def run(self, command: str) -> str:
            self.commands.append(command)
            db.set_session_micro_compact_override("cold-hydrate", True)
            return "worker saved"

    monkeypatch.setattr(server, "_SlashWorker", DurableWorker)
    result = _result(
        _call(server, "slash.exec", session_id="sid-cold-hydrate", command="micro on")
    )
    assert result["output"] == "worker saved"

    agent = LiveAgent(db, "cold-hydrate")
    from hermes_cli.micro_command import hydrate_micro_compact_policy_for_session

    hydrate_micro_compact_policy_for_session(
        agent=agent,
        session_db=db,
        session_id="cold-hydrate",
        config_loader=lambda: {"compression": {"micro_compact": False}},
    )
    assert agent.micro_compact_enabled is True
    assert agent.micro_compact_override is True


def test_slash_exec_live_micro_never_calls_worker(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "direct-key")
    _session(server, db, "sid-direct", "direct-key", agent)

    def fail_worker(*_args, **_kwargs):
        pytest.fail("live /micro must not start a slash worker")

    monkeypatch.setattr(server, "_SlashWorker", fail_worker)
    _result(_call(server, "slash.exec", session_id="sid-direct", command="micro on"))
    assert agent.context_compressor.enabled is True


def test_live_micro_rejects_mismatched_session_identity_without_mutation(gateway, monkeypatch):
    server, db, _home = gateway
    session_key = "session-row"
    agent_key = "agent-row"
    db.create_session(session_key, source="tui", model="test-model")
    db.create_session(agent_key, source="tui", model="test-model")
    agent = LiveAgent(db, agent_key)
    session = _session(server, db, "sid-mismatch", session_key, agent)
    session_row_before = db.get_session(session_key)
    agent_row_before = db.get_session(agent_key)
    engine_enabled_before = agent.context_compressor.enabled

    def fail_worker(*_args, **_kwargs):
        pytest.fail("mismatched live /micro must not start a slash worker")

    monkeypatch.setattr(server, "_SlashWorker", fail_worker)
    response = _call(server, "slash.exec", session_id="sid-mismatch", command="/micro on")

    output = _result(response)["output"]
    assert "session identity mismatch" in output.lower()
    assert db.get_session(session_key) == session_row_before
    assert db.get_session(agent_key) == agent_row_before
    assert db.session_micro_compact_override(db.get_session(session_key)) is None
    assert db.session_micro_compact_override(db.get_session(agent_key)) is None
    assert agent.ensure_calls == 0
    assert agent.context_compressor.enabled is engine_enabled_before
    assert agent.context_compressor.calls == []
    assert session["slash_worker"] is None
