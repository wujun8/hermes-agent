"""TUI parity tests for the session-scoped ``/micro`` command."""

from __future__ import annotations

import contextlib
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


class RecordingLease:
    def __init__(self, lease_id: str = "lease") -> None:
        self.lease_id = lease_id
        self.release_calls = 0

    def release(self) -> None:
        self.release_calls += 1


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


class _MicroControlAbort(BaseException):
    """Non-Exception failure used to prove lease cleanup covers BaseException."""


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


def _assert_history_lock_available(lock: threading.Lock) -> None:
    acquired = threading.Event()

    def acquire_and_release() -> None:
        with lock:
            acquired.set()

    thread = threading.Thread(target=acquire_and_release)
    thread.start()
    assert acquired.wait(2), "history_lock was held during blocking live work"
    thread.join(timeout=2)
    assert not thread.is_alive()


def test_live_micro_config_and_db_work_run_without_history_lock(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "outside-lock")
    session = _session(server, db, "sid-outside-lock", "outside-lock", agent)
    config_entered = threading.Event()
    config_release = threading.Event()
    setter_entered = threading.Event()
    setter_release = threading.Event()
    apply_entered = threading.Event()
    apply_release = threading.Event()
    original_setter = db.set_session_micro_compact_override
    from agent import agent_init

    original_apply = agent_init.apply_micro_compact_policy

    def blocked_config():
        config_entered.set()
        assert config_release.wait(2)
        return {"compression": {"micro_compact": False}}

    def blocked_setter(session_id, override):
        setter_entered.set()
        assert setter_release.wait(2)
        return original_setter(session_id, override)

    def blocked_apply(*args, **kwargs):
        apply_entered.set()
        assert apply_release.wait(2)
        return original_apply(*args, **kwargs)

    monkeypatch.setattr(server, "_session_micro_config_loader", lambda _session: blocked_config)
    monkeypatch.setattr(db, "set_session_micro_compact_override", blocked_setter)
    monkeypatch.setattr(agent_init, "apply_micro_compact_policy", blocked_apply)
    result = {}

    def run_micro() -> None:
        result["response"] = _call(
            server, "slash.exec", session_id="sid-outside-lock", command="micro on"
        )

    thread = threading.Thread(target=run_micro)
    thread.start()
    try:
        assert config_entered.wait(2)
        _assert_history_lock_available(session["history_lock"])
        config_release.set()
        assert setter_entered.wait(2)
        _assert_history_lock_available(session["history_lock"])
        setter_release.set()
        assert apply_entered.wait(2)
        _assert_history_lock_available(session["history_lock"])
    finally:
        config_release.set()
        setter_release.set()
        apply_release.set()
        thread.join(timeout=2)
    assert not thread.is_alive()
    assert "result" in result["response"]


def test_live_micro_lease_blocks_prompt_and_second_micro_until_release(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "lease-busy")
    session = _session(server, db, "sid-lease-busy", "lease-busy", agent)
    setter_entered = threading.Event()
    setter_release = threading.Event()
    original_setter = db.set_session_micro_compact_override

    def blocked_setter(session_id, override):
        setter_entered.set()
        assert setter_release.wait(2)
        return original_setter(session_id, override)

    monkeypatch.setattr(db, "set_session_micro_compact_override", blocked_setter)
    monkeypatch.setattr(server, "_ensure_active_session_slot", lambda *_args: None)
    first = {}
    second = {}
    prompt = {}
    second_done = threading.Event()
    prompt_done = threading.Event()

    def run_first() -> None:
        first["response"] = _call(
            server, "slash.exec", session_id="sid-lease-busy", command="micro on"
        )

    def run_second() -> None:
        second["response"] = _call(
            server, "slash.exec", session_id="sid-lease-busy", command="micro off"
        )
        second_done.set()

    def run_prompt() -> None:
        prompt["response"] = _call(
            server, "prompt.submit", session_id="sid-lease-busy", text="must wait"
        )
        prompt_done.set()

    first_thread = threading.Thread(target=run_first)
    second_thread = threading.Thread(target=run_second)
    prompt_thread = threading.Thread(target=run_prompt)
    first_thread.start()
    try:
        assert setter_entered.wait(2)
        second_thread.start()
        prompt_thread.start()
        assert second_done.wait(2), "second /micro did not get a stable busy response"
        assert prompt_done.wait(2), "prompt.submit did not get a stable busy response"
    finally:
        setter_release.set()
        first_thread.join(timeout=2)
        second_thread.join(timeout=2)
        prompt_thread.join(timeout=2)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert not prompt_thread.is_alive()
    assert second["response"]["error"]["code"] == 4009
    assert prompt["response"]["error"]["code"] == 4009
    assert session["running"] is False
    assert session["history"] == []
    assert agent.run_conversation_calls == 0


def test_prompt_control_race_releases_only_new_active_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-control-race")
    session = _session(server, db, "sid-prompt-control-race", "prompt-control-race", agent)

    lease = RecordingLease("prompt-control-race")
    claim_entered = threading.Event()
    claim_release = threading.Event()

    def blocked_claim(*_args, **_kwargs):
        claim_entered.set()
        assert claim_release.wait(5), "active-session claim was not released"
        return lease, None

    monkeypatch.setattr(server, "_claim_active_session_slot", blocked_claim)
    prompt_result: dict = {}

    def run_prompt() -> None:
        prompt_result["response"] = _call(
            server,
            "prompt.submit",
            session_id="sid-prompt-control-race",
            text="must not run",
        )

    prompt_thread = threading.Thread(target=run_prompt)
    prompt_thread.start()
    assert claim_entered.wait(2), "prompt did not reach active-session claim"

    micro_thread, micro_release, micro_result = _start_blocked_micro(
        server,
        db,
        monkeypatch,
        sid="sid-prompt-control-race",
    )
    try:
        claim_release.set()
        prompt_thread.join(timeout=5)
        assert not prompt_thread.is_alive()
        response = prompt_result["response"]
        assert response["error"]["code"] == 4009
        assert session["running"] is False
        assert session["history"] == []
        assert agent.run_conversation_calls == 0
        assert session.get("active_session_lease") is None
        assert lease.release_calls == 1
    finally:
        _join_blocked_micro(micro_thread, micro_release)
    assert "result" in micro_result["response"]


def test_prompt_control_race_preserves_preexisting_active_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-control-preexisting")
    session = _session(
        server,
        db,
        "sid-prompt-control-preexisting",
        "prompt-control-preexisting",
        agent,
    )
    lease = RecordingLease("preexisting")
    session["active_session_lease"] = lease
    config_entered = threading.Event()
    config_release = threading.Event()

    def blocked_config():
        config_entered.set()
        assert config_release.wait(5), "prompt preflight was not released"
        return {}

    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", blocked_config)
    prompt_result: dict = {}

    def run_prompt() -> None:
        prompt_result["response"] = _call(
            server,
            "prompt.submit",
            session_id="sid-prompt-control-preexisting",
            text="must not run",
        )

    prompt_thread = threading.Thread(target=run_prompt)
    prompt_thread.start()
    assert config_entered.wait(2), "prompt did not reach the preflight barrier"
    micro_thread, micro_release, micro_result = _start_blocked_micro(
        server,
        db,
        monkeypatch,
        sid="sid-prompt-control-preexisting",
    )
    try:
        config_release.set()
        prompt_thread.join(timeout=5)
        assert not prompt_thread.is_alive()
        assert prompt_result["response"]["error"]["code"] == 4009
        assert session.get("active_session_lease") is lease
        assert lease.release_calls == 0
        assert session["running"] is False
        assert session["history"] == []
        assert agent.run_conversation_calls == 0
    finally:
        _join_blocked_micro(micro_thread, micro_release)
    assert "result" in micro_result["response"]


def test_prompt_exact_lease_cleanup_skips_replaced_session(gateway, monkeypatch):
    server, db, _home = gateway
    old_agent = LiveAgent(db, "prompt-aba-old")
    old_session = _session(server, db, "sid-prompt-aba", "prompt-aba-old", old_agent)
    old_lease = RecordingLease("aba-old")
    old_session["active_session_lease"] = old_lease
    new_agent = LiveAgent(db, "prompt-aba-new")
    new_session = _session(server, db, "sid-prompt-aba-new", "prompt-aba-new", new_agent)
    new_lease = RecordingLease("aba-new")
    new_session["active_session_lease"] = new_lease

    registry_checked = threading.Event()
    registry_release = threading.Event()

    def blocked_child_check(_session_key: str) -> bool:
        registry_checked.set()
        assert registry_release.wait(5), "cleanup registry seam was not released"
        return False

    monkeypatch.setattr(server, "_child_run_active", blocked_child_check)
    cleanup_result: dict = {}

    def run_cleanup() -> None:
        cleanup_result["value"] = server._release_prompt_active_session_slot(
            "sid-prompt-aba",
            old_session,
            old_lease,
            lease_token=old_lease.lease_id,
        )

    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.start()
    assert registry_checked.wait(2), "cleanup did not reach the replacement seam"
    with server._sessions_lock:
        server._sessions["sid-prompt-aba"] = new_session
    registry_release.set()
    cleanup_thread.join(timeout=5)
    assert not cleanup_thread.is_alive()
    assert cleanup_result["value"] is False
    assert old_session.get("active_session_lease") is old_lease
    assert old_lease.release_calls == 0
    assert new_session.get("active_session_lease") is new_lease
    assert new_lease.release_calls == 0


def test_prompt_exact_lease_cleanup_skips_changed_lease(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-lease-aba")
    session = _session(server, db, "sid-prompt-lease-aba", "prompt-lease-aba", agent)
    old_lease = RecordingLease("lease-old")
    new_lease = RecordingLease("lease-new")
    session["active_session_lease"] = new_lease

    result = server._release_prompt_active_session_slot(
        "sid-prompt-lease-aba",
        session,
        old_lease,
        lease_token=old_lease.lease_id,
    )

    assert result is False
    assert session.get("active_session_lease") is new_lease
    assert old_lease.release_calls == 0
    assert new_lease.release_calls == 0


def test_successful_prompt_transfers_new_lease_to_normal_turn_lifecycle(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-success-lease")
    session = _session(server, db, "sid-prompt-success-lease", "prompt-success-lease", agent)
    lease = RecordingLease("prompt-success")
    monkeypatch.setattr(server, "_claim_active_session_slot", lambda *a, **k: (lease, None))
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", lambda: {})
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: None)

    response = _call(
        server,
        "prompt.submit",
        session_id="sid-prompt-success-lease",
        text="normal turn",
    )

    assert response["result"]["status"] == "streaming"
    assert session["running"] is True
    assert session.get("active_session_lease") is lease
    assert lease.release_calls == 0
    server._release_active_session_slot(session)
    assert lease.release_calls == 1


def test_concurrent_prompts_do_not_release_winner_active_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-two-winners")
    session = _session(server, db, "sid-prompt-two-winners", "prompt-two-winners", agent)
    lease = RecordingLease("prompt-two-winners")
    claim_entered = threading.Event()
    claim_release = threading.Event()
    preflight_barrier = threading.Barrier(2)
    claim_calls = 0
    claim_lock = threading.Lock()

    def blocked_claim(*_args, **_kwargs):
        nonlocal claim_calls
        with claim_lock:
            claim_calls += 1
            first = claim_calls == 1
        if first:
            claim_entered.set()
            assert claim_release.wait(5), "first prompt claim was not released"
            return lease, None
        raise AssertionError("second prompt tried to acquire a replacement lease")

    def synchronized_config():
        preflight_barrier.wait(timeout=5)
        return {}

    monkeypatch.setattr(server, "_claim_active_session_slot", blocked_claim)
    monkeypatch.setattr(server, "_load_dashboard_process_isolation_config", synchronized_config)
    monkeypatch.setattr(server, "_ensure_session_db_row", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_persist_branch_seed", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_start_agent_build", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_wait_agent_for_prompt", lambda *_a, **_k: None)
    monkeypatch.setattr(server, "_run_prompt_submit", lambda *_a, **_k: None)
    monkeypatch.setattr(
        server,
        "_handle_busy_submit",
        lambda *_a, **_k: {"error": {"code": 4009, "message": "session busy"}},
    )
    responses: list[dict] = []

    def run_prompt() -> None:
        responses.append(
            _call(
                server,
                "prompt.submit",
                session_id="sid-prompt-two-winners",
                text="concurrent",
            )
        )

    first_thread = threading.Thread(target=run_prompt)
    second_thread = threading.Thread(target=run_prompt)
    first_thread.start()
    assert claim_entered.wait(2), "first prompt did not reach active-session claim"
    second_thread.start()
    claim_release.set()
    first_thread.join(timeout=5)
    second_thread.join(timeout=5)
    assert not first_thread.is_alive()
    assert not second_thread.is_alive()
    assert len(responses) == 2
    assert any(response.get("result", {}).get("status") == "streaming" for response in responses)
    assert any(response.get("error", {}).get("code") == 4009 for response in responses)
    assert session["running"] is True
    assert session.get("active_session_lease") is lease
    assert lease.release_calls == 0
    server._release_active_session_slot(session)


def test_prompt_exact_lease_cleanup_failure_is_non_masking_and_unlocks(gateway, monkeypatch, caplog):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-cleanup-failure")
    session = _session(server, db, "sid-prompt-cleanup-failure", "prompt-cleanup-failure", agent)
    lease = RecordingLease("cleanup-failure")
    session["active_session_lease"] = lease
    calls: list[tuple[tuple, dict]] = []

    class CleanupAbort(BaseException):
        pass

    def failing_canonical(*args, **kwargs):
        calls.append((args, kwargs))
        raise CleanupAbort("release seam failed")

    monkeypatch.setattr(server, "_release_active_session_slot", failing_canonical)
    result = server._release_prompt_active_session_slot(
        "sid-prompt-cleanup-failure",
        session,
        lease,
        lease_token=lease.lease_id,
    )

    assert result is False
    assert len(calls) == 1
    assert "sid-prompt-cleanup-failure" in caplog.text
    with session["history_lock"]:
        pass
    with server._sessions_lock:
        pass


def test_prompt_exact_lease_release_does_not_hold_coordination_locks(gateway):
    server, db, _home = gateway
    agent = LiveAgent(db, "prompt-release-seam")
    session = _session(server, db, "sid-prompt-release-seam", "prompt-release-seam", agent)
    release_entered = threading.Event()
    release_gate = threading.Event()

    class BlockingLease(RecordingLease):
        def release(self) -> None:
            release_entered.set()
            assert release_gate.wait(5), "lease release seam was not released"
            super().release()

    lease = BlockingLease("release-seam")
    session["active_session_lease"] = lease
    cleanup_result: dict = {}

    def run_cleanup() -> None:
        cleanup_result["value"] = server._release_prompt_active_session_slot(
            "sid-prompt-release-seam",
            session,
            lease,
            lease_token=lease.lease_id,
        )

    cleanup_thread = threading.Thread(target=run_cleanup)
    cleanup_thread.start()
    try:
        assert release_entered.wait(2), "cleanup did not reach the release seam"
        _assert_history_lock_available(session["history_lock"])
        _assert_history_lock_available(server._sessions_lock)
    finally:
        release_gate.set()
        cleanup_thread.join(timeout=5)
    assert not cleanup_thread.is_alive()
    assert cleanup_result["value"] is True
    assert lease.release_calls == 1


def test_manual_compression_routes_reject_before_any_mutation_during_micro(
    gateway, monkeypatch
):
    server, db, _home = gateway
    key = "manual-compress-busy"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-manual-compress-busy", key, agent)
    session["history"] = [{"role": "user", "content": "before"}]
    history_before = list(session["history"])
    row_before = db.get_session(key)
    key_before = session["session_key"]
    generation_before = session.get("session_generation", 0)
    compressor_calls: list[tuple] = []

    def fail_compress(*args, **kwargs):
        compressor_calls.append((args, kwargs))
        pytest.fail("manual compression must reject before invoking the compressor")

    monkeypatch.setattr(server, "_compress_session_history", fail_compress)
    thread, release, first = _start_blocked_micro(
        server, db, monkeypatch, sid="sid-manual-compress-busy"
    )
    try:
        responses = [
            _call(
                server,
                "slash.exec",
                session_id="sid-manual-compress-busy",
                command="/compress",
            ),
            _call(
                server,
                "slash.exec",
                session_id="sid-manual-compress-busy",
                command="/compact",
            ),
            _call(
                server,
                "command.dispatch",
                session_id="sid-manual-compress-busy",
                name="compress",
                arg="",
            ),
            _call(
                server,
                "session.compress",
                session_id="sid-manual-compress-busy",
            ),
        ]
        assert all(response.get("error", {}).get("code") == 4009 for response in responses)
        assert all("busy" in response["error"]["message"].lower() for response in responses)
        assert compressor_calls == []
        assert session["session_key"] == key_before
        assert session.get("session_generation", 0) == generation_before
        assert session["history"] == history_before
        assert db.get_session(key) == row_before
    finally:
        _join_blocked_micro(thread, release)
    assert "result" in first["response"]


def _assert_micro_lease_cleared(
    session: dict, *, expected_running: bool = False
) -> None:
    assert session.get("_session_control_inflight") is None
    assert session.get("running") is expected_running


@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, _MicroControlAbort])
def test_live_micro_config_failure_clears_lease_and_allows_retry(gateway, failure_type):
    server, db, _home = gateway
    key = "config-failure"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-config-failure", key, agent)
    row_before = db.get_session(key)
    failure = failure_type("config boom")

    def failing_config():
        raise failure

    with patch.object(server, "_session_micro_config_loader", return_value=failing_config):
        if issubclass(failure_type, Exception):
            response = _call(
                server,
                "slash.exec",
                session_id="sid-config-failure",
                command="micro on",
            )
            assert "could not read global configuration" in _result(response)["output"]
        else:
            with pytest.raises(failure_type):
                _call(
                    server,
                    "slash.exec",
                    session_id="sid-config-failure",
                    command="micro on",
                )

    _assert_micro_lease_cleared(session)
    assert db.get_session(key) == row_before
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []

    retry = _result(
        _call(server, "slash.exec", session_id="sid-config-failure", command="micro on")
    )
    assert "override saved: ON" in retry["output"]
    assert db.session_micro_compact_override(db.get_session(key)) is True
    assert agent.context_compressor.calls == [True]
    _assert_micro_lease_cleared(session)


@pytest.mark.parametrize("boundary", ["db_context", "db_read", "ensure", "db_set", "live_apply"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, _MicroControlAbort])
def test_live_micro_failure_matrix_releases_lease_and_preserves_db_first_contract(
    gateway, monkeypatch, boundary, failure_type
):
    server, db, _home = gateway
    key = f"{boundary}-failure-{failure_type.__name__}"
    agent = LiveAgent(db, key)
    session = _session(server, db, f"sid-{key}", key, agent)
    row_before = db.get_session(key)
    failure = failure_type(f"{boundary} boom")

    def invoke_and_assert(*, propagates: bool = False):
        if propagates or not issubclass(failure_type, Exception):
            with pytest.raises(failure_type):
                _call(server, "slash.exec", session_id=f"sid-{key}", command="micro on")
            return
        response = _call(server, "slash.exec", session_id=f"sid-{key}", command="micro on")
        assert "result" in response, response
        assert boundary in _result(response)["output"]

    if boundary == "db_context":
        @contextlib.contextmanager
        def failing_session_db(_session_view):
            raise failure
            yield None

        with patch.object(server, "_session_db", failing_session_db):
            invoke_and_assert(propagates=True)
    elif boundary == "db_read":
        with patch.object(agent, "_ensure_db_session", return_value=None), patch.object(
            db, "get_session", side_effect=failure
        ):
            invoke_and_assert()
    elif boundary == "ensure":
        with patch.object(agent, "_ensure_db_session", side_effect=failure):
            invoke_and_assert()
    elif boundary == "db_set":
        with patch.object(db, "set_session_micro_compact_override", side_effect=failure):
            invoke_and_assert()
    else:
        with patch("agent.agent_init.apply_micro_compact_policy", side_effect=failure):
            invoke_and_assert()

    _assert_micro_lease_cleared(session)
    if boundary == "live_apply":
        assert db.session_micro_compact_override(db.get_session(key)) is True
        assert agent.micro_compact_enabled is False
    else:
        assert db.get_session(key) == row_before
    assert agent.context_compressor.calls == []

    # Restore the failed seam and prove the same session can claim the lease
    # again.  ``off`` also checks that a DB-first failure did not leave a stale
    # durable override behind.
    retry = _result(
        _call(server, "slash.exec", session_id=f"sid-{key}", command="micro off")
    )
    assert "override saved: OFF" in retry["output"]
    assert db.session_micro_compact_override(db.get_session(key)) is False
    _assert_micro_lease_cleared(session)


@pytest.mark.parametrize("failure_type", [RuntimeError, _MicroControlAbort])
def test_live_micro_profile_session_db_open_failure_clears_lease_and_allows_retry(
    gateway, tmp_path, failure_type
):
    server, db, _home = gateway
    from hermes_state import SessionDB

    key = f"profile-open-{failure_type.__name__}"
    profile_home = tmp_path / "profile-open-home"
    profile_home.mkdir()
    profile_db = SessionDB(db_path=profile_home / "state.db")
    profile_db.create_session(key, source="tui", model="test-model")
    profile_db.close()
    agent = LiveAgent(db, key)
    session = _session(server, db, f"sid-{key}", key, agent)
    session["profile_home"] = str(profile_home)
    row_before = db.get_session(key)
    failure = failure_type("profile SessionDB open boom")

    class FailingSessionDB:
        def __init__(self, **_kwargs):
            raise failure

    with patch("hermes_state.SessionDB", FailingSessionDB), patch.object(
        server, "_session_micro_config_loader", return_value=lambda: {"compression": {}}
    ):
        if issubclass(failure_type, Exception):
            response = _call(server, "slash.exec", session_id=f"sid-{key}", command="micro on")
            assert "session database is not available" in _result(response)["output"]
        else:
            with pytest.raises(failure_type):
                _call(server, "slash.exec", session_id=f"sid-{key}", command="micro on")

    _assert_micro_lease_cleared(session)
    assert db.get_session(key) == row_before
    assert agent.context_compressor.calls == []

    # The profile DB opens normally once the seam is restored, and the exact
    # target row can be mutated by the next claim.
    with patch.object(agent, "_ensure_db_session", return_value=None):
        retry = _result(
            _call(server, "slash.exec", session_id=f"sid-{key}", command="micro off")
        )
    assert "override saved: OFF" in retry["output"]
    check_db = SessionDB(db_path=profile_home / "state.db")
    try:
        assert check_db.session_micro_compact_override(check_db.get_session(key)) is False
    finally:
        check_db.close()
    assert agent.context_compressor.calls == [False]
    _assert_micro_lease_cleared(session)


@pytest.mark.parametrize("busy_source", ["running", "child"])
def test_busy_micro_rejects_before_lease_config_db_or_live_mutation(
    gateway, monkeypatch, busy_source
):
    server, db, _home = gateway
    key = f"busy-{busy_source}"
    agent = LiveAgent(db, key)
    session = _session(server, db, f"sid-{key}", key, agent)
    row_before = db.get_session(key)
    if busy_source == "running":
        session["running"] = True
    else:
        monkeypatch.setattr(server, "_child_run_active", lambda _key: True)

    def fail_config(_session):
        pytest.fail("busy /micro must reject before loading config")

    monkeypatch.setattr(server, "_session_micro_config_loader", fail_config)
    monkeypatch.setattr(
        db,
        "set_session_micro_compact_override",
        lambda *_args, **_kwargs: pytest.fail("busy /micro must not write the DB"),
    )

    response = _call(server, "slash.exec", session_id=f"sid-{key}", command="micro on")

    assert response["error"]["code"] == 4009
    _assert_micro_lease_cleared(
        session, expected_running=(busy_source == "running")
    )
    assert db.get_session(key) == row_before
    assert agent.ensure_calls == 0
    assert agent.context_compressor.calls == []


def _start_blocked_micro(
    server, db, monkeypatch, *, sid: str, blocked_session_id: str | None = None
):
    entered = threading.Event()
    release = threading.Event()
    result: dict = {}
    original_setter = db.set_session_micro_compact_override
    if blocked_session_id is None:
        blocked_session_id = str(
            server._sessions[sid].get("session_key") or sid
        )

    def blocked_setter(session_id, override):
        if session_id == blocked_session_id:
            entered.set()
            assert release.wait(5), "blocked micro was not released"
        return original_setter(session_id, override)

    monkeypatch.setattr(db, "set_session_micro_compact_override", blocked_setter)

    def run_micro() -> None:
        result["response"] = _call(server, "slash.exec", session_id=sid, command="micro on")

    thread = threading.Thread(target=run_micro)
    thread.start()
    assert entered.wait(2), "micro did not reach the blocking DB setter"
    return thread, release, result


def _join_blocked_micro(thread: threading.Thread, release: threading.Event) -> None:
    release.set()
    thread.join(timeout=5)
    assert not thread.is_alive()


def test_session_resume_reuse_rejects_during_micro_and_reuses_after_release(gateway, monkeypatch):
    server, db, _home = gateway
    key = "resume-while-micro"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-resume-while-micro", key, agent)
    thread, release, first = _start_blocked_micro(
        server, db, monkeypatch, sid="sid-resume-while-micro"
    )
    try:
        busy = _call(server, "session.resume", session_id=key)
        assert busy["error"]["code"] == 4009
        assert server._sessions.get("sid-resume-while-micro") is session
        assert db.session_micro_compact_override(db.get_session(key)) is None
    finally:
        _join_blocked_micro(thread, release)

    assert "result" in first["response"]
    resumed = _call(server, "session.resume", session_id=key)
    assert "result" in resumed, resumed
    assert resumed["result"]["session_id"] == "sid-resume-while-micro"
    assert server._sessions.get("sid-resume-while-micro") is session
    assert db.session_micro_compact_override(db.get_session(key)) is True


def test_session_reuse_releases_resume_lock_before_blocking_db_payload_read(gateway, monkeypatch):
    server, db, _home = gateway
    key = "resume-payload-lock"
    agent = LiveAgent(db, key)
    _session(server, db, "sid-resume-payload-lock", key, agent)
    db_read_entered = threading.Event()
    db_read_release = threading.Event()
    resume_result: dict = {}

    def blocked_messages(*_args, **_kwargs):
        db_read_entered.set()
        assert db_read_release.wait(5), "blocked resume payload read was not released"
        return []

    monkeypatch.setattr(db, "get_messages_as_conversation", blocked_messages)

    def resume() -> None:
        resume_result["response"] = _call(server, "session.resume", session_id=key)

    resume_thread = threading.Thread(target=resume)
    resume_thread.start()
    assert db_read_entered.wait(2)
    lock_acquired = threading.Event()

    def acquire_resume_lock() -> None:
        with server._session_resume_lock:
            lock_acquired.set()

    lock_thread = threading.Thread(target=acquire_resume_lock)
    lock_thread.start()
    try:
        assert lock_acquired.wait(2), "resume held the global lock during DB payload I/O"
    finally:
        db_read_release.set()
        resume_thread.join(timeout=5)
        lock_thread.join(timeout=5)
    assert not resume_thread.is_alive()
    assert not lock_thread.is_alive()
    assert "result" in resume_result["response"]


def test_session_close_rejects_during_micro_and_succeeds_after_release(gateway, monkeypatch):
    server, db, _home = gateway
    key = "close-while-micro"
    agent = LiveAgent(db, key)
    session = _session(server, db, "sid-close-while-micro", key, agent)
    thread, release, first = _start_blocked_micro(
        server, db, monkeypatch, sid="sid-close-while-micro"
    )
    try:
        busy = _call(server, "session.close", session_id="sid-close-while-micro")
        assert busy["error"]["code"] == 4009
        assert server._sessions.get("sid-close-while-micro") is session
        assert db.session_micro_compact_override(db.get_session(key)) is None
    finally:
        _join_blocked_micro(thread, release)

    assert "result" in first["response"]
    closed = _call(server, "session.close", session_id="sid-close-while-micro")
    assert closed["result"]["closed"] is True
    assert "sid-close-while-micro" not in server._sessions
    assert db.session_micro_compact_override(db.get_session(key)) is True


def test_compression_rotation_defers_during_micro_and_reanchors_after_release(gateway, monkeypatch):
    server, db, _home = gateway
    old_key = "rotation-old"
    new_key = "rotation-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-rotation", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    generation_before = session.get("session_generation", 0)
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    thread, release, first = _start_blocked_micro(
        server, db, monkeypatch, sid="sid-rotation"
    )
    try:
        # Compression has already produced a continuation id, but the gateway
        # must defer the registry/session-key re-anchor while /micro owns the
        # exact session object.  The production lease release below performs
        # the retry; this test deliberately never calls sync after release.
        agent.session_id = new_key
        status = server._sync_session_key_after_compress("sid-rotation", session)
        assert status == "deferred"
        assert session["session_key"] == old_key
        assert session.get("session_generation", 0) == generation_before
        assert db.session_micro_compact_override(db.get_session(old_key)) is None
        assert db.session_micro_compact_override(db.get_session(new_key)) is None
    finally:
        _join_blocked_micro(thread, release)

    assert "result" in first["response"]
    assert "warning" in first["response"]["result"]["output"].lower()
    assert "old row" in first["response"]["result"]["output"].lower()
    assert session["session_key"] == new_key
    assert session["session_generation"] == generation_before + 1
    assert session.get("_session_control_inflight") is None
    assert session.get("_deferred_compression_rotation") is None
    # The defensive fail-closed path is explicit: /micro changed only the old
    # row and the user must retry on the canonical continuation.
    assert db.session_micro_compact_override(db.get_session(old_key)) is True
    assert db.session_micro_compact_override(db.get_session(new_key)) is None


def test_deferred_rotations_coalesce_latest_target_and_canonical_fields(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "coalesce-old"
    first_key = "coalesce-first"
    latest_key = "coalesce-latest"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-coalesce", old_key, agent)
    db.create_session(first_key, source="tui", model="test-model")
    db.create_session(latest_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    generation_before = session.get("session_generation", 0)
    target, claim_error = server._claim_live_micro_control("sid-coalesce", session)
    assert claim_error is None
    assert target is not None
    try:
        agent.session_id = first_key
        assert server._sync_session_key_after_compress("sid-coalesce", session) == "deferred"
        marker = session["_deferred_compression_rotation"]
        assert marker["agent_session_id"] == first_key
        agent.session_id = latest_key
        assert server._sync_session_key_after_compress("sid-coalesce", session) == "deferred"
        assert len(session["_deferred_compression_rotation"]) <= 7
        assert session["_deferred_compression_rotation"]["agent_session_id"] == latest_key
    finally:
        with patch("tools.approval.unregister_gateway_notify"), patch(
            "tools.approval.enable_session_yolo"
        ), patch("tools.approval.disable_session_yolo"), patch(
            "tools.approval.is_session_yolo_enabled", return_value=False
        ), patch("tools.approval.register_gateway_notify"):
            release_result = server._release_live_micro_control(target)
    assert release_result
    assert session["session_key"] == latest_key
    assert session["session_generation"] == generation_before + 1
    assert session.get("_session_control_inflight") is None
    assert session.get("_deferred_compression_rotation") is None


def test_deferred_reanchor_failure_retains_marker_for_next_safe_trigger(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "retry-old"
    new_key = "retry-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-retry", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_transfer_active_session_slot", lambda *_a, **_k: True)
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    target, claim_error = server._claim_live_micro_control("sid-retry", session)
    assert claim_error is None
    assert target is not None
    agent.session_id = new_key
    assert server._sync_session_key_after_compress("sid-retry", session) == "deferred"
    real_sync = server._sync_session_key_after_compress

    def fail_sync(*_args, **_kwargs):
        raise RuntimeError("injected re-anchor failure")

    monkeypatch.setattr(server, "_sync_session_key_after_compress", fail_sync)
    failed = server._release_live_micro_control(target)
    assert not failed
    assert "retry" in failed.warning.lower()
    assert "injected re-anchor failure" in failed.warning
    assert session.get("_session_control_inflight") is None
    assert session.get("_deferred_compression_rotation") is not None
    assert session["session_key"] == old_key

    monkeypatch.setattr(server, "_sync_session_key_after_compress", real_sync)
    with patch("tools.approval.unregister_gateway_notify"), patch(
        "tools.approval.enable_session_yolo"
    ), patch("tools.approval.disable_session_yolo"), patch(
        "tools.approval.is_session_yolo_enabled", return_value=False
    ), patch("tools.approval.register_gateway_notify"):
        assert real_sync("sid-retry", session) == "applied"
    assert session["session_key"] == new_key
    assert session.get("_deferred_compression_rotation") is None


def test_release_close_barrier_does_not_touch_detached_target_or_new_lease(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "close-barrier-old"
    new_key = "close-barrier-new"
    old_agent = LiveAgent(db, old_key)
    old_session = _session(server, db, "sid-close-barrier", old_key, old_agent)
    db.create_session(new_key, source="tui", model="test-model")
    target, claim_error = server._claim_live_micro_control("sid-close-barrier", old_session)
    assert claim_error is None
    assert target is not None
    new_agent = LiveAgent(db, new_key)
    new_session = _session(server, db, "sid-close-barrier-new", new_key, new_agent)
    try:
        with server._sessions_lock:
            server._sessions.pop("sid-close-barrier", None)
            server._sessions["sid-close-barrier"] = new_session
        result = server._release_live_micro_control(target)
        assert not result
        assert "changed" in result.warning.lower()
        assert old_session.get("_session_control_inflight") is None
        assert new_session.get("_session_control_inflight") is None
        assert server._sessions["sid-close-barrier"] is new_session
    finally:
        with server._sessions_lock:
            if server._sessions.get("sid-close-barrier") is new_session:
                server._sessions.pop("sid-close-barrier", None)


def test_release_reanchor_leaves_history_and_registry_locks_available(
    gateway, monkeypatch
):
    server, db, _home = gateway
    old_key = "lock-race-old"
    new_key = "lock-race-new"
    agent = LiveAgent(db, old_key)
    session = _session(server, db, "sid-lock-race", old_key, agent)
    db.create_session(new_key, source="tui", model="test-model")
    monkeypatch.setattr(server, "_restart_slash_worker", lambda *_a, **_k: None)
    target, claim_error = server._claim_live_micro_control("sid-lock-race", session)
    assert claim_error is None
    assert target is not None
    agent.session_id = new_key
    assert server._sync_session_key_after_compress("sid-lock-race", session) == "deferred"
    transfer_entered = threading.Event()
    transfer_release = threading.Event()

    def blocked_transfer(*_args, **_kwargs):
        transfer_entered.set()
        assert transfer_release.wait(2), "blocked re-anchor was not released"
        return True

    monkeypatch.setattr(server, "_transfer_active_session_slot", blocked_transfer)
    release_result: dict = {}

    def release_target() -> None:
        with patch("tools.approval.unregister_gateway_notify"), patch(
            "tools.approval.enable_session_yolo"
        ), patch("tools.approval.disable_session_yolo"), patch(
            "tools.approval.is_session_yolo_enabled", return_value=False
        ), patch("tools.approval.register_gateway_notify"):
            release_result["value"] = server._release_live_micro_control(target)

    release_thread = threading.Thread(target=release_target)
    release_thread.start()
    assert transfer_entered.wait(2)
    history_acquired = threading.Event()
    registry_acquired = threading.Event()

    def acquire_history() -> None:
        with session["history_lock"]:
            history_acquired.set()

    def acquire_registry() -> None:
        with server._sessions_lock:
            registry_acquired.set()

    history_thread = threading.Thread(target=acquire_history)
    registry_thread = threading.Thread(target=acquire_registry)
    history_thread.start()
    registry_thread.start()
    try:
        assert history_acquired.wait(2), "history_lock remained held during re-anchor I/O"
        assert registry_acquired.wait(2), "_sessions_lock remained held during re-anchor I/O"
    finally:
        transfer_release.set()
        release_thread.join(timeout=5)
        history_thread.join(timeout=5)
        registry_thread.join(timeout=5)
    assert not release_thread.is_alive()
    assert not history_thread.is_alive()
    assert not registry_thread.is_alive()
    assert release_result["value"]


def test_old_micro_finally_cannot_clear_replacement_session_lease(gateway, monkeypatch):
    server, db, _home = gateway
    old_key = "aba-old"
    new_key = "aba-new"
    old_agent = LiveAgent(db, old_key)
    old_session = _session(server, db, "sid-aba", old_key, old_agent)
    thread, release, first = _start_blocked_micro(server, db, monkeypatch, sid="sid-aba")
    new_target = None
    try:
        new_agent = LiveAgent(db, new_key)
        db.create_session(new_key, source="tui", model="test-model")
        new_session = {
            "session_key": new_key,
            "agent": new_agent,
            "history": [],
            "history_lock": threading.Lock(),
            "history_version": 0,
            "session_generation": 1,
            "running": False,
            "profile_home": None,
            "slash_worker": None,
            "model": "test-model",
        }
        with server._sessions_lock:
            server._sessions["sid-aba"] = new_session
        new_target, claim_error = server._claim_live_micro_control("sid-aba", new_session)
        assert claim_error is None
        assert new_target is not None
        new_token = new_session["_session_control_inflight"]
        assert new_token is new_target.token

        release.set()
        thread.join(timeout=5)
        assert not thread.is_alive()
        assert "result" in first["response"]
        output = first["response"]["result"]["output"]
        assert isinstance(output, str)
        assert "live session changed" in output
        assert old_session.get("_session_control_inflight") is None
        assert new_session.get("_session_control_inflight") is new_token
        assert db.session_micro_compact_override(db.get_session(old_key)) is True
        assert db.session_micro_compact_override(db.get_session(new_key)) is None
        assert old_agent.context_compressor.calls == [True]
        assert new_agent.context_compressor.calls == []
    finally:
        release.set()
        thread.join(timeout=5)
        if new_target is not None:
            server._release_live_micro_control(new_target)


def test_blocked_micro_on_one_session_does_not_block_other_session_or_status(
    gateway, monkeypatch
):
    server, db, _home = gateway
    agent_a = LiveAgent(db, "isolated-a")
    agent_b = LiveAgent(db, "isolated-b")
    _session(server, db, "sid-isolated-a", "isolated-a", agent_a)
    _session(server, db, "sid-isolated-b", "isolated-b", agent_b)
    thread, release, first = _start_blocked_micro(
        server,
        db,
        monkeypatch,
        sid="sid-isolated-a",
        blocked_session_id="isolated-a",
    )
    other_result: dict = {}
    other_done = threading.Event()

    def run_other_micro() -> None:
        other_result["response"] = _call(
            server, "slash.exec", session_id="sid-isolated-b", command="micro on"
        )
        other_done.set()

    other_thread = threading.Thread(target=run_other_micro)
    other_thread.start()
    try:
        assert other_done.wait(2), "session B was blocked by session A's micro I/O"
        assert "result" in other_result["response"]
        assert db.session_micro_compact_override(db.get_session("isolated-b")) is True
        status = _call(server, "session.status", session_id="sid-isolated-b")
        assert "result" in status
    finally:
        _join_blocked_micro(thread, release)
        other_thread.join(timeout=5)
    assert "result" in first["response"]
    assert not other_thread.is_alive()
