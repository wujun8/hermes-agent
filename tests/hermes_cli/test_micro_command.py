"""Focused tests for the DB-first classic /micro command service."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from hermes_cli.cli_commands_mixin import CLICommandsMixin
from hermes_cli.micro_command import (
    execute_micro_command,
    hydrate_micro_compact_policy_for_session,
)
from hermes_cli.micro_compaction import MICRO_COMPACT_OVERRIDE_KEY
from hermes_state import SessionDB


class _Engine:
    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self.calls: list[bool] = []

    def set_micro_compact_enabled(self, enabled: bool) -> None:
        self.calls.append(enabled)
        self.enabled = enabled


@pytest.fixture()
def db(tmp_path):
    value = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield value
    finally:
        value.close()


def _config(db: SessionDB, session_id: str) -> dict:
    row = db.get_session(session_id)
    assert row is not None
    raw = row["model_config"]
    return {} if raw in (None, "") else json.loads(raw)


def _agent(session_id: str, *, engine=None, model_config=None):
    return SimpleNamespace(
        session_id=session_id,
        context_compressor=engine if engine is not None else _Engine(),
        _session_init_model_config=dict(model_config or {"keep": "live"}),
    )


def _run(db, agent, raw_args, *, global_value=False, ensure_session=None):
    return execute_micro_command(
        agent=agent,
        session_db=db,
        session_id=agent.session_id,
        raw_args=raw_args,
        ensure_session=ensure_session or (lambda: None),
        config_loader=lambda: {"compression": {"micro_compact": global_value}},
    )


def _run_cold(
    db,
    session_id,
    raw_args,
    *,
    global_value=False,
    ensure_session=None,
):
    return execute_micro_command(
        agent=None,
        session_db=db,
        session_id=session_id,
        raw_args=raw_args,
        ensure_session=ensure_session,
        config_loader=lambda: {"compression": {"micro_compact": global_value}},
    )


def test_on_off_inherit_are_db_first_and_preserve_unrelated_config(db):
    session_id = "micro-service"
    db.create_session(
        session_id,
        source="cli",
        model_config={"yolo_mode": True, "nested": {"x": 1}},
    )
    agent = _agent(session_id, model_config={"keep": "live"})

    result = _run(db, agent, "on")
    assert "Micro-compaction: ON" in result
    assert "Source: session" in result
    assert _config(db, session_id)[MICRO_COMPACT_OVERRIDE_KEY] is True
    assert _config(db, session_id)["yolo_mode"] is True
    assert agent.context_compressor.enabled is True

    result = _run(db, agent, "off")
    assert "Micro-compaction: OFF" in result
    assert _config(db, session_id)[MICRO_COMPACT_OVERRIDE_KEY] is False
    assert agent.context_compressor.enabled is False

    result = _run(db, agent, "inherit", global_value=True)
    assert "Micro-compaction: ON" in result
    assert "global (inherited)" in result
    assert MICRO_COMPACT_OVERRIDE_KEY not in _config(db, session_id)
    assert agent.context_compressor.enabled is True
    assert agent._session_init_model_config == {"keep": "live"}


def test_status_is_read_only_and_missing_row_means_inherit(db):
    session_id = "micro-status"
    db.create_session(session_id, source="cli", model_config={MICRO_COMPACT_OVERRIDE_KEY: True})
    agent = _agent(session_id)
    ensure_calls: list[str] = []

    result = _run(
        db,
        agent,
        "status",
        global_value=False,
        ensure_session=lambda: ensure_calls.append("ensure"),
    )
    assert "Micro-compaction: ON" in result
    assert ensure_calls == []
    assert agent.context_compressor.calls == []

    missing = _agent("missing-status")
    result = _run(
        db,
        missing,
        "status",
        global_value=True,
        ensure_session=lambda: ensure_calls.append("ensure"),
    )
    assert "Micro-compaction: ON" in result
    assert "global (inherited)" in result
    assert ensure_calls == []
    assert missing.context_compressor.calls == []


def test_status_without_session_id_or_db_reads_global_only():
    config_calls: list[str] = []

    def config_loader():
        config_calls.append("config")
        return {"compression": {"micro_compact": True}}

    class _UnexpectedDBRead:
        def get_session(self, _session_id):
            raise AssertionError("status must not read a DB without an exact ID")

    for session_db, session_id in ((_UnexpectedDBRead(), ""), (None, "existing-id")):
        result = execute_micro_command(
            agent=None,
            session_db=session_db,
            session_id=session_id,
            raw_args="status",
            config_loader=config_loader,
        )
        assert "Micro-compaction: ON" in result
        assert "global (inherited)" in result

    assert config_calls == ["config", "config"]


def test_status_config_failure_is_reported_without_db_access():
    class _UnexpectedDBRead:
        def get_session(self, _session_id):
            raise AssertionError("config failure must happen before any DB read")

    result = execute_micro_command(
        agent=None,
        session_db=_UnexpectedDBRead(),
        session_id="",
        raw_args="status",
        config_loader=lambda: (_ for _ in ()).throw(RuntimeError("config unavailable")),
    )

    assert "could not read global configuration" in result
    assert "config unavailable" in result


def test_mutations_still_require_exact_id_and_database():
    ensure_calls: list[str] = []

    missing_id = execute_micro_command(
        agent=None,
        session_db=object(),
        session_id="",
        raw_args="on",
        ensure_session=lambda: ensure_calls.append("ensure"),
        config_loader=lambda: {"compression": {"micro_compact": False}},
    )
    missing_db = execute_micro_command(
        agent=None,
        session_db=None,
        session_id="existing-id",
        raw_args="off",
        ensure_session=lambda: ensure_calls.append("ensure"),
        config_loader=lambda: {"compression": {"micro_compact": False}},
    )

    assert missing_id == "Micro-compaction error: a non-empty exact session ID is required"
    assert missing_db == "Micro-compaction error: session database is not available"
    assert ensure_calls == []


def test_invalid_or_bare_command_has_no_db_or_runtime_mutation(db):
    session_id = "micro-invalid"
    db.create_session(session_id, source="cli", model_config={"keep": 1})
    agent = _agent(session_id)
    ensure_calls: list[str] = []

    for raw in ("", "maybe", "on off", "status extra"):
        result = _run(
            db,
            agent,
            raw,
            ensure_session=lambda: ensure_calls.append("ensure"),
        )
        assert "Usage: /micro on|off|inherit|status" in result
    assert _config(db, session_id) == {"keep": 1}
    assert ensure_calls == []
    assert agent.context_compressor.calls == []


def test_config_failure_and_db_setter_failure_do_not_apply_live_policy(db, monkeypatch):
    session_id = "micro-failures"
    db.create_session(session_id, source="cli", model_config={"keep": 1})
    agent = _agent(session_id)
    agent.micro_compact_enabled = False
    agent.context_compressor.enabled = False

    result = execute_micro_command(
        agent=agent,
        session_db=db,
        session_id=session_id,
        raw_args="on",
        config_loader=lambda: (_ for _ in ()).throw(RuntimeError("config unavailable")),
    )
    assert "config" in result.lower()
    assert _config(db, session_id) == {"keep": 1}
    assert agent.context_compressor.calls == []

    def fail_setter(*args, **kwargs):
        raise RuntimeError("database locked")

    monkeypatch.setattr(db, "set_session_micro_compact_override", fail_setter)
    result = _run(db, agent, "on")
    assert "database" in result.lower() or "locked" in result.lower()
    assert agent.context_compressor.calls == []
    assert agent.context_compressor.enabled is False


def test_cold_on_off_inherit_persist_without_live_policy_application(db):
    session_id = "micro-cold-policy"
    db.create_session(
        session_id,
        source="gateway",
        model_config={"keep": "cold"},
    )
    ensure_calls: list[str] = []

    def ensure_session():
        ensure_calls.append("ensure")

    with patch("hermes_cli.micro_command._apply_live_policy") as apply_live:
        result = _run_cold(
            db,
            session_id,
            "on",
            ensure_session=ensure_session,
        )
        assert "Micro-compaction override saved: ON" in result
        assert "Micro-compaction: ON" in result
        assert db.session_micro_compact_override(db.get_session(session_id)) is True

        result = _run_cold(
            db,
            session_id,
            "off",
            ensure_session=ensure_session,
        )
        assert "Micro-compaction override saved: OFF" in result
        assert db.session_micro_compact_override(db.get_session(session_id)) is False

        result = _run_cold(
            db,
            session_id,
            "inherit",
            global_value=True,
            ensure_session=ensure_session,
        )
        assert "Micro-compaction override saved: ON" in result
        assert "global (inherited)" in result
        assert db.session_micro_compact_override(db.get_session(session_id)) is None

    assert ensure_calls == ["ensure", "ensure", "ensure"]
    apply_live.assert_not_called()
    assert _config(db, session_id) == {"keep": "cold"}


def test_cold_status_is_read_only_and_does_not_need_ensure(db):
    session_id = "micro-cold-status"
    db.create_session(
        session_id,
        source="gateway",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: True},
    )
    ensure_session = MagicMock()

    with patch("hermes_cli.micro_command._apply_live_policy") as apply_live:
        result = _run_cold(
            db,
            session_id,
            "status",
            ensure_session=ensure_session,
        )

    assert "Micro-compaction: ON" in result
    assert "Source: session" in result
    ensure_session.assert_not_called()
    apply_live.assert_not_called()
    assert _config(db, session_id) == {MICRO_COMPACT_OVERRIDE_KEY: True}


def test_cold_invalid_command_and_missing_ensure_do_not_mutate(db):
    session_id = "micro-cold-invalid"
    db.create_session(session_id, source="gateway", model_config={"keep": 1})
    ensure_session = MagicMock()

    with patch("hermes_cli.micro_command._apply_live_policy") as apply_live:
        result = _run_cold(
            db,
            session_id,
            "maybe",
            ensure_session=ensure_session,
        )
        assert "Usage: /micro on|off|inherit|status" in result
        ensure_session.assert_not_called()

        result = _run_cold(db, session_id, "on")
        assert "canonical session ensure callback" in result

        result = _run_cold(
            db,
            " ",
            "on",
            ensure_session=ensure_session,
        )
        assert "non-empty exact session ID" in result

    apply_live.assert_not_called()
    assert _config(db, session_id) == {"keep": 1}


def test_cold_database_failures_do_not_apply_live_policy(db, monkeypatch):
    session_id = "micro-cold-db-failure"
    db.create_session(session_id, source="gateway", model_config={"keep": 1})
    ensure_session = MagicMock()

    def fail_setter(*_args, **_kwargs):
        raise RuntimeError("database locked")

    monkeypatch.setattr(db, "set_session_micro_compact_override", fail_setter)
    with patch("hermes_cli.micro_command._apply_live_policy") as apply_live:
        result = _run_cold(
            db,
            session_id,
            "on",
            ensure_session=ensure_session,
        )

    assert "database" in result.lower() or "locked" in result.lower()
    ensure_session.assert_called_once()
    apply_live.assert_not_called()
    assert _config(db, session_id) == {"keep": 1}


def test_cold_config_and_ensure_failures_do_not_persist_or_apply(db):
    session_id = "micro-cold-early-failure"
    db.create_session(session_id, source="gateway", model_config={"keep": 1})
    ensure_session = MagicMock(side_effect=RuntimeError("ensure failed"))

    def fail_config():
        raise RuntimeError("config unavailable")

    with patch("hermes_cli.micro_command._apply_live_policy") as apply_live:
        result = execute_micro_command(
            agent=None,
            session_db=db,
            session_id=session_id,
            raw_args="on",
            ensure_session=MagicMock(),
            config_loader=fail_config,
        )
        assert "config" in result.lower()

        result = _run_cold(
            db,
            session_id,
            "on",
            ensure_session=ensure_session,
        )
        assert "ensure failed" in result

    ensure_session.assert_called_once()
    apply_live.assert_not_called()
    assert _config(db, session_id) == {"keep": 1}


def test_cold_policy_row_hydrates_when_a_live_agent_starts(db):
    session_id = "micro-cold-hydration"
    db.create_session(session_id, source="gateway", model_config={"keep": "row"})

    result = _run_cold(db, session_id, "on", ensure_session=lambda: None)
    assert "Micro-compaction: ON" in result

    live_agent = _agent(session_id)
    assert live_agent.context_compressor.calls == []
    supported = hydrate_micro_compact_policy_for_session(
        agent=live_agent,
        session_db=db,
        session_id=session_id,
        config_loader=lambda: {"compression": {"micro_compact": False}},
    )

    assert supported is True
    assert live_agent.micro_compact_override is True
    assert live_agent.micro_compact_enabled is True
    assert live_agent.micro_compact_source == "session"
    assert live_agent.context_compressor.calls == [True]
    assert live_agent._session_init_model_config == {"keep": "live", MICRO_COMPACT_OVERRIDE_KEY: True}


def test_missing_row_mutation_calls_canonical_ensure_then_persists(db):
    session_id = "micro-lazy"
    agent = _agent(session_id)
    ensure_calls: list[str] = []

    def ensure_session():
        ensure_calls.append("ensure")
        db.create_session(session_id, source="cli", model_config={"keep": 1})

    result = _run(db, agent, "on", ensure_session=ensure_session)
    assert ensure_calls == ["ensure"]
    assert db.session_micro_compact_override(db.get_session(session_id)) is True
    assert "Micro-compaction: ON" in result


def test_plugin_without_runtime_setter_gets_saved_but_explicit_warning(db):
    session_id = "micro-plugin"
    db.create_session(session_id, source="cli", model_config={"keep": 1})
    agent = _agent(session_id, engine=SimpleNamespace())

    result = _run(db, agent, "on")
    assert db.session_micro_compact_override(db.get_session(session_id)) is True
    assert "Micro-compaction: ON" in result
    assert "does not support" in result.lower()
    assert "runtime" in result.lower()


def test_session_id_is_exact_and_must_not_fall_back_to_other_row(db):
    db.create_session("actual", source="cli", model_config={"keep": 1})
    agent = _agent(" ")
    result = execute_micro_command(
        agent=agent,
        session_db=db,
        session_id=" ",
        raw_args="on",
        config_loader=lambda: {"compression": {"micro_compact": True}},
        ensure_session=lambda: pytest.fail("must not ensure an empty session id"),
    )
    assert "session" in result.lower()
    assert db.session_micro_compact_override(db.get_session("actual")) is None


class _BoundaryAgent:
    """Small live-agent seam that preserves reset_session_state's contract."""

    def __init__(self, session_id: str, override: bool, global_value):
        self.session_id = session_id
        self.session_start = datetime.now()
        self.context_compressor = _Engine(enabled=override)
        self.context_compressor._micro_compact_cursor = 17
        self.context_compressor._micro_compact_rolling_summary = "rolling"
        self._session_init_model_config = {
            "keep": "live",
            MICRO_COMPACT_OVERRIDE_KEY: override,
        }
        self.micro_compact_override = override
        self.micro_compact_enabled = override
        self.micro_compact_source = "session"
        self._micro_compact_global_value = global_value
        self.micro_compact_runtime_supported = True
        self._session_db_created = False
        self._last_flushed_db_idx = 3
        self.reset_calls = 0
        self.flush_calls = []

    def reset_session_state(self):
        self.reset_calls += 1
        self._last_flushed_db_idx = 0
        # Deliberately do not touch policy or context-engine micro state.  This
        # mirrors the real reset contract; the boundary helper owns only the
        # policy transition.

    def _flush_messages_to_session_db(self, *args, **kwargs):
        self.flush_calls.append((args, kwargs))


def _make_boundary_cli(db, agent):
    import cli as cli_module

    cli = cli_module.HermesCLI.__new__(cli_module.HermesCLI)
    cli.session_id = agent.session_id
    cli.session_start = agent.session_start
    cli._session_db = db
    cli.agent = agent
    cli.conversation_history = []
    cli._pending_title = None
    cli._resumed = False
    cli.reasoning_config = {}
    cli.max_turns = 7
    cli.model = "stub/model"
    cli.provider = "stub"
    cli.requested_provider = "stub"
    cli.base_url = "http://stub"
    cli.api_key = ""
    cli.api_mode = "chat_completions"
    cli.service_tier = None
    cli._pending_one_turn_model_restore = None
    cli._notify_session_boundary = lambda *args, **kwargs: None
    cli._launch_session_boundary_memory_flush = lambda *args, **kwargs: None
    cli._discard_session_if_empty = MagicMock()
    cli._pending_resume_sessions = None
    cli.resume_display = "minimal"
    cli._display_resumed_history = MagicMock()
    cli._restore_session_cwd = lambda *_args, **_kwargs: None
    cli._restore_session_yolo = lambda *_args, **_kwargs: None
    return cli


def _cli_config():
    return {
        "agent": {"reasoning_effort": "", "service_tier": ""},
        "model": {"default": "stub/model", "provider": "stub"},
    }


@pytest.mark.parametrize(
    ("old_override", "global_value"),
    [(True, False), (False, True)],
)
def test_new_session_isolates_live_micro_policy_and_preserves_old_row(
    db, old_override, global_value
):
    import cli as cli_module

    old_session_id = f"new-old-{old_override}"
    db.create_session(
        old_session_id,
        source="cli",
        model="stub/model",
        model_config={
            "keep": "old",
            MICRO_COMPACT_OVERRIDE_KEY: old_override,
        },
    )
    agent = _BoundaryAgent(old_session_id, old_override, global_value)
    cli = _make_boundary_cli(db, agent)

    with (
        patch.object(cli_module, "CLI_CONFIG", _cli_config()),
        patch(
            "hermes_cli.micro_command.load_config_readonly",
            return_value={"compression": {"micro_compact": global_value}},
        ),
    ):
        cli.new_session(silent=True)

    old_row = db.get_session(old_session_id)
    new_row = db.get_session(cli.session_id)
    assert old_row is not None
    assert db.session_micro_compact_override(old_row) is old_override
    assert new_row is not None
    assert db.session_micro_compact_override(new_row) is None
    assert MICRO_COMPACT_OVERRIDE_KEY not in _config(db, cli.session_id)

    assert agent.reset_calls == 1
    assert agent.micro_compact_override is None
    assert agent.micro_compact_source == "global"
    assert agent.micro_compact_enabled is global_value
    assert agent.context_compressor.enabled is global_value
    assert agent._session_init_model_config == {"keep": "live"}
    assert agent.context_compressor._micro_compact_cursor == 17
    assert agent.context_compressor._micro_compact_rolling_summary == "rolling"
    cli._discard_session_if_empty.assert_not_called()


def test_new_session_config_failure_removes_old_live_override_using_cached_global(db):
    import cli as cli_module

    old_session_id = "new-config-failure-old"
    db.create_session(
        old_session_id,
        source="cli",
        model="stub/model",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: True},
    )
    agent = _BoundaryAgent(old_session_id, True, False)
    cli = _make_boundary_cli(db, agent)

    with (
        patch.object(cli_module, "CLI_CONFIG", _cli_config()),
        patch(
            "hermes_cli.micro_command.load_config_readonly",
            side_effect=RuntimeError("config unavailable"),
        ),
    ):
        cli.new_session(silent=True)

    assert db.session_micro_compact_override(db.get_session(old_session_id)) is True
    assert db.get_session(cli.session_id) is not None
    assert db.session_micro_compact_override(db.get_session(cli.session_id)) is None
    assert agent.micro_compact_override is None
    assert agent.micro_compact_source == "global"
    assert agent.micro_compact_enabled is False
    assert agent.context_compressor.enabled is False
    assert MICRO_COMPACT_OVERRIDE_KEY not in agent._session_init_model_config


def _resume_to(cli, target_id):
    with (
        patch("hermes_cli.main._resolve_session_by_name_or_id", return_value=None),
        patch("cli._cprint"),
    ):
        cli._handle_resume_command(f"/resume {target_id}")


def test_in_process_resume_hydrates_exact_target_and_clears_previous_override(db):
    import cli as cli_module

    db.create_session(
        "resume-a",
        source="cli",
        model="stub/model",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: True},
    )
    db.create_session(
        "resume-b",
        source="cli",
        model="stub/model",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: False},
    )
    db.create_session(
        "resume-c",
        source="cli",
        model="stub/model",
        model_config={"keep": "target"},
    )
    agent = _BoundaryAgent("resume-a", True, True)
    cli = _make_boundary_cli(db, agent)

    with patch(
        "hermes_cli.micro_command.load_config_readonly",
        return_value={"compression": {"micro_compact": True}},
    ):
        _resume_to(cli, "resume-b")
        assert cli.session_id == "resume-b"
        assert agent.micro_compact_override is False
        assert agent.micro_compact_source == "session"
        assert agent.micro_compact_enabled is False
        assert agent.context_compressor.enabled is False
        assert agent._session_init_model_config[MICRO_COMPACT_OVERRIDE_KEY] is False

        _resume_to(cli, "resume-c")

    assert cli.session_id == "resume-c"
    assert agent.micro_compact_override is None
    assert agent.micro_compact_source == "global"
    assert agent.micro_compact_enabled is True
    assert agent.context_compressor.enabled is True
    assert MICRO_COMPACT_OVERRIDE_KEY not in agent._session_init_model_config
    assert db.session_micro_compact_override(db.get_session("resume-a")) is True
    assert db.session_micro_compact_override(db.get_session("resume-b")) is False
    assert db.session_micro_compact_override(db.get_session("resume-c")) is None
    assert _config(db, "resume-c") == {"keep": "target"}


def test_resume_config_failure_still_hydrates_exact_policy_and_clears_inherit(
    db, caplog
):
    db.create_session(
        "resume-failure-a",
        source="cli",
        model="stub/model",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: True},
    )
    db.create_session(
        "resume-failure-b",
        source="cli",
        model="stub/model",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: False},
    )
    db.create_session(
        "resume-failure-c",
        source="cli",
        model="stub/model",
        model_config={"keep": "target"},
    )
    agent = _BoundaryAgent("resume-failure-a", True, True)
    cli = _make_boundary_cli(db, agent)

    with patch(
        "hermes_cli.micro_command.load_config_readonly",
        side_effect=RuntimeError("config unavailable"),
    ):
        _resume_to(cli, "resume-failure-b")
        assert agent.micro_compact_override is False
        assert agent.micro_compact_enabled is False
        assert agent._session_init_model_config[MICRO_COMPACT_OVERRIDE_KEY] is False

        _resume_to(cli, "resume-failure-c")

    assert agent.micro_compact_override is None
    assert agent.micro_compact_source == "global"
    assert agent.micro_compact_enabled is True
    assert agent.context_compressor.enabled is True
    assert MICRO_COMPACT_OVERRIDE_KEY not in agent._session_init_model_config
    assert "config" in caplog.text.lower()
    assert _config(db, "resume-failure-c") == {"keep": "target"}


def test_micro_handler_renders_service_result_without_provider_turn(db):
    session_id = "micro-handler"
    db.create_session(session_id, source="cli", model_config={"keep": 1})
    agent = _agent(session_id)
    cli = CLICommandsMixin.__new__(CLICommandsMixin)
    cli.agent = agent
    cli._session_db = db
    cli._console_print = MagicMock()

    with patch(
        "hermes_cli.micro_command.load_config_readonly",
        return_value={"compression": {"micro_compact": False}},
    ):
        cli._handle_micro_command("/micro status")

    cli._console_print.assert_called_once()
    assert "Micro-compaction: OFF" in cli._console_print.call_args.args[0]
    assert agent.context_compressor.calls == []
