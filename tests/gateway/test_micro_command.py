"""Focused gateway coverage for the DB-first ``/micro`` command."""

from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from gateway.config import Platform
from gateway.platforms.base import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource
from hermes_cli.commands import resolve_command
from hermes_state import SessionDB


class _Store:
    def __init__(self, entries: dict[tuple, str]):
        self.entries = entries
        self.calls: list[SessionSource] = []
        self._store = None

    async def get_or_create_session(self, source):
        self.calls.append(source)
        return SimpleNamespace(session_id=self.entries[self._key(source)])

    @staticmethod
    def _key(source: SessionSource) -> tuple:
        return (
            source.platform.value if source.platform else None,
            source.user_id,
            source.chat_id,
            source.thread_id,
        )


class _Engine:
    def __init__(self, db: SessionDB, session_id: str):
        self.db = db
        self.session_id = session_id
        self.calls: list[tuple[bool, bool | None]] = []

    def set_micro_compact_enabled(self, enabled: bool) -> None:
        row = self.db.get_session(self.session_id)
        override = self.db.session_micro_compact_override(row)
        self.calls.append((enabled, override))


class _Agent:
    def __init__(self, db: SessionDB, session_id: str, *, supported: bool = True):
        self.session_id = session_id
        self._session_db = db
        self._session_init_model_config = {"keep": "live"}
        self._ensure_calls = 0
        self.context_compressor = (
            _Engine(db, session_id) if supported else object()
        )

    def _ensure_db_session(self):
        self._ensure_calls += 1


def _source(chat_id: str = "chat-a", *, thread_id: str | None = None) -> SessionSource:
    return SessionSource(
        platform=Platform.TELEGRAM,
        user_id="user-a",
        chat_id=chat_id,
        chat_type="dm",
        thread_id=thread_id,
    )


def _event(source: SessionSource, args: str) -> MessageEvent:
    text = "/micro" if not args else f"/micro {args}"
    return MessageEvent(
        text=text,
        message_type=MessageType.TEXT,
        source=source,
        message_id=f"message-{source.chat_id}",
        internal=True,
    )


def _runner(
    db: SessionDB,
    entries: dict[tuple, str],
    *,
    route_keys: dict[str, str] | None = None,
):
    runner = object.__new__(GatewayRunner)
    runner.session_store = SimpleNamespace()
    runner._async_session_store = _Store(entries)
    runner._async_session_store._store = runner.session_store
    runner._session_db = SimpleNamespace(_db=db)
    runner._agent_cache = {}
    runner._agent_cache_lock = threading.RLock()
    route_keys = route_keys or {}
    runner._session_key_for_source = lambda source: route_keys.get(
        source.chat_id, f"route:{source.chat_id}"
    )
    return runner


@pytest.fixture()
def db(tmp_path):
    value = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield value
    finally:
        value.close()


@pytest.fixture(autouse=True)
def readonly_micro_config(monkeypatch):
    monkeypatch.setattr(
        "hermes_cli.micro_command.load_config_readonly",
        lambda: {"compression": {"micro_compact": False}},
    )


def _entry_map(*pairs: tuple[SessionSource, str]) -> dict[tuple, str]:
    return {_Store._key(source): session_id for source, session_id in pairs}


def _override(db: SessionDB, session_id: str):
    row = db.get_session(session_id)
    return None if row is None else db.session_micro_compact_override(row)


@pytest.mark.asyncio
async def test_cold_on_status_inherit_creates_only_canonical_row(db):
    source = _source()
    session_id = "cold-session"
    runner = _runner(db, _entry_map((source, session_id)))

    status = await runner._handle_micro_command(_event(source, "status"))
    assert "global (inherited)" in status
    assert db.get_session(session_id) is None

    saved = await runner._handle_micro_command(_event(source, "on"))
    assert "Micro-compaction override saved: ON" in saved
    assert "will apply when this session's agent starts or resumes" in saved
    row = db.get_session(session_id)
    assert row is not None
    assert row["source"] == "telegram"
    assert row["user_id"] == "user-a"
    assert row["chat_id"] == "chat-a"
    assert row["chat_type"] == "dm"
    assert _override(db, session_id) is True
    assert runner._agent_cache == {}

    inherited = await runner._handle_micro_command(_event(source, "inherit"))
    assert "global (inherited)" in inherited
    assert _override(db, session_id) is None
    assert runner._agent_cache == {}


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_shape", ["plain", "tuple"])
async def test_live_off_on_uses_exact_id_cache_shape_and_engine(db, cached_shape):
    source = _source()
    session_id = "live-session"
    db.create_session(
        session_id,
        source="telegram",
        user_id="user-a",
        chat_id="chat-a",
        chat_type="dm",
        model_config={"keep": "durable"},
    )
    runner = _runner(db, _entry_map((source, session_id)))
    agent = _Agent(db, session_id)
    route_key = runner._session_key_for_source(source)
    runner._agent_cache[route_key] = (
        agent if cached_shape == "plain" else (agent, "runtime-metadata")
    )

    off_result = await runner._handle_micro_command(_event(source, "off"))
    assert "Micro-compaction override saved: OFF" in off_result
    assert _override(db, session_id) is False
    assert agent.context_compressor.calls == [(False, False)]

    result = await runner._handle_micro_command(_event(source, "on"))

    assert "Micro-compaction: ON" in result
    assert agent._ensure_calls == 2
    assert _override(db, session_id) is True
    assert agent.context_compressor.calls == [(False, False), (True, True)]


@pytest.mark.asyncio
@pytest.mark.parametrize("cached_session_id", ["other-session", ""])
async def test_cached_identity_mismatch_or_empty_refuses_without_mutation(
    db, cached_session_id
):
    source = _source()
    session_id = "durable-session"
    db.create_session(
        session_id,
        source="telegram",
        user_id="user-a",
        chat_id="chat-a",
        chat_type="dm",
        model_config={"keep": "original"},
    )
    runner = _runner(db, _entry_map((source, session_id)))
    agent = _Agent(db, cached_session_id)
    runner._agent_cache[runner._session_key_for_source(source)] = agent

    result = await runner._handle_micro_command(_event(source, "on"))

    assert "refusing to mutate" in result
    assert _override(db, session_id) is None
    assert db.get_session(session_id)["model_config"] == '{"keep": "original"}'
    assert agent._ensure_calls == 0
    assert agent.context_compressor.calls == []


@pytest.mark.asyncio
async def test_invalid_and_bare_commands_do_not_mutate_row_or_engine(db):
    source = _source()
    session_id = "invalid-session"
    db.create_session(
        session_id,
        source="telegram",
        user_id="user-a",
        chat_id="chat-a",
        chat_type="dm",
        model_config={"keep": "original"},
    )
    runner = _runner(db, _entry_map((source, session_id)))
    agent = _Agent(db, session_id)
    runner._agent_cache[runner._session_key_for_source(source)] = agent

    for args in ("", "maybe", "on off", "status extra"):
        result = await runner._handle_micro_command(_event(source, args))
        assert "Usage: /micro on|off|inherit|status" in result

    assert _override(db, session_id) is None
    assert db.get_session(session_id)["model_config"] == '{"keep": "original"}'
    assert agent._ensure_calls == 0
    assert agent.context_compressor.calls == []


@pytest.mark.asyncio
async def test_plugin_without_runtime_support_warns_after_durable_save(db):
    source = _source()
    session_id = "unsupported-session"
    db.create_session(
        session_id,
        source="telegram",
        user_id="user-a",
        chat_id="chat-a",
        chat_type="dm",
    )
    runner = _runner(db, _entry_map((source, session_id)))
    agent = _Agent(db, session_id, supported=False)
    runner._agent_cache[runner._session_key_for_source(source)] = agent

    result = await runner._handle_micro_command(_event(source, "on"))

    assert _override(db, session_id) is True
    assert "does not support switching micro-compaction" in result
    assert agent.micro_compact_enabled is True
    assert agent.micro_compact_runtime_supported is False


@pytest.mark.asyncio
async def test_two_routing_lanes_mutate_only_their_durable_rows(db):
    source_a = _source("chat-a")
    source_b = _source("chat-b")
    session_a = "durable-a"
    session_b = "durable-b"
    for source, session_id in ((source_a, session_a), (source_b, session_b)):
        db.create_session(
            session_id,
            source="telegram",
            user_id="user-a",
            chat_id=source.chat_id,
            chat_type="dm",
        )
    runner = _runner(
        db,
        _entry_map((source_a, session_a), (source_b, session_b)),
        route_keys={"chat-a": "lane-a", "chat-b": "lane-b"},
    )
    agent_a = _Agent(db, session_a)
    agent_b = _Agent(db, session_b)
    runner._agent_cache["lane-a"] = agent_a
    runner._agent_cache["lane-b"] = (agent_b, "runtime-metadata")

    await runner._handle_micro_command(_event(source_a, "on"))
    await runner._handle_micro_command(_event(source_b, "off"))

    assert _override(db, session_a) is True
    assert _override(db, session_b) is False
    assert agent_a.context_compressor.calls == [(True, True)]
    assert agent_b.context_compressor.calls == [(False, False)]


@pytest.mark.asyncio
async def test_gateway_normal_typed_dispatch_reaches_micro_handler(db):
    source = _source()
    session_id = "dispatch-session"
    runner = _runner(db, _entry_map((source, session_id)))
    runner._peek_session_state = lambda _key: None
    runner._is_session_running = lambda _key: False
    runner._check_slash_access = lambda *_args: None
    runner.hooks = SimpleNamespace(emit_collect=AsyncMock(return_value=[]))

    result = await runner._handle_message(_event(source, "on"))

    assert "Micro-compaction override saved: ON" in result
    assert _override(db, session_id) is True


@pytest.mark.asyncio
async def test_busy_registry_dispatch_rejects_micro_without_calling_handler(db):
    source = _source()
    session_id = "busy-session"
    db.create_session(
        session_id,
        source="telegram",
        user_id="user-a",
        chat_id="chat-a",
        chat_type="dm",
    )
    runner = object.__new__(GatewayRunner)
    runner._handle_micro_command = AsyncMock(return_value="must not run")
    event = _event(source, "on")
    command = resolve_command("micro")

    result = await runner._dispatch_busy_slash_command(
        event,
        command,
        "route:chat-a",
        source,
    )

    assert result == (
        "⏳ Agent is running — `/micro` can't run mid-turn. "
        "Wait for the current response or `/stop` first."
    )
    runner._handle_micro_command.assert_not_awaited()
    assert _override(db, session_id) is None


@pytest.mark.asyncio
async def test_handler_uses_bound_runner_db_without_constructing_default_sessiondb(
    db, monkeypatch
):
    source = _source()
    session_id = "profile-bound-session"
    runner = _runner(db, _entry_map((source, session_id)))

    with patch("hermes_state.SessionDB", side_effect=AssertionError("default DB")):
        result = await runner._handle_micro_command(_event(source, "on"))

    assert "Micro-compaction override saved: ON" in result
    assert _override(db, session_id) is True


@pytest.mark.asyncio
async def test_handler_returns_stable_error_without_session_db(db):
    source = _source()
    session_id = "no-db-session"
    runner = _runner(db, _entry_map((source, session_id)))
    runner._session_db = None

    result = await runner._handle_micro_command(_event(source, "on"))

    assert result == "Micro-compaction error: session database is not available"
    assert db.get_session(session_id) is None


@pytest.mark.asyncio
async def test_handler_returns_stable_error_for_empty_durable_id(db):
    source = _source()
    runner = _runner(db, _entry_map((source, "")))

    result = await runner._handle_micro_command(_event(source, "on"))

    assert result == "Micro-compaction error: a non-empty exact session ID is required"
    assert db.get_session("") is None
