"""Agent/runtime tests for durable micro-compaction policy application."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from agent.agent_init import (
    apply_micro_compact_policy,
    hydrate_micro_compact_policy,
    refresh_micro_compact_policy,
)
from agent.context_compressor import ContextCompressor
from agent.turn_finalizer import finalize_turn
from hermes_cli.micro_compaction import MICRO_COMPACT_OVERRIDE_KEY
from hermes_state import SessionDB


class _MutableEngine:
    def __init__(self):
        self.enabled = False
        self.setter_calls = []

    def set_micro_compact_enabled(self, enabled: bool) -> None:
        self.enabled = enabled
        self.setter_calls.append(enabled)


class _PluginWithoutMicroCapability:
    """A plugin engine must not be mutated through private attributes."""

    pass


def test_builtin_compressor_setter_is_runtime_only_and_preserves_all_state():
    compressor = ContextCompressor(
        model="test-model",
        threshold_percent=0.75,
        protect_first_n=1,
        protect_last_n=2,
        quiet_mode=True,
        config_context_length=40960,
        provider="test",
    )
    compressor._micro_compact_cursor = 17
    compressor._micro_compact_rolling_summary = "rolling"
    compressor._micro_compact_consecutive_failures = 2
    compressor._micro_compact_last_failure_cursor = 11
    compressor._micro_compact_passes = 4
    compressor._micro_compact_tokens_saved_total = 123
    compressor._micro_compact_turns_since_pass = 3
    compressor._previous_summary = "batch summary"
    compressor._last_summary_error = "last error"
    compressor._last_compression_telemetry = {"attempt": 1}
    compressor._active_compression_telemetry = {"active": True}
    compressor.last_prompt_tokens = 111
    compressor.last_completion_tokens = 222
    compressor.last_total_tokens = 333
    compressor.compression_count = 5
    compressor._session_id = "session"
    compressor._session_db = object()

    tracked = {
        name: getattr(compressor, name)
        for name in (
            "_micro_compact_cursor",
            "_micro_compact_rolling_summary",
            "_micro_compact_consecutive_failures",
            "_micro_compact_last_failure_cursor",
            "_micro_compact_passes",
            "_micro_compact_tokens_saved_total",
            "_micro_compact_turns_since_pass",
            "_previous_summary",
            "_last_summary_error",
            "_last_compression_telemetry",
            "_active_compression_telemetry",
            "last_prompt_tokens",
            "last_completion_tokens",
            "last_total_tokens",
            "compression_count",
            "_session_id",
            "_session_db",
        )
    }
    identity = id(compressor)

    compressor.set_micro_compact_enabled(True)
    assert compressor.get_micro_compact_enabled() is True
    assert id(compressor) == identity
    assert {name: getattr(compressor, name) for name in tracked} == tracked

    compressor.set_micro_compact_enabled(False)
    assert compressor.get_micro_compact_enabled() is False
    assert {name: getattr(compressor, name) for name in tracked} == tracked


def _policy_agent(engine, model_config=None):
    return SimpleNamespace(
        context_compressor=engine,
        _session_init_model_config=dict(model_config or {}),
    )


@pytest.fixture()
def db(tmp_path):
    session_db = SessionDB(db_path=tmp_path / "state.db")
    try:
        yield session_db
    finally:
        session_db.close()


@pytest.mark.parametrize(
    ("stored_override", "global_value", "expected_enabled"),
    [(True, False, True), (False, True, False)],
)
def test_hydrate_micro_compact_policy_resumes_stored_true_false(
    db: SessionDB, stored_override, global_value, expected_enabled
):
    session_id = f"resume-{stored_override}"
    db.create_session(
        session_id,
        source="cli",
        model_config={MICRO_COMPACT_OVERRIDE_KEY: stored_override},
    )
    engine = _MutableEngine()
    agent = _policy_agent(engine, {"unrelated": "preserved"})

    hydrate_micro_compact_policy(
        agent,
        session_db=db,
        session_id=session_id,
        global_value=global_value,
    )

    assert agent.micro_compact_override is stored_override
    assert agent.micro_compact_enabled is expected_enabled
    assert engine.enabled is expected_enabled
    assert agent._session_init_model_config[MICRO_COMPACT_OVERRIDE_KEY] is stored_override
    assert agent._session_init_model_config["unrelated"] == "preserved"


def test_inherited_policy_follows_global_without_durable_override(db: SessionDB):
    engine = _MutableEngine()
    agent = _policy_agent(engine, {"unrelated": True})

    hydrate_micro_compact_policy(
        agent,
        session_db=db,
        session_id="not-created-yet",
        global_value=False,
    )
    assert agent.micro_compact_override is None
    assert agent.micro_compact_enabled is False
    assert MICRO_COMPACT_OVERRIDE_KEY not in agent._session_init_model_config

    apply_micro_compact_policy(agent, session_override=None, global_value=True)
    assert agent.micro_compact_enabled is True
    assert agent.micro_compact_source == "global"
    assert MICRO_COMPACT_OVERRIDE_KEY not in agent._session_init_model_config


def test_runtime_policy_application_plugin_without_capability_is_safe():
    plugin = _PluginWithoutMicroCapability()
    agent = _policy_agent(plugin, {"unrelated": 1})

    supported = apply_micro_compact_policy(
        agent,
        session_override=True,
        global_value=False,
    )

    assert supported is False
    assert agent.micro_compact_override is True
    assert agent.micro_compact_enabled is True
    assert agent._session_init_model_config[MICRO_COMPACT_OVERRIDE_KEY] is True
    assert not hasattr(plugin, "_micro_compact_enabled")


class _Budget:
    used = 1
    max_total = 3
    remaining = 2


class _FinalizerMicroEngine:
    last_prompt_tokens = 0

    def __init__(self):
        self._micro_compact_enabled = False
        self.calls = 0
        self.setter_calls = []

    def set_micro_compact_enabled(self, enabled: bool) -> None:
        self._micro_compact_enabled = enabled
        self.setter_calls.append(enabled)

    def _micro_compact(self, messages):
        self.calls += 1
        return messages


class _FinalizerAgent:
    def __init__(self, override=None):
        self.max_iterations = 3
        self.iteration_budget = _Budget()
        self.context_compressor = _FinalizerMicroEngine()
        self.model = "stub/model"
        self.provider = "stub"
        self.base_url = "http://stub"
        self.session_id = "finalizer-session"
        self.platform = "cli"
        self.quiet_mode = True
        self._interrupt_requested = False
        self._interrupt_message = None
        self._tool_guardrail_halt_decision = None
        self._response_was_previewed = False
        self._stream_callback = None
        self._skill_nudge_interval = 0
        self._iters_since_skill = 0
        self.valid_tool_names = set()
        self._turn_preflight_display_snapshot = None
        self._turn_received_provider_response = False
        self._turn_failed_file_mutations = {}
        self.request_overrides = {}
        self._session_init_model_config = {}
        self.micro_compact_override = override
        self.micro_compact_enabled = False
        self.micro_compact_source = "global"
        self._micro_compact_global_value = False
        self._refresh_micro_compact_policy = lambda: refresh_micro_compact_policy(self)
        for attr in (
            "session_input_tokens",
            "session_output_tokens",
            "session_cache_read_tokens",
            "session_cache_write_tokens",
            "session_reasoning_tokens",
            "session_prompt_tokens",
            "session_completion_tokens",
            "session_total_tokens",
            "session_estimated_cost_usd",
        ):
            setattr(self, attr, 0)
        self.session_cost_status = "ok"
        self.session_cost_source = "stub"

    def _save_trajectory(self, *args, **kwargs):
        return None

    def _cleanup_task_resources(self, *args, **kwargs):
        return None

    def _drop_trailing_empty_response_scaffolding(self, *args, **kwargs):
        return None

    def _persist_session(self, *args, **kwargs):
        return None

    def _emit_status(self, *args, **kwargs):
        return None

    def _safe_print(self, *args, **kwargs):
        return None

    def _file_mutation_verifier_enabled(self):
        return False

    def _turn_completion_explainer_enabled(self):
        return False

    def _drain_pending_steer(self):
        return None

    def clear_interrupt(self):
        return None

    def _sync_external_memory_for_turn(self, **kwargs):
        return None


def _finalize_once(agent):
    return finalize_turn(
        agent,
        final_response="done",
        api_call_count=1,
        interrupted=False,
        failed=False,
        messages=[{"role": "user", "content": "hello"}],
        conversation_history=None,
        effective_task_id="task",
        turn_id="turn",
        user_message="hello",
        original_user_message="hello",
        _should_review_memory=False,
        _turn_exit_reason="text_response(done)",
    )


def test_turn_finalizer_refreshes_inherited_global_policy_each_completed_turn():
    agent = _FinalizerAgent(override=None)

    with patch(
        "agent.agent_init.load_config_readonly",
        return_value={"compression": {"micro_compact": True}},
    ):
        _finalize_once(agent)
    assert agent.micro_compact_enabled is True
    assert agent.context_compressor.calls == 1

    with patch(
        "agent.agent_init.load_config_readonly",
        return_value={"compression": {"micro_compact": False}},
    ):
        _finalize_once(agent)
    assert agent.micro_compact_enabled is False
    assert agent.context_compressor.calls == 1


def test_turn_finalizer_explicit_policy_does_not_follow_global_changes():
    agent = _FinalizerAgent(override=True)
    apply_micro_compact_policy(agent, session_override=True, global_value=False)

    with patch(
        "agent.agent_init.load_config_readonly",
        return_value={"compression": {"micro_compact": False}},
    ):
        _finalize_once(agent)

    assert agent.micro_compact_enabled is True
    assert agent.context_compressor.calls == 1


def test_turn_finalizer_global_config_failure_retains_last_effective_value(caplog):
    agent = _FinalizerAgent(override=None)
    apply_micro_compact_policy(agent, session_override=None, global_value=True)

    with patch(
        "agent.agent_init.load_config_readonly",
        side_effect=RuntimeError("config unavailable"),
    ):
        _finalize_once(agent)

    assert agent.micro_compact_enabled is True
    assert agent.context_compressor.calls == 1
    assert "micro-compaction" in caplog.text.lower()
