from __future__ import annotations

import threading
import time
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest

from agent.api_outage_recovery import (
    ApiOutageRecoveryConfig,
    ApiOutageRecoveryWaiter,
)
from run_agent import AIAgent


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, seconds):
        self.now += seconds


class FakeAgent:
    def __init__(self):
        self._interrupt_requested = False
        self.touches = []

    def _touch_activity(self, detail):
        self.touches.append(detail)


def _waiter(results, *, interval=600, timeout=60, clock=None):
    clock = clock or FakeClock()
    calls = []

    def runner(argv, **kwargs):
        calls.append((clock.now, argv, kwargs))
        result = results.pop(0)
        if isinstance(result, BaseException):
            raise result
        return SimpleNamespace(returncode=result, stdout="secret-output", stderr="secret-error")

    waiter = ApiOutageRecoveryWaiter(
        ApiOutageRecoveryConfig(True, "probe --ready", interval, timeout),
        clock=clock.monotonic,
        sleep=clock.sleep,
        runner=runner,
    )
    return waiter, calls, clock


def test_probe_fail_fail_success_is_interval_throttled_and_safe():
    waiter, calls, clock = _waiter([1, 1, 0])
    agent = FakeAgent()
    notices = []

    result = waiter.wait(
        agent,
        endpoint_key="custom|model|https://api.example/v1",
        on_parked=notices.append,
        on_recovered=notices.append,
    )

    assert result == "recovered"
    probe_times = [at for at, _argv, _kwargs in calls]
    assert probe_times[0] == 0.0
    assert all(later - earlier >= 600 for earlier, later in zip(probe_times, probe_times[1:]))
    assert all(argv == ["probe", "--ready"] for _at, argv, _kwargs in calls)
    assert all(kwargs["shell"] is False for _at, _argv, kwargs in calls)
    assert all(kwargs["timeout"] == 60 for _at, _argv, kwargs in calls)
    assert all(kwargs["capture_output"] is True for _at, _argv, kwargs in calls)
    assert len(notices) == 2
    assert "停放" in notices[0]
    assert "恢复" in notices[1]
    assert agent.touches


def test_false_positive_then_repark_waits_remaining_interval_before_next_probe():
    waiter, calls, clock = _waiter([0, 0])
    agent = FakeAgent()

    assert waiter.wait(agent, "same-endpoint", lambda _m: None, lambda _m: None) == "recovered"
    clock.now += 5
    assert waiter.wait(agent, "same-endpoint", lambda _m: None, lambda _m: None) == "recovered"

    probe_times = [at for at, _argv, _kwargs in calls]
    assert probe_times[0] == 0.0
    assert probe_times[1] - probe_times[0] >= 600


@pytest.mark.parametrize("command", ["", "   "])
def test_missing_probe_command_never_executes_and_remains_interruptible(command):
    clock = FakeClock()
    calls = []
    agent = FakeAgent()

    def sleep(seconds):
        clock.sleep(seconds)
        if clock.now >= 0.6:
            agent._interrupt_requested = True

    waiter = ApiOutageRecoveryWaiter(
        ApiOutageRecoveryConfig(True, command, 600, 60),
        clock=clock.monotonic,
        sleep=sleep,
        runner=lambda *a, **k: calls.append((a, k)),
    )

    assert waiter.wait(agent, "endpoint", lambda _m: None, lambda _m: None) == "interrupted"
    assert calls == []


def test_probe_timeout_counts_as_failure_and_interrupt_stops_wait():
    clock = FakeClock()
    agent = FakeAgent()

    def runner(*_args, **_kwargs):
        agent._interrupt_requested = True
        raise TimeoutError("probe timed out")

    waiter = ApiOutageRecoveryWaiter(
        ApiOutageRecoveryConfig(True, "probe", 600, 5),
        clock=clock.monotonic,
        sleep=clock.sleep,
        runner=runner,
    )

    assert waiter.wait(agent, "endpoint", lambda _m: None, lambda _m: None) == "interrupted"


def test_real_probe_process_is_interruptible_during_execution():
    agent = FakeAgent()
    waiter = ApiOutageRecoveryWaiter(
        ApiOutageRecoveryConfig(True, "/bin/sleep 5", 600, 60)
    )
    interrupter = threading.Timer(
        0.1, setattr, args=(agent, "_interrupt_requested", True)
    )
    interrupter.start()
    started = time.monotonic()
    try:
        assert waiter.wait(
            agent, "real-process", lambda _m: None, lambda _m: None
        ) == "interrupted"
    finally:
        interrupter.cancel()
    assert time.monotonic() - started < 1.5


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ({}, ApiOutageRecoveryConfig(False, "", 600, 60)),
        ({"enabled": "yes", "probe_command": "check", "probe_interval_seconds": 1, "probe_timeout_seconds": 0}, ApiOutageRecoveryConfig(True, "check", 10, 1)),
        ({"enabled": "no", "probe_command": 123, "probe_interval_seconds": "bad", "probe_timeout_seconds": None}, ApiOutageRecoveryConfig(False, "", 600, 60)),
    ],
)
def test_config_parsing_defaults_booleans_and_floors(raw, expected):
    assert ApiOutageRecoveryConfig.from_mapping(raw) == expected


def test_config_defaults_are_known_and_disabled():
    from hermes_cli.config import DEFAULT_CONFIG, _validate_config_key

    assert DEFAULT_CONFIG["agent"]["api_outage_recovery"] == {
        "enabled": False,
        "probe_command": "",
        "probe_interval_seconds": 600,
        "probe_timeout_seconds": 60,
    }
    assert _validate_config_key("agent.api_outage_recovery.enabled") == (True, None)


def _make_agent(*, api_mode="chat_completions", max_retries=1):
    with (
        patch("run_agent.get_tool_definitions", return_value=[]),
        patch("run_agent.check_toolset_requirements", return_value={}),
        patch("run_agent.OpenAI", return_value=MagicMock()),
        patch(
            "hermes_cli.config.load_config",
            return_value={
                "agent": {
                    "api_max_retries": max_retries,
                    "api_outage_recovery": {
                        "enabled": True,
                        "probe_command": "probe --ready",
                        "probe_interval_seconds": 600,
                        "probe_timeout_seconds": 60,
                    },
                }
            },
        ),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://api.example/v1",
            provider="custom",
            model="test-model",
            api_mode=api_mode,
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
    agent.client = MagicMock()
    return agent


def _chat_ok(text="done"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text, tool_calls=None), finish_reason="stop")],
        model="test-model",
        usage=None,
    )


def _codex_failed(code):
    return SimpleNamespace(
        status="failed",
        error=SimpleNamespace(code=code, message="upstream failed"),
        output=None,
        output_text=None,
        model="test-model",
        usage=None,
        id="failed",
    )


def _codex_ok(text="done"):
    return SimpleNamespace(
        status="completed",
        error=None,
        output=[SimpleNamespace(type="message", content=[SimpleNamespace(type="output_text", text=text)])],
        output_text=text,
        model="test-model",
        usage=SimpleNamespace(input_tokens=1, output_tokens=1, total_tokens=2),
        id="ok",
    )


class HttpError(Exception):
    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.body = {"error": {"message": message}}
        self.response = SimpleNamespace(headers={})


def _run(agent, api_side_effect, *, conversation_history=None):
    with (
        patch.object(agent, "_interruptible_api_call", side_effect=api_side_effect),
        patch.object(agent, "_try_recover_primary_transport", return_value=False),
        patch.object(agent, "_persist_session"),
        patch.object(agent, "_save_trajectory"),
        patch.object(agent, "_cleanup_task_resources"),
        patch("agent.conversation_loop.time.sleep"),
        patch("agent.conversation_loop.jittered_backoff", return_value=0.0),
        patch("agent.agent_runtime_helpers.time.sleep"),
    ):
        return agent.run_conversation(
            "hello", conversation_history=conversation_history
        )


@pytest.mark.parametrize(
    "transient_error",
    [
        HttpError(500, "server error"),
        HttpError(503, "server overloaded"),
        TimeoutError("upstream timed out"),
    ],
)
def test_transient_outage_parks_probes_and_retries_same_boundary_without_message_replay(
    transient_error,
):
    agent = _make_agent()
    clock = FakeClock()
    probe_calls = []
    agent._api_outage_recovery_waiter = ApiOutageRecoveryWaiter(
        agent._api_outage_recovery_config,
        clock=clock.monotonic,
        sleep=clock.sleep,
        runner=lambda argv, **kwargs: probe_calls.append((argv, kwargs)) or SimpleNamespace(returncode=0, stdout="", stderr=""),
    )
    snapshots = []

    def transport(_kwargs):
        snapshots.append([dict(m) for m in agent._session_messages])
        if len(snapshots) == 1:
            raise transient_error
        return _chat_ok()

    completed_tool_history = [
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_done",
                    "type": "function",
                    "function": {"name": "terminal", "arguments": "{}"},
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call_done",
            "name": "terminal",
            "content": "already completed",
        },
    ]
    result = _run(
        agent, transport, conversation_history=completed_tool_history
    )

    assert result["completed"] is True
    assert result["final_response"] == "done"
    assert len(snapshots) == 2
    assert snapshots[0] == snapshots[1]
    assert [m["role"] for m in result["messages"]].count("user") == 1
    assert [m["role"] for m in result["messages"]].count("tool") == 1
    assert len(probe_calls) == 1


@pytest.mark.parametrize("status,message", [(400, "invalid request"), (401, "invalid api key"), (402, "payment required")])
def test_nontransient_errors_do_not_probe_and_keep_failed(status, message):
    agent = _make_agent()
    waiter = MagicMock()
    agent._api_outage_recovery_waiter = waiter

    result = _run(agent, HttpError(status, message))

    assert result["completed"] is False
    assert result["failed"] is True
    waiter.wait.assert_not_called()


def test_codex_transient_terminal_parks_then_resumes_but_generic_malformed_does_not():
    transient = _make_agent(api_mode="codex_responses")
    transient_waiter = MagicMock()
    transient_waiter.wait.return_value = "recovered"
    transient._api_outage_recovery_waiter = transient_waiter
    result = _run(transient, [_codex_failed("upstream_error"), _codex_ok()])
    assert result["completed"] is True
    transient_waiter.wait.assert_called_once()

    malformed = _make_agent(api_mode="codex_responses")
    malformed_waiter = MagicMock()
    malformed._api_outage_recovery_waiter = malformed_waiter
    result = _run(malformed, [SimpleNamespace(status="completed", output=None, output_text=None, error=None), SimpleNamespace(status="completed", output=None, output_text=None, error=None)])
    assert result["failed"] is True
    malformed_waiter.wait.assert_not_called()


def test_interrupt_while_parked_returns_interrupted_not_failed():
    agent = _make_agent()
    waiter = MagicMock()
    waiter.wait.return_value = "interrupted"
    agent._api_outage_recovery_waiter = waiter
    agent.clear_interrupt = MagicMock()

    result = _run(agent, HttpError(503, "server overloaded"))

    assert result["completed"] is False
    assert result["interrupted"] is True
    assert "failed" not in result
    agent.clear_interrupt.assert_called_once()
