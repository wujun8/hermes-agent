"""Delegation lifecycle semantics while a child waits for API recovery."""
from __future__ import annotations

from concurrent.futures import Future, TimeoutError as FuturesTimeoutError
import threading
import time
from unittest.mock import MagicMock

import pytest

from tools import delegate_tool


class _OutageChild:
    def __init__(self, *, waiting: bool = False, activity_age: float = 0.0):
        self._api_outage_waiting = waiting
        self._last_activity_ts = time.time() - activity_age
        self._subagent_id = "sa-outage-test"
        self._delegate_depth = 1
        self._delegate_role = "leaf"
        self.max_iterations = 30
        self.model = "test/model"
        self._interrupt_requested = False
        self._done = threading.Event()

    def touch(self):
        self._last_activity_ts = time.time()

    def get_activity_summary(self):
        return {
            "api_call_count": 1,
            "max_iterations": self.max_iterations,
            "current_tool": None,
            "last_activity_desc": "waiting for API outage recovery",
        }

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        self._done.wait(2.0)
        return {
            "final_response": "recovered",
            "completed": True,
            "api_calls": 1,
        }

    def interrupt(self):
        self._interrupt_requested = True
        self._done.set()


class _TimeoutErrorChild(_OutageChild):
    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        raise TimeoutError("transport failed inside child")


@pytest.mark.parametrize(
    ("waiting", "activity_age", "expected"),
    [
        (True, 0.0, True),
        (True, 91.0, False),
        (False, 0.0, False),
    ],
)
def test_live_api_outage_wait_requires_flag_and_fresh_activity(
    waiting, activity_age, expected
):
    child = _OutageChild(waiting=waiting, activity_age=activity_age)

    assert delegate_tool._is_live_api_outage_wait(child) is expected


_MISSING = object()


@pytest.mark.parametrize(
    ("last_activity", "expected"),
    [
        (_MISSING, False),
        (None, False),
        ("999", False),
        (True, False),
        (1001.0, False),
        (911.0, True),
        (909.0, False),
    ],
)
def test_live_api_outage_wait_activity_timestamp_is_fail_closed_at_boundaries(
    last_activity, expected
):
    child = _OutageChild(waiting=True)
    if last_activity is _MISSING:
        del child._last_activity_ts
    else:
        child._last_activity_ts = last_activity

    assert delegate_tool._is_live_api_outage_wait(child, now=1000.0) is expected


def test_wait_helper_pauses_timeout_budget_then_future_can_complete():
    child = _OutageChild(waiting=True)
    future = Future()

    def recover_and_finish():
        time.sleep(0.15)  # Longer than the configured active-time budget.
        child._api_outage_waiting = False
        time.sleep(0.03)
        future.set_result("done")

    thread = threading.Thread(target=recover_and_finish, daemon=True)
    thread.start()

    assert delegate_tool._wait_for_child_result(
        future, child, timeout=0.08, poll_interval=0.01
    ) == "done"
    thread.join(timeout=1.0)


def test_wait_helper_live_outage_can_exceed_configured_timeout():
    child = _OutageChild(waiting=True)
    future = Future()

    def finish_during_outage():
        time.sleep(0.12)
        child.touch()
        future.set_result("done")

    thread = threading.Thread(target=finish_during_outage, daemon=True)
    thread.start()

    assert delegate_tool._wait_for_child_result(
        future, child, timeout=0.04, poll_interval=0.01
    ) == "done"
    thread.join(timeout=1.0)


def test_wait_helper_stale_outage_flag_does_not_pause_timeout():
    child = _OutageChild(waiting=True, activity_age=91.0)
    future = Future()
    started = time.monotonic()

    with pytest.raises(FuturesTimeoutError):
        delegate_tool._wait_for_child_result(
            future, child, timeout=0.05, poll_interval=0.01
        )

    assert time.monotonic() - started < 0.3


def test_wait_helper_propagates_completed_future_timeout_error():
    child = _OutageChild(waiting=False)
    future = Future()
    future.set_exception(TimeoutError("child failure"))

    with pytest.raises(TimeoutError, match="child failure"):
        delegate_tool._wait_for_child_result(future, child, timeout=1.0)


def test_run_single_child_does_not_misclassify_child_timeout_error(monkeypatch):
    child = _TimeoutErrorChild()
    parent = MagicMock()
    parent._current_task_id = None
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 1.0)

    result = delegate_tool._run_single_child(
        task_index=2, goal="fail", child=child, parent_agent=parent
    )

    assert result["status"] == "error"
    assert result["exit_reason"] == "error"
    assert result["error"] == "transport failed inside child"


def test_run_single_child_keeps_heartbeat_and_timeout_alive_during_outage(
    monkeypatch,
):
    child = _OutageChild(waiting=True)
    parent = MagicMock()
    parent._current_task_id = None
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 0.05)
    monkeypatch.setattr(delegate_tool, "_HEARTBEAT_INTERVAL", 0.01)

    def finish_after_long_outage():
        # Keep the fail-closed freshness signal live while waiting.
        for _ in range(8):
            time.sleep(0.01)
            child.touch()
        child._api_outage_waiting = False
        child._done.set()

    thread = threading.Thread(target=finish_after_long_outage, daemon=True)
    thread.start()
    result = delegate_tool._run_single_child(
        task_index=3,
        goal="recover",
        child=child,
        parent_agent=parent,
    )
    thread.join(timeout=1.0)

    assert result["status"] == "completed"
    assert child._interrupt_requested is False
    parent._touch_activity.assert_any_call(
        "delegate_task: subagent waiting for API recovery (iteration 1/30)"
    )


class _StructuredResultChild(_OutageChild):
    def __init__(self, result):
        super().__init__()
        self._result = result

    def run_conversation(self, user_message, task_id=None, stream_callback=None):
        return dict(self._result)


@pytest.mark.parametrize(
    ("child_result", "expected"),
    [
        (
            {
                "final_response": "provider failed",
                "completed": False,
                "failed": True,
                "error": "HTTP 403",
                "failure_reason": "provider",
                "turn_exit_reason": "non_retryable_client_error",
                "api_calls": 14,
            },
            {
                "status": "error",
                "exit_reason": "error",
                "truncated": False,
            },
        ),
        (
            {
                "final_response": "usable summary",
                "completed": False,
                "failed": False,
                "turn_exit_reason": "max_iterations_reached(70/70)",
                "api_calls": 70,
            },
            {
                "status": "completed",
                "exit_reason": "max_iterations",
                "truncated": True,
            },
        ),
    ],
)
def test_run_single_child_classifies_structured_provider_failure_and_max_iteration(
    monkeypatch, child_result, expected
):
    child = _StructuredResultChild(child_result)
    parent = MagicMock()
    parent._current_task_id = None
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 1.0)

    result = delegate_tool._run_single_child(
        task_index=4, goal="classify", child=child, parent_agent=parent
    )

    for key, value in expected.items():
        assert result[key] == value
    if child_result.get("failed"):
        assert result["error"] == "HTTP 403"
        assert result["turn_exit_reason"] == "non_retryable_client_error"
        assert result["failure_reason"] == "provider"


def test_run_single_child_parent_stop_and_generic_interrupt_have_distinct_reasons(
    monkeypatch,
):
    parent = MagicMock()
    parent._current_task_id = None
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 1.0)

    parent_stop = delegate_tool._run_single_child(
        task_index=5,
        goal="stop",
        child=_StructuredResultChild(
            {
                "final_response": "",
                "completed": False,
                "interrupted": True,
                "interrupt_message": "[delegate_task parent-control] stop requested",
                "turn_exit_reason": "interrupted",
            }
        ),
        parent_agent=parent,
    )
    generic = delegate_tool._run_single_child(
        task_index=6,
        goal="interrupt",
        child=_StructuredResultChild(
            {
                "final_response": "",
                "completed": False,
                "interrupted": True,
                "interrupt_message": "Operation interrupted by user",
                "turn_exit_reason": "interrupted_by_user",
            }
        ),
        parent_agent=parent,
    )

    assert parent_stop["status"] == "interrupted"
    assert parent_stop["exit_reason"] == "parent_stop"
    assert parent_stop["truncated"] is False
    assert generic["status"] == "interrupted"
    assert generic["exit_reason"] == "interrupted"
    assert generic["truncated"] is False


def test_run_single_child_preserves_non_failure_incomplete_reason(monkeypatch):
    child = _StructuredResultChild(
        {
            "final_response": "",
            "completed": False,
            "failed": False,
            "turn_exit_reason": "partial_stream_recovery",
        }
    )
    parent = MagicMock()
    parent._current_task_id = None
    monkeypatch.setattr(delegate_tool, "_get_child_timeout", lambda: 1.0)

    result = delegate_tool._run_single_child(
        task_index=7, goal="incomplete", child=child, parent_agent=parent
    )

    assert result["status"] == "incomplete"
    assert result["exit_reason"] == "partial_stream_recovery"
    assert result["truncated"] is False
