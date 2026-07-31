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
