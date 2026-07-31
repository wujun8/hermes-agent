"""Optional active-turn parking while a transient model API outage persists."""

from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping


_TRUE_VALUES = frozenset({"true", "1", "yes", "on"})


@dataclass(frozen=True)
class ApiOutageRecoveryConfig:
    enabled: bool = False
    probe_command: str = ""
    probe_interval_seconds: int = 600
    probe_timeout_seconds: int = 60

    @classmethod
    def from_mapping(cls, raw: Any) -> "ApiOutageRecoveryConfig":
        if not isinstance(raw, Mapping):
            raw = {}
        command_raw = raw.get("probe_command", "")
        command = command_raw.strip() if isinstance(command_raw, str) else ""
        enabled = (
            str(raw.get("enabled", False)).strip().lower() in _TRUE_VALUES
            and bool(command)
        )
        try:
            interval = max(10, int(raw.get("probe_interval_seconds", 600)))
        except (TypeError, ValueError):
            interval = 600
        try:
            timeout = max(1, int(raw.get("probe_timeout_seconds", 60)))
        except (TypeError, ValueError):
            timeout = 60
        return cls(enabled, command, interval, timeout)


class ApiOutageRecoveryWaiter:
    """Run a throttled external readiness probe while keeping a turn alive.

    Probe timestamps are retained per endpoint so a successful probe followed by
    another failed model call cannot produce a tight probe/API loop.
    """

    def __init__(
        self,
        config: ApiOutageRecoveryConfig,
        *,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        runner: Callable[..., Any] | None = None,
    ) -> None:
        self.config = config
        self._clock = clock
        self._sleep = sleep
        self._runner = runner
        self._last_probe_at: dict[str, float] = {}

    @staticmethod
    def _stop_process(process: subprocess.Popen) -> None:
        """Stop a probe and its descendants without leaking a process."""
        if process.poll() is not None:
            return
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGTERM)
            else:
                process.terminate()
            process.wait(timeout=1.0)
            return
        except (OSError, subprocess.TimeoutExpired):
            pass
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except OSError:
            pass
        try:
            process.wait(timeout=1.0)
        except (OSError, subprocess.TimeoutExpired):
            pass

    def _run_probe(self, argv: list[str], agent: Any) -> str:
        """Return success/failure/interrupted without exposing probe output."""
        if self._runner is not None:
            try:
                completed = self._runner(
                    argv,
                    shell=False,
                    timeout=self.config.probe_timeout_seconds,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return "success" if completed.returncode == 0 else "failure"
            except Exception:
                return "failure"

        popen_kwargs = {
            "shell": False,
            "stdout": subprocess.DEVNULL,
            "stderr": subprocess.DEVNULL,
            "text": True,
        }
        if os.name == "posix":
            # A dedicated session lets interruption/timeout stop descendants,
            # not just the immediate probe executable.
            popen_kwargs["start_new_session"] = True
        try:
            process = subprocess.Popen(argv, **popen_kwargs)
        except Exception:
            return "failure"

        deadline = self._clock() + self.config.probe_timeout_seconds
        next_touch = self._clock() + 30.0
        while process.poll() is None:
            if getattr(agent, "_interrupt_requested", False):
                self._stop_process(process)
                return "interrupted"
            now = self._clock()
            if now >= deadline:
                self._stop_process(process)
                return "failure"
            if now >= next_touch:
                agent._touch_activity("running API outage recovery probe")
                next_touch = now + 30.0
            self._sleep(0.2)
        return "success" if process.returncode == 0 else "failure"

    def wait(
        self,
        agent: Any,
        endpoint_key: str,
        on_parked: Callable[[str], None],
        on_recovered: Callable[[str], None],
    ) -> str:
        agent._api_outage_waiting = True
        try:
            agent._touch_activity("waiting for API outage recovery probe")
            interval = self.config.probe_interval_seconds
            last_probe = self._last_probe_at.get(endpoint_key)
            now = self._clock()
            delay = (
                max(0.0, interval - (now - last_probe))
                if last_probe is not None
                else 0.0
            )
            on_parked(
                "⏸️ API outage: active turn parked in memory; "
                f"next external probe in ~{int(delay)}s."
            )
            next_touch = now + 30.0

            while True:
                if getattr(agent, "_interrupt_requested", False):
                    return "interrupted"

                now = self._clock()
                if now >= next_touch:
                    agent._touch_activity("waiting for API outage recovery probe")
                    next_touch = now + 30.0

                last_probe = self._last_probe_at.get(endpoint_key)
                due = last_probe is None or now - last_probe >= interval
                if due and self.config.probe_command:
                    # Record before launching: timeout/failure and false-positive
                    # recovery are both throttled from this exact attempt.
                    self._last_probe_at[endpoint_key] = now
                    try:
                        argv = shlex.split(self.config.probe_command)
                    except ValueError:
                        argv = []
                    probe_result = self._run_probe(argv, agent) if argv else "failure"
                    if probe_result == "interrupted":
                        return "interrupted"
                    if probe_result == "success":
                        on_recovered(
                            "✅ External API recovery probe succeeded; "
                            "resuming from the same API boundary."
                        )
                        return "recovered"
                    on_parked(
                        "⚠️ Probe still failing; "
                        f"next probe in ~{interval}s."
                    )

                self._sleep(0.2)
        finally:
            agent._api_outage_waiting = False
            try:
                agent._touch_activity("API outage recovery wait ended")
            except Exception:
                # Cleanup observability is best-effort and must not replace the
                # wait result or an exception from a status callback.
                pass
