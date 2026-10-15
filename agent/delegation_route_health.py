"""In-memory cooldown and half-open probes for temporary delegation routes."""
from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class RoutePermit:
    key: tuple[str, ...]
    generation: int
    cooldown_seconds: float


@dataclass
class _RouteState:
    generation: int
    cooldown_until: float
    probe_lease_until: float | None = None


class RouteHealthRegistry:
    """Keep temporary route failures isolated by the caller-supplied route key."""

    def __init__(self, *, clock: Callable[[], float] = time.monotonic):
        self._clock = clock
        self._lock = threading.Lock()
        self._states: dict[tuple[str, ...], _RouteState] = {}
        # Healthy routes are not stored. A registry-wide epoch keeps permits stale even
        # after a recovered route's state has been removed.
        self._generation = 0

    def acquire(
        self,
        key: tuple[str, ...],
        *,
        cooldown_seconds: float = 300.0,
        probe_lease_seconds: float = 60.0,
    ) -> RoutePermit | None:
        _validate_duration(cooldown_seconds, "cooldown_seconds", allow_zero=True)
        _validate_duration(probe_lease_seconds, "probe_lease_seconds", allow_zero=False)

        with self._lock:
            state = self._states.get(key)
            if state is not None:
                now = self._clock()
                if now < state.cooldown_until:
                    return None
                if state.probe_lease_until is not None and now < state.probe_lease_until:
                    return None
                state.probe_lease_until = now + probe_lease_seconds
                generation = state.generation
            else:
                generation = self._generation

            return RoutePermit(key, generation, cooldown_seconds)

    def fail(self, permit: RoutePermit) -> None:
        with self._lock:
            now = self._clock()
            self._generation = max(self._generation, permit.generation) + 1
            cooldown_until = now + permit.cooldown_seconds
            previous = self._states.get(permit.key)
            if previous is not None:
                cooldown_until = max(cooldown_until, previous.cooldown_until)
            self._states[permit.key] = _RouteState(
                generation=self._generation,
                cooldown_until=cooldown_until,
            )

    def succeed(self, permit: RoutePermit) -> None:
        with self._lock:
            state = self._states.get(permit.key)
            if state is not None and state.generation == permit.generation:
                del self._states[permit.key]


def _validate_duration(value: float, name: str, *, allow_zero: bool) -> None:
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {qualifier}")


ROUTE_HEALTH = RouteHealthRegistry()
