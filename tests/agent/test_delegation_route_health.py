"""Behavioral contracts for temporary delegation route health."""
from __future__ import annotations

import threading

from agent.delegation_route_health import ROUTE_HEALTH, RouteHealthRegistry


class FakeClock:
    def __init__(self, now: float = 0.0):
        self.now = now

    def __call__(self) -> float:
        return self.now


def test_failure_cooldown_half_open_probe_and_success():
    clock = FakeClock()
    registry = RouteHealthRegistry(clock=clock)
    key = ("profile-a", "selector", "endpoint")

    permit = registry.acquire(key, cooldown_seconds=10)
    assert permit is not None
    assert registry.acquire(key) is not None
    registry.fail(permit)
    assert registry.acquire(key) is None

    clock.now = 9.999
    assert registry.acquire(key) is None
    clock.now = 10
    probe = registry.acquire(key, cooldown_seconds=10, probe_lease_seconds=5)
    assert probe is not None
    assert registry.acquire(key) is None

    registry.succeed(probe)
    assert registry.acquire(key) is not None
    assert registry.acquire(key) is not None


def test_stale_failure_extends_cooldown_and_invalidates_half_open_success():
    clock = FakeClock()
    registry = RouteHealthRegistry(clock=clock)
    key = ("profile-a", "selector", "endpoint")

    original = registry.acquire(key, cooldown_seconds=10)
    assert original is not None
    registry.fail(original)
    clock.now = 10
    probe = registry.acquire(key, cooldown_seconds=10, probe_lease_seconds=5)
    assert probe is not None

    # A late failure from the original, now-stale attempt starts a fresh cooldown.
    registry.fail(original)
    registry.succeed(probe)
    assert registry.acquire(key) is None
    clock.now = 19.999
    assert registry.acquire(key) is None
    clock.now = 20
    assert registry.acquire(key) is not None


def test_abandoned_half_open_permit_releases_after_lease_expiry():
    clock = FakeClock()
    registry = RouteHealthRegistry(clock=clock)
    key = ("profile-a", "selector", "endpoint")

    permit = registry.acquire(key, cooldown_seconds=10)
    assert permit is not None
    registry.fail(permit)
    clock.now = 10
    abandoned = registry.acquire(key, probe_lease_seconds=5)
    assert abandoned is not None
    clock.now = 14.999
    assert registry.acquire(key) is None
    clock.now = 15
    retry = registry.acquire(key)
    assert retry is not None
    assert retry.generation == abandoned.generation


def test_failure_for_one_profile_does_not_block_another():
    registry = RouteHealthRegistry(clock=FakeClock())
    first = registry.acquire(("profile-a", "same-route"), cooldown_seconds=30)
    assert first is not None
    registry.fail(first)

    assert registry.acquire(("profile-a", "same-route")) is None
    assert registry.acquire(("profile-b", "same-route")) is not None


def test_concurrent_half_open_acquires_have_one_winner():
    clock = FakeClock()
    registry = RouteHealthRegistry(clock=clock)
    key = ("profile-a", "selector", "endpoint")
    permit = registry.acquire(key, cooldown_seconds=10)
    assert permit is not None
    registry.fail(permit)
    clock.now = 10

    barrier = threading.Barrier(3)
    permits = []
    result_lock = threading.Lock()

    def acquire_after_barrier():
        barrier.wait(timeout=5)
        result = registry.acquire(key, probe_lease_seconds=5)
        with result_lock:
            permits.append(result)

    workers = [threading.Thread(target=acquire_after_barrier) for _ in range(2)]
    for worker in workers:
        worker.start()
    barrier.wait(timeout=5)
    for worker in workers:
        worker.join(timeout=5)

    assert all(not worker.is_alive() for worker in workers)
    assert len(permits) == 2
    assert sum(permit is not None for permit in permits) == 1


def test_module_level_registry_is_available():
    assert isinstance(ROUTE_HEALTH, RouteHealthRegistry)


def test_live_agent_property_route_failure_and_usable_response_recovery(monkeypatch):
    from types import SimpleNamespace

    from agent import delegation_route_recovery as recovery
    from run_agent import AIAgent

    clock = FakeClock()
    registry = RouteHealthRegistry(clock=clock)
    monkeypatch.setattr(recovery, "ROUTE_HEALTH", registry)
    agent = AIAgent.__new__(AIAgent)
    agent.requested_provider = "custom:worker"
    agent.provider = "custom"
    agent.model = "deepseek-flash"
    agent.base_url = "https://worker.example/v1"
    agent.api_mode = "chat_completions"
    agent._client_kwargs = {"default_query": {"tenant": "worker"}}
    key = ("profile", "selector", "worker")
    identity = recovery._route_identity({
        "provider": "custom:worker", "base_url": agent.base_url + "?tenant=worker",
        "model": agent.model, "api_mode": agent.api_mode,
    })
    agent._selected_route_recovery = {
        "permit": registry.acquire(key, cooldown_seconds=10), "identity": identity,
    }

    recovery.record_route_failure(agent, "server_error")
    assert registry.acquire(key) is None
    assert agent._selected_route_recovery is None

    clock.now = 10
    probe = registry.acquire(key)
    agent._selected_route_recovery = {"permit": probe, "identity": identity}
    agent._api_outage_wait_elapsed_seconds = 3
    recovery.record_route_success(agent, SimpleNamespace(content="", tool_calls=[]))
    assert registry.acquire(key) is None
    assert agent._api_outage_wait_elapsed_seconds == 3

    recovery.record_route_success(agent, SimpleNamespace(content="ready", tool_calls=[]))
    assert registry.acquire(key) is not None
    assert registry.acquire(key) is not None
    assert agent._api_outage_wait_elapsed_seconds == 0
