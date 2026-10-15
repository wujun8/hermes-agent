"""A delegated outage probes its real route and cannot park indefinitely."""

import json
from types import SimpleNamespace

from agent.api_outage_probe import active_route_input
from agent.api_outage_recovery import ApiOutageRecoveryConfig, ApiOutageRecoveryWaiter, park_for_api_outage_recovery
from agent.delegation_route_recovery import configure_child_recovery
from tests.agent.test_api_outage_recovery import FakeAgent, FakeClock


def _route_agent():
    agent = FakeAgent()
    agent.model = "deepseek-flash"
    agent.provider = "custom"
    agent.requested_provider = "custom:worker"
    agent.base_url = "https://worker.example/v1"
    agent.api_mode = "chat_completions"
    agent.api_key = "outdated-key"
    agent._client_kwargs = {"api_key": "live-worker-key", "default_headers": {"X-Route": "worker"}}
    agent.request_overrides = {"extra_body": {"routing": "worker"}}
    return agent


def test_active_probe_passes_live_route_via_private_stdin(monkeypatch, tmp_path):
    from agent import api_outage_probe

    profile_home = str(tmp_path / "own-profile")
    monkeypatch.setattr(api_outage_probe, "active_probe_environment", lambda _a: {"HERMES_HOME": profile_home})
    calls = []
    waiter = ApiOutageRecoveryWaiter(
        ApiOutageRecoveryConfig(True, "probe --active-route-stdin", 120, 60),
        runner=lambda argv, **kw: calls.append((argv, kw)) or SimpleNamespace(returncode=0),
    )
    agent = _route_agent()
    agent._client_kwargs["default_query"] = {"tenant": "worker", "api-key": "private-query-key"}
    agent.request_overrides["extra_query"] = {"tenant": "override"}
    assert waiter.wait(agent, "worker", lambda _s: None, lambda _s: None) == "recovered"
    argv, kwargs = calls[0]
    payload = json.loads(kwargs["input"])
    assert payload["model"] == {
        "provider": "custom:worker", "default": "deepseek-flash", "base_url": "https://worker.example/v1",
        "api_mode": "chat_completions", "api_key": "live-worker-key", "default_headers": {"X-Route": "worker"},
    }
    assert payload["request_overrides"] == {
        "extra_body": {"routing": "worker"},
        "extra_query": {"tenant": "override", "api-key": "private-query-key"},
    }
    assert "live-worker-key" not in " ".join(argv)
    assert "private-query-key" not in " ".join(argv)
    assert agent.request_overrides["extra_query"] == {"tenant": "override"}
    assert kwargs["env"] == {"HERMES_HOME": profile_home}


def test_child_wait_has_cumulative_deadline_and_returns_terminal_failure():
    clock = FakeClock()
    agent = _route_agent()
    configure_child_recovery(agent, {}, {"api_outage_max_wait_seconds": 3})
    config = ApiOutageRecoveryConfig(True, "probe", 10, 60)
    calls = []
    waiter = ApiOutageRecoveryWaiter(
        config, clock=clock.monotonic, sleep=clock.sleep,
        runner=lambda argv, **kw: calls.append(kw) or SimpleNamespace(returncode=1),
    )
    agent._api_outage_recovery_config = config
    agent._api_outage_recovery_waiter = waiter
    agent._persist_session = lambda *_a: None
    agent._emit_status = lambda _s: None
    agent._api_call_count = 3
    result = park_for_api_outage_recovery(agent, [], None)
    assert result["completed"] is False and result["failed"] is True
    assert result["failure_reason"] == "timeout"
    assert not result.get("interrupted") and "resume_reason" not in result
    assert agent._api_outage_waiting is False
    assert 3 <= clock.now < 3.3
    assert calls[0]["timeout"] <= 3
    # Another exhausted attempt cannot reset the child's cumulative allowance.
    waiter.wait(agent, "worker", lambda _s: None, lambda _s: None)
    assert len(calls) == 1


def test_active_probe_rejects_unsupported_transport_without_serializing_secrets():
    import pytest

    agent = _route_agent()
    agent.api_mode = "unsupported"
    with pytest.raises(ValueError, match="transport") as error:
        active_route_input(agent)
    assert "live-worker-key" not in str(error.value)
