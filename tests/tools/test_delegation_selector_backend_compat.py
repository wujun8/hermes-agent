"""Legacy route selectors remain available through the default per-task backend."""

from types import SimpleNamespace

from hermes_cli.delegation_routes import DelegationRouteSelection
from tools import delegate_tool, delegate_tool_routes
from hermes_cli import plugins_delegation_routes as legacy_routes


def _parent():
    return SimpleNamespace(session_id="selector-compat-session")


def _credentials(route, _parent_agent):
    return {
        "model": route.get("model"),
        "provider": "parent-provider",
        "base_url": "https://parent.example/v1",
        "api_key": "parent-key",
        "api_mode": "chat_completions",
    }


def test_default_backend_runs_selector_once_with_legacy_settings(monkeypatch):
    from hermes_cli import plugins
    import tools.delegate_tool_tuning as tuning

    config = {
        "backend": "default",
        "backends": {},
        "model": "configured-model",
        "provider": "",
        "base_url": "",
        "api_key": "",
        "api_mode": "",
        "fallback_providers": None,
        "reasoning_effort": "low",
        "api_outage_max_wait_seconds": 600,
        "schedule": {"strategy": "least_busy", "window_seconds": 45},
    }
    default_provider = delegate_tool_routes._active_provider(config)
    route_calls = []

    def route_task(task, _config):
        route_calls.append(task["goal"])
        return {
            "model": "backend-model",
            "reasoning_effort": "high",
            "service_tier": "fast",
            "label": "scheduled route",
        }

    monkeypatch.setattr(default_provider, "route_task", route_task)
    credential_calls = []

    def credentials(route, parent_agent):
        credential_calls.append(dict(route))
        return _credentials(route, parent_agent)

    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", credentials)
    monkeypatch.setattr(tuning, "task_service_tier", lambda tier, _creds, _parent: {"service_tier": tier})
    selector_settings = []

    def select(settings):
        selector_settings.append(dict(settings))
        return DelegationRouteSelection("selected-model")

    manager = SimpleNamespace(_delegation_route_selectors={"legacy-scheduler": {"select": select}})
    monkeypatch.setattr(plugins, "_ensure_plugins_discovered", lambda: manager)
    resolve_inputs = []
    resolve_route = legacy_routes.resolve_delegation_route

    def record_resolve(route_config, parent_agent, **kwargs):
        resolve_inputs.append(dict(route_config))
        return resolve_route(route_config, parent_agent, **kwargs)

    monkeypatch.setattr(legacy_routes, "resolve_delegation_route", record_resolve)

    routes, error = delegate_tool_routes.resolve_task_routes(
        [{"goal": "first"}, {"goal": "second"}], config, _parent(),
    )

    assert error is None
    assert route_calls == ["first", "second"]
    assert len(selector_settings) == 1
    assert selector_settings[0]["api_outage_max_wait_seconds"] == 600
    assert selector_settings[0]["schedule"] == {"strategy": "least_busy", "window_seconds": 45}
    assert len(resolve_inputs) == 1
    assert resolve_inputs[0]["api_outage_max_wait_seconds"] == 600
    assert resolve_inputs[0]["schedule"] == {"strategy": "least_busy", "window_seconds": 45}
    assert resolve_inputs[0]["model"] == "backend-model"
    assert resolve_inputs[0]["reasoning_effort"] == "high"
    assert resolve_inputs[0]["service_tier"] == "fast"
    assert len(credential_calls) == 1
    assert [route.creds["model"] for route in routes] == ["selected-model", "selected-model"]
    assert [route.creds["service_tier"] for route in routes] == ["fast", "fast"]
    assert [route.reasoning_effort for route in routes] == ["high", "high"]
    assert [route.label for route in routes] == ["scheduled route", "scheduled route"]
    assert routes[0].routing_cfg["model"] == "selected-model"


def test_nondefault_backend_bypasses_legacy_selectors(monkeypatch):
    calls = []

    def fail_selector(*_args, **_kwargs):
        calls.append("called")
        raise AssertionError("selector resolver was called")

    provider = SimpleNamespace(
        name="alternate",
        task_parameters=lambda _config: {},
        route_task=lambda _task, _config: {"model": "alternate-model", "label": "alternate"},
    )
    monkeypatch.setattr(delegate_tool_routes, "_active_provider", lambda _config: provider)
    monkeypatch.setattr(legacy_routes, "resolve_delegation_route", fail_selector)
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", _credentials)

    routes, error = delegate_tool_routes.resolve_task_routes(
        [{"goal": "one"}], {"backend": "alternate", "backends": {"alternate": {}}}, _parent(),
    )

    assert error is None
    assert calls == []
    assert routes[0].backend == "alternate"
    assert routes[0].creds["model"] == "alternate-model"
    assert routes[0].label == "alternate"


def test_credentials_cfg_bypasses_selectors(monkeypatch):
    monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", _credentials)
    monkeypatch.setattr(
        legacy_routes,
        "resolve_delegation_route",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("selector resolver was called")),
    )

    routing_cfg = {"model": "review-model", "provider": "review-provider", "api_key": "review-key"}
    routes, error = delegate_tool._routes_for(
        [{"goal": "review"}], {"backend": "default", "backends": {}}, routing_cfg, _parent(),
    )

    assert error is None
    assert len(routes) == 1
    assert routes[0].routing_cfg is routing_cfg
    assert routes[0].creds["model"] == "review-model"
