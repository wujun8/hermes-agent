"""Optional route policy stays in core while provider routes remain plain JSON."""

import os
from pathlib import Path
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from hermes_cli import plugins as plugins_mod
from hermes_cli.plugins import PluginManager


_PROVIDER = '''
import os
from agent.delegation_provider import DelegationProvider

class RouteProbe(DelegationProvider):
    @property
    def name(self):
        return "route-policy-probe"

    def task_parameters(self, config):
        return {
            "option": {"type": "string", "enum": sorted(self.settings(config).get("routes", {}))},
            # Route policy is core metadata and must never become a model-facing parameter.
            "route_policy": {"type": "object"},
        }

    def route_task(self, task, config):
        route = dict(self.settings(config)["routes"][task["option"]])
        if route.pop("pid_suffix", False):
            route["model"] = f"{route['model']}-{os.getpid()}"
        return route

def register(ctx):
    ctx.register_delegation_provider(RouteProbe())
'''


def _environment(tmp_path, monkeypatch, delegation, *, isolation="in_process"):
    home = tmp_path / "os-home" / ".hermes"
    plugin_dir = home / "plugins" / "route-policy-probe"
    plugin_dir.mkdir(parents=True)
    (plugin_dir / "plugin.yaml").write_text(
        yaml.safe_dump({"name": "route-policy-probe", "version": "1.0.0"}), encoding="utf-8",
    )
    (plugin_dir / "__init__.py").write_text(_PROVIDER, encoding="utf-8")
    document = {
        "plugins": {"enabled": ["route-policy-probe"], "isolation": isolation},
        "delegation": delegation,
    }
    (home / "config.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")

    bundled = tmp_path / "bundled"
    bundled.mkdir()
    bundled.joinpath("delegation").symlink_to(
        Path(__file__).resolve().parents[2] / "plugins" / "delegation", target_is_directory=True,
    )
    monkeypatch.setenv("HOME", str(home.parent))
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    monkeypatch.setattr(plugins_mod, "get_bundled_plugins_dir", lambda: bundled)
    manager = PluginManager()
    manager.discover_and_load()
    return home, manager, document["delegation"]


def _parent():
    return SimpleNamespace(
        session_id="route-policy-test", model="parent-model", provider="custom",
        base_url="https://parent.invalid/v1", api_key="parent-secret", api_mode="chat_completions",
        request_overrides={}, _delegate_depth=0,
    )


def test_task_route_cache_ignores_labels_without_reusing_another_routes_credentials(
    tmp_path, monkeypatch,
):
    from tools import delegate_tool
    from tools.delegate_tool_routes import resolve_task_routes, schema_extension

    shared_route = {
        "model": "worker-model", "base_url": "https://worker.invalid/v1",
        "api_key": "worker-a-secret", "api_mode": "chat_completions",
        "route_policy": {"fallback_to_configured": False},
    }
    other_route = {**shared_route, "api_key": "worker-b-secret"}
    config = {
        "backend": "route-policy-probe",
        "model": "configured-model", "base_url": "https://configured.invalid/v1",
        "api_key": "configured-secret", "api_mode": "chat_completions",
        "backends": {"route-policy-probe": {"routes": {
            "first": {**shared_route, "label": "first label"},
            "same": {**shared_route, "label": "same route, other label"},
            "other": {**other_route, "label": "other credential"},
        }}},
    }
    _, manager, _ = _environment(tmp_path, monkeypatch, config)
    try:
        properties, _description = schema_extension(config)
        assert properties["option"]["enum"] == ["first", "other", "same"]
        assert "route_policy" not in properties

        original_resolver = delegate_tool._resolve_delegation_credentials
        resolved = []

        def record_resolution(route, parent):
            resolved.append(dict(route))
            return original_resolver(route, parent)

        monkeypatch.setattr(delegate_tool, "_resolve_delegation_credentials", record_resolution)
        routes, error = resolve_task_routes(
            [{"goal": "first", "option": "first"},
             {"goal": "same route", "option": "same"},
             {"goal": "other credentials", "option": "other"}],
            config, _parent(),
        )

        assert error is None
        assert len(resolved) == 2
        assert [route.label for route in routes] == [
            "first label", "same route, other label", "other credential",
        ]
        assert [(route.creds["model"], route.creds["api_key"]) for route in routes] == [
            ("worker-model", "worker-a-secret"),
            ("worker-model", "worker-a-secret"),
            ("worker-model", "worker-b-secret"),
        ]
    finally:
        manager.unload()


@pytest.mark.parametrize(("invalid", "fallback"), [
    ("mode", True), ("mode", False), ("host", True), ("host", False),
])
def test_invalid_optional_route_policy_falls_back_only_when_opted_in_without_logging_secrets(
    tmp_path, monkeypatch, caplog, invalid, fallback,
):
    from tools.delegate_tool_routes import resolve_task_routes

    selected = {
        "model": "optional-model", "base_url": "https://optional.invalid/v1",
        "api_key": "optional-route-secret", "api_mode": "chat_completions",
        "route_policy": {"fallback_to_configured": fallback, "failure_cooldown_seconds": 60},
    }
    if invalid == "mode":
        selected["api_mode"] = "not-a-supported-mode"
    else:
        selected["route_policy"]["required_endpoint_host"] = "required.invalid"
    config = {
        "backend": "route-policy-probe",
        "model": "configured-model", "base_url": "https://configured.invalid/v1",
        "api_key": "configured-route-secret", "api_mode": "codex_responses",
        "backends": {"route-policy-probe": {"routes": {"optional": selected}}},
    }
    _, manager, _ = _environment(tmp_path, monkeypatch, config)
    try:
        caplog.set_level("DEBUG", logger="tools.delegate_tool")
        routes, error = resolve_task_routes(
            [{"goal": "take optional route", "option": "optional"}], config, _parent(),
        )
        assert error is None
        route = routes[0]
        if fallback:
            assert route.error is None
            assert (route.creds["model"], route.creds["base_url"], route.creds["api_key"],
                    route.creds["api_mode"]) == (
                "configured-model", "https://configured.invalid/v1", "configured-route-secret",
                "codex_responses",
            )
        else:
            assert route.error
            assert route.creds == {}
            expected = "api_mode" if invalid == "mode" else "endpoint"
            assert expected in route.error
            assert "optional-route-secret" not in route.error
            assert "configured-route-secret" not in route.error
        assert "optional-route-secret" not in caplog.text
        assert "configured-route-secret" not in caplog.text
    finally:
        manager.unload()


def test_profile_route_keeps_its_own_credentials_and_internal_credentials_cfg_bypasses_backend(
    tmp_path, monkeypatch,
):
    from tools import delegate_tool
    from tools.delegate_tool_routes import resolve_task_routes, schema_extension

    home = tmp_path / "os-home" / ".hermes"
    profile = home / "profiles" / "worker"
    profile.mkdir(parents=True)
    (profile / "config.yaml").write_text(yaml.safe_dump({
        "model": {"provider": "custom", "base_url": "https://worker.invalid/v1",
                  "default": "profile-model", "key_env": "PROFILE_KEY"},
    }), encoding="utf-8")
    (profile / ".env").write_text("PROFILE_KEY=profile-secret\n", encoding="utf-8")
    monkeypatch.setenv("PROFILE_KEY", "launch-profile-secret")

    config = {
        "backend": "route-policy-probe",
        "model": "configured-model", "base_url": "https://configured.invalid/v1",
        "api_key": "configured-secret", "api_mode": "chat_completions",
        "backends": {"route-policy-probe": {"routes": {
            "profile": {"profile": "worker", "route_policy": {
                "fallback_to_configured": "invalid", "failure_cooldown_seconds": 0,
            }},
        }}},
    }
    _, manager, _ = _environment(tmp_path, monkeypatch, config)
    try:
        properties, _description = schema_extension(config)
        assert "route_policy" not in properties
        routes, error = resolve_task_routes(
            [{"goal": "use the worker profile", "option": "profile"}], config, _parent(),
        )
        assert error is None
        profile_route = routes[0]
        assert profile_route.profile_home == profile
        assert (profile_route.creds["model"], profile_route.creds["base_url"], profile_route.creds["api_key"]) == (
            "profile-model", "https://worker.invalid/v1", "profile-secret",
        )

        # This is an internal owner route (for example /review); even an invalid policy cannot redirect it.
        internal_cfg = {
            "model": "review-model", "base_url": "https://review.invalid/v1",
            "api_key": "review-secret", "api_mode": "chat_completions",
            "route_policy": {"fallback_to_configured": "invalid"},
        }
        monkeypatch.setattr(
            "tools.delegate_tool_routes._active_provider",
            lambda _config: (_ for _ in ()).throw(AssertionError("internal route consulted the backend")),
        )
        internal_routes, internal_error = delegate_tool._routes_for(
            [{"goal": "internal review route"}], config, internal_cfg, _parent(),
        )
        assert internal_error is None
        assert internal_routes[0].routing_cfg is internal_cfg
        assert (internal_routes[0].creds["model"], internal_routes[0].creds["api_key"]) == (
            "review-model", "review-secret",
        )
    finally:
        manager.unload()


@pytest.mark.platforms("posix")
def test_hosted_provider_returns_json_policy_for_core_validation(tmp_path, monkeypatch):
    from tools.delegate_tool_routes import resolve_task_routes, schema_extension

    config = {
        "backend": "route-policy-probe",
        "model": "configured-model", "base_url": "https://configured.invalid/v1",
        "api_key": "configured-secret", "api_mode": "chat_completions",
        "backends": {"route-policy-probe": {"routes": {"hosted": {
            "model": "host-model", "pid_suffix": True,
            "base_url": "https://hosted.invalid/v1", "api_key": "hosted-secret",
            "api_mode": "chat_completions",
            "route_policy": {"required_endpoint_host": "hosted.invalid"},
        }}}},
    }
    _, manager, _ = _environment(tmp_path, monkeypatch, config, isolation="host")
    host = manager._plugin_host()
    try:
        assert manager._plugins["route-policy-probe"].error is None
        params, _description = schema_extension(config)
        assert params["option"]["enum"] == ["hosted"]
        assert "route_policy" not in params
        assert manager._plugins["route-policy-probe"].module is None
        # The route itself carries a child-process identity, proving JSON crossed the host boundary.
        routes, error = resolve_task_routes(
            [{"goal": "use hosted optional route", "option": "hosted"}], config, _parent(),
        )
        assert error is None
        route = routes[0]
        routed_pid = int(route.creds["model"].rsplit("-", 1)[1])
        assert routed_pid != os.getpid()
        assert route.routing_cfg["route_policy"] == {"required_endpoint_host": "hosted.invalid"}
        assert (route.creds["base_url"], route.creds["api_key"]) == (
            "https://hosted.invalid/v1", "hosted-secret",
        )
    finally:
        manager.unload()
        host.shutdown()
