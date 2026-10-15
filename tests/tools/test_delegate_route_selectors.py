"""Delegation selectors discovered as real plugins, with host-owned routing.

Profiles, config, secrets, ownership ledgers and SessionDB handles are real.
Only child construction and batch execution are replaced; no provider API is called.
"""

from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
import json
from pathlib import Path
from textwrap import dedent, indent
from types import SimpleNamespace

import hermes_yaml as yaml
import pytest

from agent import secret_scope
from hermes_cli import plugins
from hermes_cli.config import load_config_readonly
from hermes_constants import reset_hermes_home_override, set_hermes_home_override
from hermes_state import SessionDB
from tools import delegate_tool
from tools.delegate_tool_routes import resolve_delegation_route


# This list comes from the public selector-input contract, not the resolver's table.
ROUTE_FIELDS = {
    "model", "provider", "base_url", "api_key", "api_mode", "request_overrides",
    "command", "args", "acp_command", "acp_args", "fallback_providers",
}


def _normal_config(profile):
    return {
        "model": "normal-worker",
        "provider": "custom:normal",
        "base_url": "https://normal.example/v1",
        "api_key": "normal-route-secret",
        "api_mode": "codex_responses",
        "request_overrides": {"extra_body": {"normal_route": True}},
        "command": "unused-command",
        "args": ["--unused"],
        "acp_command": "unused-acp",
        "acp_args": ["--unused-acp"],
        "fallback_providers": [{"provider": "custom:normal", "model": f"fallback-{profile}"}],
        "reasoning_effort": "high",
        "max_iterations": 5,
        "max_spawn_depth": 1,
        "selector_policy": {"profile": profile, "layers": [{"mode": "original"}]},
    }


def _plugin(statement, registration=None):
    registration = registration or (
        'HANDLES.append(ctx.register_delegation_route_selector("candidate", select))'
    )
    return (
        "from copy import deepcopy\n"
        "from hermes_cli.delegation_routes import DelegationRouteSelection\n"
        "CALLS = []\nSEEN = []\nHANDLES = []\n\n"
        "def select(settings):\n"
        "    CALLS.append(settings['selector_policy']['profile'])\n"
        "    SEEN.append(deepcopy(settings))\n"
        + indent(dedent(statement).strip(), "    ")
        + "\n\ndef register(ctx):\n"
        + indent(dedent(registration).strip(), "    ")
        + "\n"
    )


@contextmanager
def _scope(home):
    home_token = set_hermes_home_override(str(home))
    multiplex_token = secret_scope.set_multiplex_context(True)
    secret_token = secret_scope.set_secret_scope(
        secret_scope.build_profile_secret_scope(home), profile_home=str(home)
    )
    try:
        yield
    finally:
        secret_scope.reset_secret_scope(secret_token)
        secret_scope.reset_multiplex_context(multiplex_token)
        reset_hermes_home_override(home_token)


@pytest.fixture
def profiles(tmp_path, monkeypatch):
    root = tmp_path / ".hermes"
    root.mkdir()
    bundled = tmp_path / "empty-bundled"
    bundled.mkdir()
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setenv("HERMES_BUNDLED_PLUGINS", str(bundled))
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.delenv("HERMES_PLUGIN_HOST_PROCESS", raising=False)
    # A profile without this secret must never borrow the launch process's key.
    monkeypatch.setenv("DEEPSEEK_API_KEY", "launch-process-secret")
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    plugins._reset_plugin_managers_for_tests()
    created = []

    def create(name="A", *, source, delegation=None, secrets="", isolation="in_process"):
        home = root / "profiles" / name
        directory = home / "plugins" / "route-probe"
        directory.mkdir(parents=True)
        (directory / "plugin.yaml").write_text(
            "name: route-probe\nversion: 1.0.0\n", encoding="utf-8"
        )
        (directory / "__init__.py").write_text(source, encoding="utf-8")
        cfg = deepcopy(_normal_config(name) if delegation is None else delegation)
        document = {
            "plugins": {"enabled": ["route-probe"], "isolation": isolation},
            "delegation": cfg,
            "custom_providers": [{
                "name": "normal", "base_url": "https://normal.example/v1",
                "api_key": "normal-provider-secret", "api_mode": "chat_completions",
            }],
        }
        (home / "config.yaml").write_text(yaml.safe_dump(document), encoding="utf-8")
        (home / ".env").write_text(secrets, encoding="utf-8")
        db = SessionDB(db_path=home / "state.db")
        session_id = f"parent-{name}"
        db.create_session(session_id, source="cli", profile_name=name, model_config={"owner": name})
        parent = SimpleNamespace(
            _session_db=db, session_id=session_id, _delegate_depth=0,
            model="parent-model", provider="custom", requested_provider="custom:normal",
            base_url="https://parent.example/v1", api_key="parent-route-secret",
            api_mode="codex_responses", request_overrides={"extra_body": {"parent_route": True}},
            reasoning_config={"enabled": True, "effort": "medium"},
            _fallback_chain=None, enabled_toolsets=[], disabled_toolsets=[],
            capabilities={"tool_search": True}, prefill_messages=None,
            tool_progress_callback=None, _safe_print=lambda _line: None,
        )
        value = SimpleNamespace(home=home, db=db, parent=parent, config=cfg)
        created.append(value)
        return value

    yield create
    plugins._reset_plugin_managers_for_tests()
    for profile in created:
        profile.db.close()


def _loaded():
    plugins.discover_plugins()
    manager = plugins.get_plugin_manager()
    loaded = manager._plugins["route-probe"]
    assert loaded.error is None
    assert loaded.module is not None
    return manager, loaded.module


def _assert_normal(cfg, route, creds):
    assert route is cfg
    assert creds["model"] == cfg["model"]
    assert creds["provider"] == "custom"
    assert creds["base_url"] == cfg["base_url"]
    assert creds["api_key"] == cfg["api_key"]
    assert creds["api_mode"] == cfg["api_mode"]
    assert creds["request_overrides"] == cfg["request_overrides"]


@pytest.fixture
def captured_dispatch(monkeypatch):
    import run_agent
    from hermes_state_registry import release_or_close

    children = []
    batches = []

    class Child:
        def __init__(self, **kwargs):
            self.kwargs = kwargs
            self.session_id = f"captured-child-{len(children)}"
            self._session_db = kwargs["session_db"]
            self._session_init_model_config = {}
            self.conversation_compressor = None
            self.tool_progress_callback = kwargs["tool_progress_callback"]
            children.append(self)

        def close(self):
            if self._session_db is not None:
                release_or_close(self._session_db)
                self._session_db = None

    def run_batch(batch, background):
        assert background is False
        batches.append(batch)
        for writer in batch.live_writers:
            writer.finalize({"status": "completed"})
        for _index, _task, child in batch.children:
            child.close()
        return "{}"

    monkeypatch.setattr(run_agent, "AIAgent", Child)
    monkeypatch.setattr(delegate_tool, "_run_batch", run_batch)
    yield children, batches
    for child in children:
        child.close()


def _dispatch(profile, captured_dispatch, **kwargs):
    children, _batches = captured_dispatch
    count = len(children)
    result = delegate_tool.delegate_task(
        goal="capture routing without requesting a provider API",
        parent_agent=profile.parent, background=False, **kwargs,
    )
    assert result == "{}"
    assert len(children) == count + 1
    return children[-1].kwargs


def test_selector_ledger_force_reload_and_unload_restore_normal_routing(profiles):
    profile = profiles(source=_plugin('return DelegationRouteSelection("selected-worker")'))
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        route, creds = resolve_delegation_route(cfg, profile.parent)
        manager, module = _loaded()
        handle = module.HANDLES[0]
        assert handle.kind == "delegation_route_selector"
        assert handle.plugin_key == "route-probe"
        assert handle.active
        assert handle in manager._ownership_ledger["route-probe"]
        assert route is not cfg and creds["model"] == "selected-worker"
        plugins.discover_plugins(force=True)
        _manager, reloaded = _loaded()
        assert not handle.active
        assert reloaded.HANDLES[0].active
        assert len(manager._delegation_route_selectors) == 1
        assert manager.unload("route-probe")
        assert not reloaded.HANDLES[0].active
        assert not manager._delegation_route_selectors
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))


@pytest.mark.parametrize("failure", [
    'raise RuntimeError("registration failed after acquisition")',
    'ctx.register_delegation_route_selector("candidate", select)',
    'ctx.register_delegation_route_selector("Invalid Name", select)',
    'ctx.register_delegation_route_selector("invalid", object())',
    'ctx.register_delegation_route_selector("invalid", async_select)',
])
def test_failed_registration_releases_already_acquired_selectors(profiles, failure):
    source = _plugin(
        'return DelegationRouteSelection("must-not-run")',
        'HANDLES.append(ctx.register_delegation_route_selector("candidate", select))\n' + failure,
    ) + "\nasync def async_select(settings):\n    return None\n"
    profile = profiles(source=source)
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
        manager = plugins.get_plugin_manager()
        assert manager._plugins["route-probe"].error is not None
        assert not manager._delegation_route_selectors
        assert not any(
            handle.active and handle.kind == "delegation_route_selector"
            for ledger in manager._ownership_ledger.values() for handle in ledger
        )


def test_selector_receives_credential_free_deep_copy_and_cannot_rewrite_config(profiles):
    source = _plugin('''
        settings["selector_policy"]["layers"][0]["mode"] = "plugin-mutated"
        settings["model"] = "injected-model"
        settings["api_key"] = "injected-secret"
        settings["fallback_providers"] = [{"model": "injected-fallback"}]
        return None
    ''')
    profile = profiles(source=source)
    config_before = (profile.home / "config.yaml").read_bytes()
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        cfg_before = deepcopy(cfg)
        assert ROUTE_FIELDS <= cfg.keys()
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
        _manager, module = _loaded()
        assert not ROUTE_FIELDS & module.SEEN[0].keys()
        assert module.SEEN[0]["selector_policy"] == cfg_before["selector_policy"]
        assert module.SEEN[0]["reasoning_effort"] == cfg_before["reasoning_effort"]
        assert cfg == cfg_before
        assert load_config_readonly()["delegation"] == cfg_before
    assert (profile.home / "config.yaml").read_bytes() == config_before


def _ordered_plugin(statement):
    return _plugin(statement, '''
        ctx.register_delegation_route_selector("abstain", abstain)
        ctx.register_delegation_route_selector("candidate", select)
        ctx.register_delegation_route_selector("later", later)
    ''') + '''
def abstain(settings):
    CALLS.append("abstain")
    return None

def later(settings):
    CALLS.append("later")
    return DelegationRouteSelection("later-worker")
'''


@pytest.mark.parametrize(("statement", "secrets"), [
    ('return {"model": "worker", "api_key": "selector-secret"}', ""),
    ('return DelegationRouteSelection("   ")', ""),
    ('return DelegationRouteSelection("worker", provider=42)', ""),
    ('return DelegationRouteSelection("worker", api_mode="selector-secret")', ""),
    ('return DelegationRouteSelection("worker", required_endpoint_host="https://selector-secret")', ""),
    ('return DelegationRouteSelection("worker", provider="selector-secret")', ""),
    ('raise RuntimeError("selector-secret")', ""),
    ('return DelegationRouteSelection("worker", provider="deepseek")', ""),
    ('return DelegationRouteSelection("worker", provider="deepseek", required_endpoint_host="api.deepseek.com")',
     "DEEPSEEK_API_KEY=selector-secret\nDEEPSEEK_BASE_URL=https://proxy.example/v1\n"),
])
def test_unavailable_first_proposal_falls_back_without_later_selector_or_secret_logs(
    profiles, caplog, statement, secrets,
):
    profile = profiles(source=_ordered_plugin(statement), secrets=secrets)
    caplog.set_level("DEBUG", logger="tools.delegate_tool")
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
        _manager, module = _loaded()
        assert module.CALLS == ["abstain", "A"]
        warnings = [r for r in caplog.records
                    if r.name == "tools.delegate_tool" and r.levelname == "WARNING"]
        assert len(warnings) == 1
        assert "selector candidate unavailable" in warnings[0].getMessage()
        assert warnings[0].exc_info is None
        assert "selector-secret" not in caplog.text
        assert "launch-process-secret" not in caplog.text
        assert "normal-route-secret" not in caplog.text


def test_none_abstains_and_first_valid_selection_wins_registration_order(profiles):
    profile = profiles(source=_ordered_plugin('return DelegationRouteSelection("selected-worker")'))
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        route, creds = resolve_delegation_route(cfg, profile.parent)
        _manager, module = _loaded()
        assert module.CALLS == ["abstain", "A"]
        assert route["model"] == creds["model"] == "selected-worker"
        assert route["fallback_providers"] == cfg["fallback_providers"]


def test_profile_local_dispatch_uses_real_scoped_credentials_and_route_owner(
    profiles, captured_dispatch,
):
    source = _plugin('''
        return DelegationRouteSelection(
            "selected-" + settings["selector_policy"]["profile"], provider="deepseek")
    ''')
    a = profiles("A", source=source, secrets="DEEPSEEK_API_KEY=profile-A-secret\n")
    b = profiles("B", source=source, secrets="DEEPSEEK_API_KEY=profile-B-secret\n")
    modules = {}
    managers = {}
    for profile, tag in ((a, "A"), (b, "B"), (a, "A")):
        before = (profile.home / "config.yaml").read_bytes()
        with _scope(profile.home):
            kwargs = _dispatch(profile, captured_dispatch)
            managers[tag], modules[tag] = _loaded()
            assert kwargs["model"] == f"selected-{tag}"
            assert kwargs["provider"] == "deepseek"
            assert kwargs["api_key"] == f"profile-{tag}-secret"
            assert kwargs["base_url"] == "https://api.deepseek.com/v1"
            assert kwargs["api_mode"] == "chat_completions"
            assert kwargs["request_overrides"] == {}
            assert kwargs["fallback_model"] == profile.config["fallback_providers"]
            assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
            assert kwargs["capabilities"] is None
            assert kwargs["parent_session_id"] == profile.parent.session_id
            assert kwargs["session_db"] is not profile.db
            assert Path(kwargs["session_db"].db_path) == profile.home / "state.db"
            assert json.loads(profile.db.get_session(profile.parent.session_id)["model_config"]) == {"owner": tag}
            assert profile.parent.api_key == "parent-route-secret"
            assert (profile.home / "config.yaml").read_bytes() == before
    assert managers["A"] is not managers["B"]
    assert modules["A"] is not modules["B"]
    assert modules["A"].CALLS == ["A", "A"]
    assert modules["B"].CALLS == ["B"]
    with _scope(a.home):
        assert managers["A"].unload("route-probe")
        assert _dispatch(a, captured_dispatch)["model"] == "normal-worker"
    with _scope(b.home):
        assert _dispatch(b, captured_dispatch)["model"] == "selected-B"


@pytest.mark.parametrize(("mode", "expected_mode"), [
    (None, "chat_completions"), ("anthropic_messages", "anthropic_messages"),
])
def test_inherited_selection_keeps_parent_bundle_owner_overrides_and_fallback(
    profiles, captured_dispatch, mode, expected_mode,
):
    cfg = {key: value for key, value in _normal_config("A").items()
           if key not in ROUTE_FIELDS or key == "fallback_providers"}
    cfg["request_overrides"] = {"extra_body": {"delegation_route": True}}
    profile = profiles(
        source=_plugin(f'return DelegationRouteSelection("inherited-worker", api_mode={mode!r})'),
        delegation=cfg,
    )
    with _scope(profile.home):
        config_before = deepcopy(load_config_readonly()["delegation"])
        document_before = (profile.home / "config.yaml").read_bytes()
        kwargs = _dispatch(profile, captured_dispatch)
        assert kwargs["model"] == "inherited-worker"
        assert kwargs["provider"] == profile.parent.provider
        assert kwargs["requested_provider"] == profile.parent.requested_provider
        assert kwargs["base_url"] == profile.parent.base_url
        assert kwargs["api_key"] == profile.parent.api_key
        assert kwargs["api_mode"] == expected_mode
        assert kwargs["capabilities"] == profile.parent.capabilities
        assert kwargs["request_overrides"] == {
            "extra_body": {"parent_route": True, "delegation_route": True},
        }
        assert kwargs["fallback_model"] == cfg["fallback_providers"]
        assert load_config_readonly()["delegation"] == config_before
        assert (profile.home / "config.yaml").read_bytes() == document_before


@pytest.mark.parametrize("empty", [False, True])
def test_internal_credentials_owner_bypasses_selectors_and_owns_its_fallback(
    profiles, captured_dispatch, empty,
):
    profile = profiles(source=_plugin('return DelegationRouteSelection("plugin-worker")'))
    internal = {} if empty else {
        "model": "review-worker", "base_url": "https://review.example/v1",
        "api_key": "review-route-secret", "api_mode": "anthropic_messages",
        "request_overrides": {"extra_body": {"review_route": True}},
        "fallback_providers": [{"provider": "custom:normal", "model": "review-fallback"}],
    }
    before = deepcopy(internal)
    with _scope(profile.home):
        assert _dispatch(profile, captured_dispatch)["model"] == "plugin-worker"
        _manager, module = _loaded()
        kwargs = _dispatch(profile, captured_dispatch, credentials_cfg=internal)
        assert module.CALLS == ["A"]
        if empty:
            assert kwargs["model"] == profile.parent.model
            assert kwargs["base_url"] == profile.parent.base_url
            assert kwargs["api_key"] == profile.parent.api_key
            assert kwargs["api_mode"] == profile.parent.api_mode
            assert kwargs["fallback_model"] is None
        else:
            assert kwargs["model"] == internal["model"]
            assert kwargs["base_url"] == internal["base_url"]
            assert kwargs["api_key"] == internal["api_key"]
            assert kwargs["api_mode"] == internal["api_mode"]
            assert kwargs["request_overrides"] == internal["request_overrides"]
            assert kwargs["fallback_model"] == internal["fallback_providers"]
        assert internal == before


@pytest.mark.parametrize("action", ["list", "steer", "stop"])
def test_control_actions_do_not_select_routes_or_construct_children(
    profiles, captured_dispatch, action,
):
    profile = profiles(source=_plugin('return DelegationRouteSelection("plugin-worker")'))
    with _scope(profile.home):
        _manager, module = _loaded()
        result = json.loads(delegate_tool.delegate_task(
            action=action, goal="ignored for control", subagent_id="not-live",
            message="control message", parent_agent=profile.parent,
        ))
        assert module.CALLS == []
        assert captured_dispatch == ([], [])
        if action == "list":
            assert result["subagents"] == []
        else:
            assert "No live subagent" in result["error"]


@pytest.mark.platforms("any")  # the real plugin host is a child process
def test_host_isolation_explicitly_refuses_in_process_route_selectors(profiles):
    profile = profiles(
        source=_plugin('return DelegationRouteSelection("must-not-run")'), isolation="host",
    )
    with _scope(profile.home):
        manager = plugins.get_plugin_manager()
        try:
            cfg = load_config_readonly()["delegation"]
            _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
            loaded = manager._plugins["route-probe"]
            assert "register_delegation_route_selector" in str(loaded.error)
            assert "cannot run in the plugin host" in str(loaded.error)
            assert loaded.module is None
            assert not manager._delegation_route_selectors
        finally:
            host = getattr(manager, "_plugin_host_instance", None)
            if host is not None:
                host.shutdown()


def _recoverable_config():
    return {
        "model": "normal-worker", "provider": "custom:normal",
        "base_url": "https://normal.example/v1", "api_key": "normal-route-secret",
        "api_mode": "codex_responses", "max_iterations": 5,
        "request_overrides": {"extra_headers": {"X-Route": "normal"}},
        "fallback_providers": [{"provider": "custom:normal", "model": "last-backup"}],
        "selector_policy": {"profile": "recovery-test"},
    }


def test_optional_route_prepends_exact_configured_backup_without_mutating_profile(profiles, captured_dispatch):
    profile = profiles(
        source=_plugin('return DelegationRouteSelection("cheap-worker", api_mode="chat_completions", fallback_to_configured=True)'),
        delegation=_recoverable_config(),
    )
    before = (profile.home / "config.yaml").read_bytes()
    with _scope(profile.home):
        kwargs = _dispatch(profile, captured_dispatch)
    assert kwargs["model"] == "cheap-worker"
    assert kwargs["api_mode"] == "chat_completions"
    backup, last = kwargs["fallback_model"]
    assert (backup["model"], backup["base_url"], backup["api_key"], backup["api_mode"]) == (
        "normal-worker", "https://normal.example/v1", "normal-route-secret", "codex_responses",
    )
    assert backup["request_overrides"] == {"extra_headers": {"X-Route": "normal"}}
    assert last["model"] == "last-backup"
    child = captured_dispatch[0][-1]
    assert child._selected_route_recovery is not None
    assert child._api_outage_max_wait_seconds == 600
    assert (profile.home / "config.yaml").read_bytes() == before


def test_route_failure_cools_only_own_profile_and_recovery_is_single_flight(profiles, monkeypatch):
    from agent import delegation_route_recovery as recovery
    from agent.delegation_route_health import RouteHealthRegistry
    from agent.error_classifier import FailoverReason

    now = [0.0]
    monkeypatch.setattr(recovery, "ROUTE_HEALTH", RouteHealthRegistry(clock=lambda: now[0]))
    source = _plugin('return DelegationRouteSelection("cheap-worker", fallback_to_configured=True)')
    profile = profiles(source=source, delegation=_recoverable_config())
    other = profiles("B", source=source, delegation=_recoverable_config())
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        route, creds = resolve_delegation_route(cfg, profile.parent)
        child = SimpleNamespace(**creds, _selected_route_recovery=route[recovery.POLICY_KEY])
        recovery.record_route_failure(child, FailoverReason.server_error)
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
    with _scope(other.home):
        _route, creds = resolve_delegation_route(load_config_readonly()["delegation"], other.parent)
        assert creds["model"] == "cheap-worker"
    now[0] = 301.0
    with _scope(profile.home):
        route, creds = resolve_delegation_route(cfg, profile.parent)
        assert creds["model"] == "cheap-worker"
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
        recovered = SimpleNamespace(**creds, _selected_route_recovery=route[recovery.POLICY_KEY])
        recovery.record_route_success(recovered, SimpleNamespace(content="ready", tool_calls=None))
        assert resolve_delegation_route(cfg, profile.parent)[1]["model"] == "cheap-worker"


@pytest.mark.parametrize("policy", ["fallback_to_configured=1", "failure_cooldown_seconds=-1"])
def test_invalid_optional_recovery_policy_retains_configured_route(profiles, policy):
    profile = profiles(source=_plugin(f'return DelegationRouteSelection("cheap-worker", {policy})'))
    with _scope(profile.home):
        cfg = load_config_readonly()["delegation"]
        _assert_normal(cfg, *resolve_delegation_route(cfg, profile.parent))
