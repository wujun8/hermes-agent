"""Behavioral invariants for profile-scoped DeepSeek off-peak delegation."""

from copy import deepcopy
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import pytest

from tools.delegate_off_peak import is_deepseek_off_peak


@pytest.mark.parametrize(
    ("instant", "expected", "warns"),
    [
        ("2026-09-28T08:59:00+08:00", True, False),
        ("2026-09-28T09:00:00+08:00", False, False),
        ("2026-09-28T11:59:00+08:00", False, False),
        ("2026-09-28T12:00:00+08:00", True, False),
        ("2026-09-28T13:59:00+08:00", True, False),
        ("2026-09-28T14:00:00+08:00", False, False),
        ("2026-09-28T17:59:00+08:00", False, False),
        ("2026-09-28T18:00:00+08:00", True, False),
        # UTC instants are converted to Beijing time before comparing boundaries.
        ("2026-09-28T00:59:00+00:00", True, False),
        ("2026-09-28T01:00:00+00:00", False, False),
        ("2026-09-28T10:00:00+00:00", True, False),
        # 2026-10-10 is a Saturday make-up workday; weekends remain off-peak.
        ("2026-10-10T10:00:00+08:00", True, False),
        # National Day is a weekday holiday, so its normal peak hours are discounted.
        ("2026-10-01T10:00:00+08:00", True, False),
        # Holiday data does not cover this ordinary weekday in 2030.
        ("2030-01-07T10:00:00+08:00", False, True),
        ("2030-01-07T08:59:00+08:00", True, False),
    ],
)
def test_deepseek_off_peak_uses_beijing_billing_windows(instant, expected, warns, caplog):
    from tools.delegate_off_peak import logger

    caplog.set_level("WARNING", logger=logger.name)
    assert is_deepseek_off_peak(datetime.fromisoformat(instant)) is expected
    warning_seen = "holiday calendar unavailable" in caplog.text
    assert warning_seen is warns


def test_deepseek_off_peak_rejects_naive_datetimes():
    with pytest.raises(ValueError, match="timezone-aware"):
        is_deepseek_off_peak(datetime(2026, 9, 28, 10, 0))


def test_new_dispatch_uses_only_the_active_profiles_native_deepseek_route(
    tmp_path, monkeypatch, caplog
):
    """Resolve real profile config/secrets and inspect the route passed to child construction."""
    import yaml

    import hermes_cli.config as hermes_config
    import hermes_constants
    import run_agent
    from tools.registry import registry
    import tools.delegate_off_peak as off_peak
    import tools.delegate_tool as delegate_tool
    import tools.delegate_tool_config as delegate_config
    from agent import secret_scope

    class OffPeakNow(datetime):
        @classmethod
        def now(cls, tz=None):
            # Noon is the start of the off-peak lunch window and needs no
            # holiday lookup, so the integration test also runs without API access.
            return cls(2026, 9, 28, 12, 0, tzinfo=tz)

    class PeakNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 28, 10, 0, tzinfo=tz)

    fallback = [{"provider": "deepseek", "model": "deepseek-chat"}]
    delegation_config = {
        "model": "base-worker",
        "provider": "custom",
        "base_url": "https://baseline.example/v1",
        "api_key": "baseline-profile-key",
        "api_mode": "codex_responses",
        "request_overrides": {"extra_body": {"baseline_marker": True}},
        "command": "old-command",
        "args": ["--old"],
        "acp_command": "old-acp-command",
        "acp_args": ["--old-acp"],
        "reasoning_effort": "high",
        "fallback_providers": fallback,
    }
    config_document = {
        "model": {"default": "parent-model", "provider": "openrouter"},
        "delegation": delegation_config,
    }
    config_document["delegation"]["max_spawn_depth"] = 2
    root = tmp_path / ".hermes"
    home_a = root / "profiles" / "profile-a"
    home_b = root / "profiles" / "profile-b"
    for home in (home_a, home_b):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump(config_document), encoding="utf-8")
    (home_a / ".env").write_text("DEEPSEEK_API_KEY=fake-profile-a-key\n", encoding="utf-8")
    (home_b / ".env").write_text("", encoding="utf-8")

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("DEEPSEEK_BASE_URL", raising=False)
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setattr(off_peak, "datetime", OffPeakNow)

    parent = SimpleNamespace(
        # Model-facing registry dispatches are backgrounded for a top-level parent.
        # At depth one, the registry uses the synchronous path while one child level remains.
        _delegate_depth=1,
        model="parent-model",
        provider="openrouter",
        base_url="https://openrouter.example/v1",
        api_key="parent-process-key",
        api_mode="chat_completions",
        request_overrides={"extra_body": {"parent_marker": True}},
        reasoning_config={"enabled": True, "effort": "medium"},
        _fallback_chain=None,
        session_id="parent-session",
        prefill_messages=None,
        tool_progress_callback=None,
        _safe_print=lambda _line: None,
    )

    constructed_agents = []
    batches = []

    class AgentStub:
        def __init__(self, **kwargs):
            self.constructor_kwargs = kwargs
            self.session_id = f"child-{len(constructed_agents)}"
            self.conversation_compressor = None
            self._session_init_model_config = None
            self.tool_progress_callback = kwargs.get("tool_progress_callback")
            constructed_agents.append(self)

    def capture_batch(batch, _background):
        batches.append(batch)
        for writer in batch.live_writers:
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        return "{}"

    monkeypatch.setattr(run_agent, "AIAgent", AgentStub)
    monkeypatch.setattr(delegate_tool, "_resolve_child_toolsets", lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(delegate_tool, "_open_child_session_db", lambda _parent: None)
    monkeypatch.setattr(delegate_tool, "_resolve_child_credential_pool", lambda *_a, **_kw: None)
    monkeypatch.setattr(delegate_tool, "_attach_child", lambda *_a, **_kw: None)
    monkeypatch.setattr(delegate_tool, "_run_batch", capture_batch)

    prior_multiplex = secret_scope.is_multiplex_active()
    prior_auto_pin = secret_scope._AUTO_PINNED_HOME
    prior_process_pin = hermes_constants._PINNED_PROCESS_HERMES_HOME
    secret_scope.set_multiplex_active(True)

    def bind_profile(home: Path):
        monkeypatch.setenv("HERMES_HOME", str(home))
        scope = secret_scope.build_profile_secret_scope(home)
        return secret_scope.set_secret_scope(scope, profile_home=str(home))

    def dispatch(*, credentials_cfg=None):
        if credentials_cfg is None:
            result = registry.dispatch(
                "delegate_task",
                {"goal": "record the child route without making a provider request"},
                parent_agent=parent,
            )
        else:
            # This trusted internal parameter is intentionally absent from the
            # model-facing registry schema.
            result = delegate_tool.delegate_task(
                goal="record the internal route without making a provider request",
                parent_agent=parent,
                credentials_cfg=credentials_cfg,
            )
        assert result == "{}"
        return constructed_agents[-1].constructor_kwargs

    try:
        # A -> B -> A is intentional: the real loader and provider resolver must not cache A's key for B.
        token = bind_profile(home_a)
        try:
            cfg = hermes_config.load_config_readonly()["delegation"]
            cfg_before = deepcopy(cfg)
            parent_before = deepcopy(vars(parent))
            assert cfg["deepseek_off_peak_enabled"] is True
            route, creds = delegate_config._resolve_delegation_route(cfg, parent)
            assert route is not cfg
            assert cfg == cfg_before
            assert vars(parent) == parent_before
            assert route["provider"] == "deepseek"
            assert route["model"] == "deepseek-flash"
            assert route["api_mode"] == "chat_completions"
            assert route["fallback_providers"] == fallback
            assert route["reasoning_effort"] == "high"
            assert not {
                "base_url", "api_key", "request_overrides", "command", "args",
                "acp_command", "acp_args",
            } & route.keys()
            assert creds["provider"] == "deepseek"
            assert creds["model"] == "deepseek-flash"
            assert creds["api_key"] == "fake-profile-a-key"
            assert creds["base_url"] == "https://api.deepseek.com/v1"
            assert creds["api_mode"] == "chat_completions"
            assert creds["request_overrides"] == {}

            child_kwargs_a1 = dispatch()
            assert child_kwargs_a1["provider"] == "deepseek"
            assert child_kwargs_a1["model"] == "deepseek-flash"
            assert child_kwargs_a1["api_key"] == "fake-profile-a-key"
            assert child_kwargs_a1["base_url"] == "https://api.deepseek.com/v1"
            assert child_kwargs_a1["api_mode"] == "chat_completions"
            assert child_kwargs_a1["request_overrides"] == {}
            assert child_kwargs_a1["reasoning_config"] == {"enabled": True, "effort": "high"}
            assert child_kwargs_a1["fallback_model"] == fallback

            # Both supported false spellings keep the configured route despite an off-peak instant.
            for disabled in (False, "false"):
                disabled_cfg = {**cfg, "deepseek_off_peak_enabled": disabled}
                disabled_route, disabled_creds = delegate_config._resolve_delegation_route(
                    disabled_cfg, parent,
                )
                assert disabled_route is disabled_cfg
                assert disabled_creds["provider"] == "custom"
                assert disabled_creds["model"] == "base-worker"
                assert disabled_creds["base_url"] == "https://baseline.example/v1"
                assert disabled_creds["api_key"] == "baseline-profile-key"

            # The internal credentials_cfg owner opts out even when the profile has a DeepSeek key.
            child_kwargs_pinned = dispatch(credentials_cfg={
                "model": "review-worker",
                "provider": "custom",
                "base_url": "https://review.example/v1",
                "api_key": "review-route-key",
                "api_mode": "chat_completions",
            })
            assert child_kwargs_pinned["provider"] == "custom"
            assert child_kwargs_pinned["model"] == "review-worker"
            assert child_kwargs_pinned["base_url"] == "https://review.example/v1"
            assert child_kwargs_pinned["api_key"] == "review-route-key"
        finally:
            secret_scope.reset_secret_scope(token)

        token = bind_profile(home_b)
        try:
            caplog.set_level("WARNING", logger="tools.delegate_tool")
            child_kwargs_b = dispatch()
            assert child_kwargs_b["provider"] == "custom"
            assert child_kwargs_b["model"] == "base-worker"
            assert child_kwargs_b["base_url"] == "https://baseline.example/v1"
            assert child_kwargs_b["api_key"] == "baseline-profile-key"
            assert "DeepSeek off-peak route unavailable" in caplog.text

            # A configured third-party DeepSeek endpoint is not eligible for the discounted route.
            (home_b / ".env").write_text(
                "DEEPSEEK_API_KEY=fake-profile-b-key\nDEEPSEEK_BASE_URL=https://proxy.example/v1\n",
                encoding="utf-8",
            )
            third_party_scope = secret_scope.build_profile_secret_scope(home_b)
            secret_scope.reset_secret_scope(token)
            token = secret_scope.set_secret_scope(third_party_scope, profile_home=str(home_b))
            caplog.clear()
            route_b, creds_b = delegate_config._resolve_delegation_route(
                hermes_config.load_config_readonly()["delegation"], parent,
            )
            assert route_b["provider"] == "custom"
            assert creds_b["base_url"] == "https://baseline.example/v1"
            assert "DeepSeek off-peak route unavailable" in caplog.text
            assert "fake-profile-b-key" not in caplog.text
        finally:
            secret_scope.reset_secret_scope(token)

        token = bind_profile(home_a)
        try:
            # Returning to A re-reads A's .env through its own scope after B's failed lookup.
            child_kwargs_a2 = dispatch()
            assert child_kwargs_a2["provider"] == "deepseek"
            assert child_kwargs_a2["api_key"] == "fake-profile-a-key"

            # At a weekday peak instant, the default-enabled profile keeps its configured route.
            monkeypatch.setattr(off_peak, "datetime", PeakNow)
            cfg_peak = hermes_config.load_config_readonly()["delegation"]
            route_peak, creds_peak = delegate_config._resolve_delegation_route(cfg_peak, parent)
            assert route_peak is cfg_peak
            assert creds_peak["provider"] == "custom"
            assert creds_peak["base_url"] == "https://baseline.example/v1"
        finally:
            secret_scope.reset_secret_scope(token)
    finally:
        secret_scope.set_multiplex_active(prior_multiplex)
        secret_scope._AUTO_PINNED_HOME = prior_auto_pin
        hermes_constants.pin_process_hermes_home(prior_process_pin)
        hermes_config._LOAD_CONFIG_CACHE.clear()


def test_inherit_off_peak_keeps_the_profiles_named_supplier_route(tmp_path, monkeypatch, caplog):
    """The cheaper model stays on each profile's normal provider, endpoint, and key."""
    import yaml

    import hermes_cli.config as hermes_config
    import run_agent
    from tools.registry import registry
    import tools.delegate_off_peak as off_peak
    import tools.delegate_tool as delegate_tool
    import tools.delegate_tool_config as delegate_config
    from agent import secret_scope
    from hermes_cli import runtime_provider

    class OffPeakNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 28, 12, 0, tzinfo=tz)

    class PeakNow(datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 9, 28, 10, 0, tzinfo=tz)

    endpoint = "https://supplier.example/v1"
    fallback = [{"provider": "deepseek", "model": "deepseek-chat"}]
    profile_config = {
        "model": {"default": "gpt-6.1-sol", "provider": "custom:custom-codex"},
        "custom_providers": [
            {
                "name": "custom-codex",
                "base_url": endpoint,
                "key_env": "CODEX_NEW_API_KEY",
                "api_mode": "codex_responses",
                "extra_body": {"supplier_marker": "same-route"},
            },
            {
                "name": "flash-fast",
                "base_url": "https://flash-provider.example/v1",
                "key_env": "FLASH_PROVIDER_API_KEY",
                "api_mode": "codex_responses",
                "extra_body": {"low_provider_marker": "own-route"},
            },
        ],
        "delegation": {
            "model": "gpt-6-luna",
            "provider": "custom:custom-codex",
            "deepseek_off_peak_provider": "inherit",
            "deepseek_off_peak_model": "deepseek-flash",
            "deepseek_off_peak_api_mode": "chat_completions",
            "fallback_providers": fallback,
            "reasoning_effort": "high",
            "max_spawn_depth": 2,
        },
    }
    root = tmp_path / ".hermes"
    home_a = root / "profiles" / "profile-a"
    home_b = root / "profiles" / "profile-b"
    for home, key in ((home_a, "profile-a-supplier-key"), (home_b, "profile-b-supplier-key")):
        home.mkdir(parents=True)
        (home / "config.yaml").write_text(yaml.safe_dump(profile_config), encoding="utf-8")
        (home / ".env").write_text(
            f"CODEX_NEW_API_KEY={key}\nFLASH_PROVIDER_API_KEY={home.name}-flash-key\n",
            encoding="utf-8",
        )

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    monkeypatch.delenv("HERMES_IGNORE_USER_CONFIG", raising=False)
    monkeypatch.setenv("CODEX_NEW_API_KEY", "launch-profile-must-not-leak")
    monkeypatch.setattr(off_peak, "datetime", OffPeakNow)
    provider_calls = []
    resolve_runtime_provider = runtime_provider.resolve_runtime_provider

    def track_runtime_provider(**kwargs):
        provider_calls.append((kwargs.get("requested"), kwargs.get("target_model")))
        return resolve_runtime_provider(**kwargs)

    monkeypatch.setattr(runtime_provider, "resolve_runtime_provider", track_runtime_provider)
    parent = SimpleNamespace(
        _delegate_depth=1,
        model="gpt-6.1-sol",
        provider="custom",
        requested_provider="custom:custom-codex",
        base_url=endpoint,
        api_key="parent-main-key",
        api_mode="codex_responses",
        request_overrides={},
        reasoning_config={"enabled": True, "effort": "medium"},
        _fallback_chain=None,
        session_id="parent-session",
        prefill_messages=None,
        tool_progress_callback=None,
        _safe_print=lambda _line: None,
    )
    constructed_agents = []

    class AgentStub:
        def __init__(self, **kwargs):
            self.constructor_kwargs = kwargs
            self.session_id = f"child-{len(constructed_agents)}"
            self.conversation_compressor = None
            self._session_init_model_config = None
            self.tool_progress_callback = kwargs.get("tool_progress_callback")
            constructed_agents.append(self)

    def capture_batch(batch, _background):
        for writer in batch.live_writers:
            close = getattr(writer, "close", None)
            if callable(close):
                close()
        return "{}"

    monkeypatch.setattr(run_agent, "AIAgent", AgentStub)
    monkeypatch.setattr(delegate_tool, "_resolve_child_toolsets", lambda *_a, **_kw: ([], []))
    monkeypatch.setattr(delegate_tool, "_open_child_session_db", lambda _parent: None)
    monkeypatch.setattr(delegate_tool, "_resolve_child_credential_pool", lambda *_a, **_kw: None)
    monkeypatch.setattr(delegate_tool, "_attach_child", lambda *_a, **_kw: None)
    monkeypatch.setattr(delegate_tool, "_run_batch", capture_batch)

    prior_multiplex = secret_scope.is_multiplex_active()
    prior_auto_pin = secret_scope._AUTO_PINNED_HOME
    secret_scope.set_multiplex_active(True)

    def bind_profile(home: Path):
        monkeypatch.setenv("HERMES_HOME", str(home))
        scope = secret_scope.build_profile_secret_scope(home)
        return secret_scope.set_secret_scope(scope, profile_home=str(home))

    def dispatch(expected_key, *, expected_model, expected_api_mode):
        result = registry.dispatch(
            "delegate_task",
            {"goal": "record the child route without making a provider request"},
            parent_agent=parent,
        )
        assert result == "{}"
        kwargs = constructed_agents[-1].constructor_kwargs
        assert kwargs["model"] == expected_model
        assert kwargs["provider"] == "custom:custom-codex"
        assert kwargs["requested_provider"] == "custom:custom-codex"
        assert kwargs["base_url"] == endpoint
        assert kwargs["api_key"] == expected_key
        assert kwargs["api_mode"] == expected_api_mode
        assert kwargs["request_overrides"] == {"extra_body": {"supplier_marker": "same-route"}}
        assert kwargs["fallback_model"] == fallback
        assert kwargs["reasoning_config"] == {"enabled": True, "effort": "high"}
        return kwargs

    try:
        for home, expected_key in (
            (home_a, "profile-a-supplier-key"),
            (home_b, "profile-b-supplier-key"),
            (home_a, "profile-a-supplier-key"),
        ):
            token = bind_profile(home)
            try:
                cfg = hermes_config.load_config_readonly()["delegation"]
                cfg_before = deepcopy(cfg)
                parent_before = deepcopy(vars(parent))
                provider_calls.clear()
                route, creds = delegate_config._resolve_delegation_route(cfg, parent)
                assert route is not cfg
                assert route["provider"] == "custom:custom-codex"
                assert route["model"] == "deepseek-flash"
                assert route["api_mode"] == "chat_completions"
                assert creds["provider"] == "custom:custom-codex"
                assert creds["model"] == "deepseek-flash"
                assert creds["base_url"] == endpoint
                assert creds["api_key"] == expected_key
                assert creds["api_mode"] == "chat_completions"
                assert creds["request_overrides"] == {"extra_body": {"supplier_marker": "same-route"}}
                assert provider_calls == [("custom:custom-codex", "deepseek-flash")]
                assert cfg == cfg_before
                assert vars(parent) == parent_before

                dispatch(expected_key, expected_model="deepseek-flash", expected_api_mode="chat_completions")
                assert provider_calls == [
                    ("custom:custom-codex", "deepseek-flash"),
                    ("custom:custom-codex", "deepseek-flash"),
                ]
                assert cfg == cfg_before
                assert vars(parent) == parent_before
            finally:
                secret_scope.reset_secret_scope(token)

        token = bind_profile(home_a)
        try:
            pure_cfg = {
                **hermes_config.load_config_readonly()["delegation"],
                "provider": "",
                "model": "",
                "base_url": "",
                "api_key": "",
                "api_mode": "",
                "deepseek_off_peak_provider": "inherit",
                "deepseek_off_peak_api_mode": "",
            }
            pure_parent = SimpleNamespace(**vars(parent))
            pure_parent.api_mode = "anthropic_messages"
            provider_calls.clear()
            pure_route, pure_creds = delegate_config._resolve_delegation_route(pure_cfg, pure_parent)
            assert pure_creds["model"] == "deepseek-flash"
            assert pure_creds["provider"] is None
            assert pure_creds["base_url"] is None
            assert pure_creds["api_key"] is None
            assert provider_calls == [("custom:custom-codex", "deepseek-flash")]
            pure_child = delegate_config._resolve_child_runtime(
                pure_parent,
                pure_route,
                pure_parent.api_key,
                model=pure_creds["model"],
                override_provider=pure_creds["provider"],
                override_base_url=pure_creds["base_url"],
                override_api_key=pure_creds["api_key"],
                override_api_mode=pure_creds["api_mode"],
                override_acp_command=None,
                override_acp_args=None,
                routing_cfg=pure_route,
            )
            assert pure_child["provider"] == "custom"
            assert pure_child["requested_provider"] == "custom:custom-codex"
            assert pure_child["model"] == "deepseek-flash"
            assert pure_child["base_url"] == endpoint
            assert pure_child["api_key"] == "parent-main-key"
            assert pure_child["api_mode"] == "codex_responses"
        finally:
            secret_scope.reset_secret_scope(token)

        token = bind_profile(home_a)
        try:
            named_cfg = {
                **hermes_config.load_config_readonly()["delegation"],
                "deepseek_off_peak_provider": "custom:flash-fast",
                "base_url": "https://normal-override.example/v1",
                "api_key": "normal-route-secret",
                "api_mode": "codex_responses",
                "request_overrides": {"extra_body": {"normal-route-marker": True}},
                "command": "normal-command",
                "args": ["--normal"],
                "acp_command": "normal-acp",
                "acp_args": ["--normal-acp"],
            }
            provider_calls.clear()
            named_route, named_creds = delegate_config._resolve_delegation_route(named_cfg, parent)
            assert named_route["provider"] == "custom:flash-fast"
            assert named_route["model"] == "deepseek-flash"
            assert named_route["api_mode"] == "chat_completions"
            assert not {
                "base_url", "api_key", "request_overrides", "command", "args", "acp_command", "acp_args",
            } & named_route.keys()
            assert named_creds["provider"] == "custom:flash-fast"
            assert named_creds["base_url"] == "https://flash-provider.example/v1"
            assert named_creds["api_key"] == "profile-a-flash-key"
            assert named_creds["api_mode"] == "chat_completions"
            assert named_creds["request_overrides"] == {"extra_body": {"low_provider_marker": "own-route"}}
            assert provider_calls == [("custom:flash-fast", "deepseek-flash")]
        finally:
            secret_scope.reset_secret_scope(token)

        token = bind_profile(home_a)
        try:
            monkeypatch.setattr(off_peak, "datetime", PeakNow)
            provider_calls.clear()
            dispatch("profile-a-supplier-key", expected_model="gpt-6-luna", expected_api_mode="codex_responses")
            assert provider_calls == [("custom:custom-codex", "gpt-6-luna")]

            monkeypatch.setattr(off_peak, "datetime", OffPeakNow)
            invalid_cfg = {
                **hermes_config.load_config_readonly()["delegation"],
                "deepseek_off_peak_api_mode": "unregistered-wire-protocol",
            }
            provider_calls.clear()
            caplog.set_level("WARNING", logger="tools.delegate_tool")
            normal_route, normal_creds = delegate_config._resolve_delegation_route(invalid_cfg, parent)
            assert normal_route is invalid_cfg
            assert normal_creds["model"] == "gpt-6-luna"
            assert normal_creds["api_mode"] == "codex_responses"
            assert provider_calls == [("custom:custom-codex", "gpt-6-luna")]
            assert "DeepSeek off-peak route unavailable" in caplog.text
        finally:
            secret_scope.reset_secret_scope(token)
    finally:
        secret_scope.set_multiplex_active(prior_multiplex)
        secret_scope._AUTO_PINNED_HOME = prior_auto_pin
        hermes_config._LOAD_CONFIG_CACHE.clear()
