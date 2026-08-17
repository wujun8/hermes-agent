"""Tests for GatewayRunner._resolve_profile_home_for_source — profile resolution logic."""

import logging
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from gateway.session import SessionSource, build_session_key
from gateway.run import (
    GatewayRunner,
    ProfileRouteRejectedError,
    SecondaryPortBindingConfigError,
)
from gateway.profile_routing import ProfileRoute, ProfileRouteRejected
from gateway.config import GatewayConfig, Platform
from gateway.platforms.base import BasePlatformAdapter, MessageEvent


@pytest.fixture
def mock_runner():
    """Create a minimal mock GatewayRunner with the methods we need."""
    runner = MagicMock(spec=GatewayRunner)
    runner.config = MagicMock(profile_routes=[])
    # Bind the actual methods to the mock
    runner._profile_name_for_source = GatewayRunner._profile_name_for_source.__get__(runner)
    runner._served_profile_names_for_source = GatewayRunner._served_profile_names_for_source.__get__(runner)
    runner._resolve_profile_home_for_source = GatewayRunner._resolve_profile_home_for_source.__get__(runner)
    return runner


@pytest.fixture
def discord_source():
    """Create a basic Discord SessionSource for testing."""
    return SessionSource(
        platform=MagicMock(value="discord"),
        chat_id="123456",
        guild_id="789",
        thread_id=None,
        parent_chat_id=None,
    )


@pytest.fixture
def telegram_source():
    """Create a basic Telegram SessionSource for testing.

    Telegram (like Slack/Feishu/etc.) has no ``guild_id`` — only ``chat_id``.
    Used to prove profile routing is platform-generic, not Discord-only.
    """
    return SessionSource(
        platform=MagicMock(value="telegram"),
        chat_id="-1001234567890",
        guild_id=None,
        thread_id=None,
        parent_chat_id=None,
    )


class TestResolutionOrder:
    """Tests that profile resolution follows the correct priority order."""
    
    def test_source_profile_wins_over_routing(self, mock_runner, discord_source):
        """source.profile should be used even if routing would match."""
        discord_source.profile = "from-source"
        mock_runner._served_profile_names = {"from-source"}

        with patch("hermes_cli.profiles.resolve_profile_home", return_value=Path("/hermes/profiles/from-source")) as resolver:
            result = mock_runner._resolve_profile_home_for_source(discord_source)

        assert result == Path("/hermes/profiles/from-source")
        resolver.assert_called_once_with("from-source", require_exists=True)
    
    
    


class TestMissingProfileWarning:
    """Invalid or missing explicit profiles are refused without fallback."""
    
    def test_nonexistent_profile_warning(self, mock_runner, discord_source, caplog):
        """When source.profile points to a nonexistent profile, refuse it."""
        discord_source.profile = "nonexistent"
        mock_runner._served_profile_names = {"nonexistent"}

        with patch("hermes_cli.profiles.resolve_profile_home", side_effect=FileNotFoundError):
            with patch("hermes_constants.get_hermes_home", side_effect=AssertionError("must not fallback")):
                with caplog.at_level(logging.WARNING):
                    with pytest.raises(ProfileRouteRejectedError, match="^profile route rejected$"):
                        mock_runner._resolve_profile_home_for_source(discord_source)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "nonexistent" not in caplog.records[0].message
        assert "does not exist" not in caplog.records[0].message
        assert "discord" in caplog.records[0].message
        assert "123456" in caplog.records[0].message
    
    
    


class TestExceptionHandling:
    """Tests for exception handling in profile resolution."""
    
    def test_get_profile_dir_exception_logs_warning(self, mock_runner, discord_source, caplog):
        """When the safe resolver raises, refuse without exposing a path."""
        discord_source.profile = "bad-profile"
        mock_runner._served_profile_names = {"bad-profile"}

        with patch("hermes_cli.profiles.resolve_profile_home", side_effect=ValueError("outside profiles root")):
            with patch("hermes_constants.get_hermes_home", side_effect=AssertionError("must not fallback")):
                with caplog.at_level(logging.WARNING):
                    with pytest.raises(ProfileRouteRejectedError, match="^profile route rejected$"):
                        mock_runner._resolve_profile_home_for_source(discord_source)

        assert len(caplog.records) == 1
        assert caplog.records[0].levelname == "WARNING"
        assert "bad-profile" not in caplog.records[0].message
        assert "outside profiles root" not in caplog.records[0].message
    


class TestRoutingConsultation:
    """Tests that _profile_name_for_source is consulted when source.profile is empty."""
    
    def test_routing_consulted_when_source_profile_empty(self, mock_runner, discord_source):
        """_profile_name_for_source should be called when source.profile is empty."""
        discord_source.profile = None
        mock_runner._served_profile_names = {"routed"}
        mock_runner._profile_name_for_source = MagicMock(return_value="routed")

        with patch("hermes_cli.profiles.resolve_profile_home", return_value=Path("/hermes/profiles/routed")):
            mock_runner._resolve_profile_home_for_source(discord_source)

        # Should have called routing
        mock_runner._profile_name_for_source.assert_called_once_with(discord_source)
    


class TestNonDiscordProfileRouting:
    """Profile routing must be platform-generic, not Discord-only.

    Regression coverage for the ``gateway_runner`` injection gap: previously
    only Discord's adapter pre-declared ``gateway_runner``, so only Discord
    ever had ``build_source`` call ``_profile_name_for_source``. Telegram /
    Feishu / Slack / etc. silently fell through to the default profile. These
    tests pin the resolution half for a non-Discord platform (Telegram).
    """

    def test_telegram_route_resolves(self, mock_runner, telegram_source):
        """A configured Telegram route resolves to its profile via the real
        ``_profile_name_for_source`` (bound onto the mock runner)."""
        mock_runner.config.profile_routes = [
            ProfileRoute(name="tg", platform="telegram", profile="tg-profile",
                         chat_id="-1001234567890"),
        ]
        telegram_source.profile = None

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", Path("/profiles/default")),
                          ("tg-profile", Path("/profiles/tg-profile"))],
        ):
            assert mock_runner._profile_name_for_source(telegram_source) == "tg-profile"

    def test_route_inside_allowlist_resolves(self, mock_runner, telegram_source):
        mock_runner.config.multiplex_profile_allowlist = ["worker"]
        mock_runner.config.profile_routes = [
            ProfileRoute(
                name="worker-route",
                platform="telegram",
                profile="worker",
                chat_id="route-chat",
            )
        ]
        telegram_source.chat_id = "route-chat"

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", Path("/profiles/default")),
                          ("worker", Path("/profiles/worker"))],
        ) as enumerate_profiles:
            assert mock_runner._profile_name_for_source(telegram_source) == "worker"

        enumerate_profiles.assert_called_once_with(
            multiplex=True, profile_allowlist=["worker"]
        )

    def test_route_outside_allowlist_rejects(self, mock_runner, telegram_source, caplog):
        mock_runner.config.multiplex_profile_allowlist = ["worker"]
        mock_runner.config.profile_routes = [
            ProfileRoute(
                name="restricted-route",
                platform="telegram",
                profile="restricted",
                chat_id="route-chat",
            )
        ]
        telegram_source.chat_id = "route-chat"

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", Path("/profiles/default")),
                          ("worker", Path("/profiles/worker"))],
        ), caplog.at_level(logging.WARNING, logger="gateway.run"):
            with pytest.raises(ProfileRouteRejected):
                mock_runner._profile_name_for_source(telegram_source)

        assert "target profile 'restricted' is not served" in caplog.text

    def test_no_route_match_preserves_default_sentinel(self, mock_runner, telegram_source):
        mock_runner.config.multiplex_profile_allowlist = ["worker"]
        mock_runner.config.profile_routes = [
            ProfileRoute(
                name="other-chat",
                platform="telegram",
                profile="worker",
                chat_id="different-chat",
            )
        ]
        telegram_source.chat_id = "route-chat"

        assert mock_runner._profile_name_for_source(telegram_source) is None
        adapter = _stub_adapter(Platform.TELEGRAM, mock_runner)
        source = adapter.build_source(chat_id="route-chat", chat_type="group")
        assert source.profile is None


class TestGatewayRunnerInjection:
    """``BasePlatformAdapter`` declares ``gateway_runner`` so the gateway's
    unconditional injection reaches every platform adapter — the foundation
    that makes the routing in TestNonDiscordProfileRouting reachable at runtime.
    """

    def test_base_adapter_declares_gateway_runner(self):
        from gateway.platforms.base import BasePlatformAdapter

        # Class-level attribute exists and defaults to None.
        assert hasattr(BasePlatformAdapter, "gateway_runner")
        assert BasePlatformAdapter.gateway_runner is None


# A concrete adapter we can instantiate without the full platform stack.
# ``build_source`` only reads ``self.platform`` and ``self.gateway_runner``, so a
# bare instance with those two attrs exercises the real BasePlatformAdapter
# method end-to-end. Clearing ``__abstractmethods__`` lets ``__new__`` bypass
# the ABC instantiation guard without stubbing connect/send/get_chat_info/…
class _StubAdapter(BasePlatformAdapter):
    pass


_StubAdapter.__abstractmethods__ = frozenset()  # type: ignore[attr-defined]


def _stub_adapter(platform: Platform, runner) -> "_StubAdapter":
    a = _StubAdapter.__new__(_StubAdapter)
    a.platform = platform
    a.gateway_runner = runner
    return a


class TestAdapterToSessionKeyIntegration:
    """Adapter -> ``source.profile`` -> session-key integration coverage.

    The review asked for integration coverage for Discord AND a non-Discord
    platform. These drive a concrete adapter's real ``build_source``
    (BasePlatformAdapter) with an injected ``gateway_runner``, assert the
    matched route's profile is stamped on the source, and that the resulting
    session key is profile-scoped (``agent:<profile>:...`` rather than the
    shared ``agent:main:...``). The Telegram case is the bug-#2 regression:
    pre-fix it never received ``gateway_runner`` and fell through to default.
    """

    @staticmethod
    def _routes():
        return [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="111", chat_id="222"),
            ProfileRoute(name="tg", platform="telegram", profile="ops",
                         chat_id="-1001234567890"),
        ]

    def test_discord_adapter_stamps_profile_and_scopes_key(self, mock_runner):
        mock_runner.config.profile_routes = self._routes()
        adapter = _stub_adapter(Platform.DISCORD, mock_runner)

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", Path("/profiles/default")),
                          ("coder", Path("/profiles/coder"))],
        ):
            source = adapter.build_source(
                chat_id="222", chat_type="group", guild_id="111", user_id="u1",
            )
        assert source.profile == "coder"

        key = build_session_key(source, profile=source.profile)
        assert key.startswith("agent:coder:"), key
        # A default-profile key would land in agent:main — must differ.
        assert key != build_session_key(source, profile=None)

    @pytest.mark.asyncio
    async def test_adapter_drops_rejected_route_before_dispatch(self, mock_runner):
        mock_runner.config.multiplex_profile_allowlist = []
        mock_runner.config.profile_routes = [
            ProfileRoute(
                name="restricted-route",
                platform="telegram",
                profile="restricted",
                chat_id="route-chat",
            )
        ]
        adapter = _stub_adapter(Platform.TELEGRAM, mock_runner)

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", Path("/profiles/default"))],
        ):
            source = adapter.build_source(chat_id="route-chat", chat_type="group")

        assert source.profile is None
        assert source.profile_route_rejected is True
        roundtrip = SessionSource.from_dict(source.to_dict())
        assert roundtrip.profile_route_rejected is False
        assert roundtrip == source
        result = await GatewayRunner._handle_message(
            mock_runner,
            MessageEvent(text="discard me", source=source),
        )
        assert result is None

    @pytest.mark.asyncio
    async def test_direct_source_is_rejected_at_shared_ingress(self, mock_runner):
        mock_runner.config.multiplex_profiles = True
        mock_runner.config.multiplex_profile_allowlist = []
        mock_runner.config.profile_routes = [
            ProfileRoute(
                name="restricted-route",
                platform="telegram",
                profile="restricted",
                chat_id="route-chat",
            )
        ]
        source = SessionSource(platform=Platform.TELEGRAM, chat_id="route-chat")

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", Path("/profiles/default"))],
        ):
            result = await GatewayRunner._handle_message(
                mock_runner,
                MessageEvent(text="discard me", source=source),
            )

        assert result is None
        assert source.profile is None
        assert source.profile_route_rejected is True


class TestMultiplexGate:
    """``profile_routes`` only activates under ``gateway.multiplex_profiles``.

    Routing stamps ``source.profile``, which namespaces session/batch keys —
    but the profile-scoped agent run (``_profile_runtime_scope``) only engages
    when multiplexing is on. Without the gate, a configured route with
    multiplexing off would split batch/session keys into ``agent:<profile>``
    while the agent still served the turn from ``agent:main``'s home.
    """

    def test_routes_ignored_when_multiplex_off(self, mock_runner, discord_source):
        mock_runner.config.multiplex_profiles = False
        mock_runner.config.profile_routes = [
            ProfileRoute(name="dc", platform="discord", profile="coder",
                         guild_id="789", chat_id="123456"),
        ]
        discord_source.profile = None

        assert mock_runner._profile_name_for_source(discord_source) is None


class TestProfileRouteContainment:
    """Runner authorization is separate from filesystem discovery and fallback."""

    @staticmethod
    def _tree(tmp_path, monkeypatch):
        import hermes_cli.profiles as profiles_mod

        default_home = tmp_path / ".hermes"
        profiles_root = default_home / "profiles"
        secondary = profiles_root / "secondary"
        active = profiles_root / "active"
        hidden = profiles_root / "hidden"
        outside = tmp_path / "outside"
        for home in (default_home, profiles_root, secondary, active, hidden, outside):
            home.mkdir(exist_ok=True)
        for home, marker in ((secondary, "secondary"), (active, "active"), (hidden, "hidden"), (outside, "outside")):
            (home / "config.yaml").write_text(f"marker: {marker}\n", encoding="utf-8")
            (home / ".env").write_text(f"PROFILE_MARKER={marker}\n", encoding="utf-8")
            (home / "state.db").write_bytes(f"{marker}-state".encode())
        monkeypatch.setattr(profiles_mod, "_get_default_hermes_home", lambda: default_home)
        monkeypatch.setattr(profiles_mod, "_get_profiles_root", lambda: profiles_root)
        return default_home, profiles_root, secondary, active, hidden, outside

    @staticmethod
    def _runner(served):
        runner = GatewayRunner.__new__(GatewayRunner)
        runner.config = GatewayConfig(multiplex_profiles=True, profile_routes=[])
        runner._served_profile_names = set(served)
        return runner

    def test_invalid_and_existing_unserved_routes_do_not_touch_outside_files(
        self, tmp_path, monkeypatch
    ):
        _default, _root, _secondary, _active, hidden, outside = self._tree(
            tmp_path, monkeypatch
        )
        runner = self._runner({"default", "secondary"})

        before = {
            name: (path / name).read_bytes()
            for path in (outside, hidden)
            for name in ("config.yaml", ".env", "state.db")
        }

        invalid = SessionSource(platform=Platform.DISCORD, chat_id="bad", profile="../outside")
        with pytest.raises(ProfileRouteRejectedError):
            runner._resolve_profile_home_for_source(invalid)

        unserved = SessionSource(platform=Platform.DISCORD, chat_id="hidden", profile="hidden")
        with patch("hermes_cli.profiles.resolve_profile_home") as resolver:
            with pytest.raises(ProfileRouteRejectedError):
                runner._resolve_profile_home_for_source(unserved)
        resolver.assert_not_called()

        after = {
            name: (path / name).read_bytes()
            for path in (outside, hidden)
            for name in ("config.yaml", ".env", "state.db")
        }
        assert after == before

    def test_configured_secondary_is_authorized_even_without_adapter_and_active_default_resolve(
        self, tmp_path, monkeypatch
    ):
        default, _root, secondary, active, _hidden, _outside = self._tree(
            tmp_path, monkeypatch
        )
        runner = self._runner({"default", "secondary", "active"})

        secondary_source = SessionSource(
            platform=Platform.DISCORD, chat_id="secondary", profile="SeCoNdArY"
        )
        assert runner._resolve_profile_home_for_source(secondary_source) == secondary.resolve()

        with patch("hermes_cli.profiles.get_active_profile_name", return_value="default"):
            default_source = SessionSource(platform=Platform.DISCORD, chat_id="default")
            assert runner._resolve_profile_home_for_source(default_source) == default.resolve()

        with patch("hermes_cli.profiles.get_active_profile_name", return_value="active"):
            active_source = SessionSource(platform=Platform.DISCORD, chat_id="active")
            assert runner._resolve_profile_home_for_source(active_source) == active.resolve()

    def test_startup_publishes_secondary_before_adapter_skip(self, tmp_path, monkeypatch):
        import asyncio

        default, _root, secondary, _active, _hidden, _outside = self._tree(
            tmp_path, monkeypatch
        )
        runner = self._runner(set())
        runner.adapters = {}
        runner._profile_adapters = {}
        runner._failed_platforms = {}
        runner.pairing_stores = {}
        runner.pairing_store = MagicMock()
        runner._start_one_profile_adapters = AsyncMock(
            side_effect=SecondaryPortBindingConfigError("shared listener")
        )

        with patch(
            "hermes_cli.profiles.profiles_to_serve",
            return_value=[("default", default), ("secondary", secondary)],
        ), patch("hermes_cli.profiles.get_active_profile_name", return_value="default"), \
                patch("gateway.status.write_runtime_status"):
            assert asyncio.run(runner._start_secondary_profile_adapters()) == 0

        assert runner._served_profile_names == {"default", "secondary"}


class TestProfileRuntimeScopeCleanup:
    """Every setup/body failure restores the exact caller context."""

    @staticmethod
    def _assert_context(home, secrets):
        from agent.secret_scope import current_secret_scope
        from hermes_constants import get_hermes_home_override

        assert get_hermes_home_override() == str(home)
        assert current_secret_scope() == secrets

    def _failure_case(self, tmp_path, patchers):
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from gateway.run import _profile_runtime_scope

        outer_home = tmp_path / "outer"
        profile_home = tmp_path / "profile"
        outer_home.mkdir()
        profile_home.mkdir()
        (profile_home / ".env").write_text("INNER=1\n", encoding="utf-8")
        outer_secrets = {"OUTER": "yes"}
        home_token = set_hermes_home_override(outer_home)
        secret_token = set_secret_scope(outer_secrets)
        try:
            with patchers:
                with pytest.raises(RuntimeError, match="scope boom"):
                    with _profile_runtime_scope(profile_home):
                        pass
            self._assert_context(outer_home, outer_secrets)
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    def test_hydrate_failure_restores_prior_home_and_secret(self, tmp_path):
        from unittest.mock import patch

        self._failure_case(
            tmp_path,
            patch(
                "hermes_cli.env_loader.hydrate_profile_secret_sources",
                side_effect=RuntimeError("scope boom"),
            ),
        )

    def test_build_scope_failure_restores_prior_home_and_secret(self, tmp_path):
        from unittest.mock import patch

        self._failure_case(
            tmp_path,
            patch(
                "agent.secret_scope.build_profile_secret_scope",
                side_effect=RuntimeError("scope boom"),
            ),
        )

    def test_set_scope_failure_restores_prior_home_and_secret(self, tmp_path):
        from unittest.mock import patch

        self._failure_case(
            tmp_path,
            patch(
                "agent.secret_scope.set_secret_scope",
                side_effect=RuntimeError("scope boom"),
            ),
        )

    def test_body_failure_restores_prior_home_and_secret(self, tmp_path):
        from agent.secret_scope import reset_secret_scope, set_secret_scope
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override
        from gateway.run import _profile_runtime_scope

        outer_home = tmp_path / "outer"
        profile_home = tmp_path / "profile"
        outer_home.mkdir()
        profile_home.mkdir()
        (profile_home / ".env").write_text("INNER=1\n", encoding="utf-8")
        outer_secrets = {"OUTER": "yes"}
        home_token = set_hermes_home_override(outer_home)
        secret_token = set_secret_scope(outer_secrets)
        try:
            with pytest.raises(RuntimeError, match="body boom"):
                with _profile_runtime_scope(profile_home):
                    raise RuntimeError("body boom")
            self._assert_context(outer_home, outer_secrets)
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

    def test_nested_success_restores_each_exact_parent_context(self, tmp_path):
        from agent.secret_scope import current_secret_scope, reset_secret_scope, set_secret_scope
        from hermes_constants import (
            get_hermes_home_override,
            reset_hermes_home_override,
            set_hermes_home_override,
        )
        from gateway.run import _profile_runtime_scope

        outer_home = tmp_path / "outer"
        profile_a = tmp_path / "profile-a"
        profile_b = tmp_path / "profile-b"
        for home, value in ((outer_home, "outer"), (profile_a, "a"), (profile_b, "b")):
            home.mkdir()
            (home / ".env").write_text(f"PROFILE={value}\n", encoding="utf-8")
        outer_secrets = {"OUTER": "yes"}
        home_token = set_hermes_home_override(outer_home)
        secret_token = set_secret_scope(outer_secrets)
        try:
            with _profile_runtime_scope(profile_a):
                assert get_hermes_home_override() == str(profile_a)
                scope_a = current_secret_scope()
                assert scope_a is not None
                assert scope_a["PROFILE"] == "a"
                with _profile_runtime_scope(profile_b):
                    assert get_hermes_home_override() == str(profile_b)
                    scope_b = current_secret_scope()
                    assert scope_b is not None
                    assert scope_b["PROFILE"] == "b"
                assert get_hermes_home_override() == str(profile_a)
                restored_a = current_secret_scope()
                assert restored_a is not None
                assert restored_a["PROFILE"] == "a"
            self._assert_context(outer_home, outer_secrets)
        finally:
            reset_secret_scope(secret_token)
            reset_hermes_home_override(home_token)

