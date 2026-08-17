"""Focused profile-home containment tests for the TUI gateway."""

from pathlib import Path
from unittest.mock import Mock

import pytest

from hermes_cli import profiles as profiles_mod

import tui_gateway.server as server


PROFILE_ERROR = "invalid or unavailable profile"


@pytest.fixture
def profile_tree(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    work_home = profiles_root / "work"
    unserved_home = profiles_root / "unserved"
    outside_home = tmp_path / "outside"
    default_home.mkdir()
    profiles_root.mkdir()
    work_home.mkdir()
    unserved_home.mkdir()
    outside_home.mkdir()
    (outside_home / "config.yaml").write_text("outside", encoding="utf-8")
    (outside_home / ".env").write_text("OUTSIDE_SECRET=sentinel\n", encoding="utf-8")
    (outside_home / "state.db").write_bytes(b"outside-state")

    monkeypatch.setattr(server, "_hermes_home", default_home)
    monkeypatch.setattr(
        "hermes_cli.profiles._get_default_hermes_home",
        lambda: default_home,
    )
    monkeypatch.setattr(
        "hermes_cli.profiles._get_profiles_root",
        lambda: profiles_root,
    )
    monkeypatch.setattr(
        profiles_mod,
        "profiles_to_serve",
        lambda multiplex: [("default", default_home), ("work", work_home)],
    )
    return default_home, profiles_root, work_home, outside_home, unserved_home


def test_valid_profile_is_resolved_to_existing_home(profile_tree):
    _default_home, _profiles_root, work_home, _outside_home, _unserved_home = profile_tree

    assert server._profile_home("  WoRk ") == work_home.resolve()


@pytest.mark.parametrize(
    "profile",
    [
        "../outside",
        "../../existing",
        "/absolute/profile",
        r"foo\bar",
        r"C:\\profile",
        r"\\\\server\\share",
        ".",
        "a\x00b",
        "a\nb",
        "a" * 65,
    ],
)
def test_invalid_profile_names_are_rejected(profile_tree, profile):
    _default_home, _profiles_root, _work_home, _outside_home, _unserved_home = profile_tree

    with pytest.raises(server.TUIProfileSelectionError) as exc_info:
        server._profile_home(profile)
    assert str(exc_info.value) == PROFILE_ERROR


def test_existing_outside_home_and_symlink_escape_are_rejected(profile_tree):
    default_home, profiles_root, _work_home, outside_home, unserved_home = profile_tree
    (profiles_root / "escape").symlink_to(outside_home, target_is_directory=True)

    for profile in (str(outside_home), "escape", "unserved"):
        with pytest.raises(server.TUIProfileSelectionError) as exc_info:
            server._profile_home(profile)
        assert str(exc_info.value) == PROFILE_ERROR
        assert str(outside_home) not in str(exc_info.value)
        assert profile not in str(exc_info.value)
    assert (outside_home / "config.yaml").read_text(encoding="utf-8") == "outside"
    assert (outside_home / ".env").read_text(encoding="utf-8") == "OUTSIDE_SECRET=sentinel\n"
    assert (outside_home / "state.db").read_bytes() == b"outside-state"
    assert server._profile_home("default") is None
    assert server._profile_home(None) is None
    assert not unserved_home.is_symlink()


def test_launch_profile_and_non_string_profile_selection(profile_tree):
    default_home, _profiles_root, work_home, _outside_home, _unserved_home = profile_tree

    assert server._profile_home("default") is None
    server._hermes_home = work_home
    assert server._profile_home("WORK") is None
    with pytest.raises(server.TUIProfileSelectionError) as exc_info:
        server._profile_home(123)  # type: ignore[arg-type]
    assert str(exc_info.value) == PROFILE_ERROR
    server._hermes_home = default_home


def test_omitted_and_launch_profile_use_launch_db_and_handler(profile_tree, monkeypatch):
    _default_home, _profiles_root, _work_home, _outside_home, _unserved_home = profile_tree
    launch_db = object()
    get_db = Mock(return_value=launch_db)
    monkeypatch.setattr(server, "_get_db", get_db)

    for profile in (None, "", "  DeFaUlT  "):
        db, owns_handle = server._db_for_profile(profile)
        assert db is launch_db
        assert owns_handle is False

    seen = []

    def handler(rid, params):
        seen.append(rid)
        return {"rid": rid}

    wrapped = server._profile_scoped(handler)
    assert wrapped(1, {}) == {"rid": 1}
    assert wrapped(2, {"profile": " DeFaUlT "}) == {"rid": 2}
    assert seen == [1, 2]
    assert get_db.call_count == 3


def test_explicit_mixed_case_served_profile_uses_resolved_db_and_handler(
    profile_tree, monkeypatch
):
    _default_home, _profiles_root, work_home, _outside_home, _unserved_home = profile_tree
    import hermes_state

    opened = []

    class FakeSessionDB:
        def __init__(self, *, db_path):
            self.db_path = Path(db_path)
            opened.append(self.db_path)

    monkeypatch.setattr(hermes_state, "SessionDB", FakeSessionDB)
    db, owns_handle = server._db_for_profile("  WoRk  ")
    assert owns_handle is True
    assert db.db_path == work_home / "state.db"
    assert opened == [work_home / "state.db"]
    assert server._response_profile_name("  WoRk  ") == "work"

    seen = []

    def handler(rid, params):
        seen.append(server.get_hermes_home().resolve())
        return {"ok": True}

    assert server._profile_scoped(handler)(3, {"profile": "  WoRk  "}) == {"ok": True}
    assert seen == [work_home.resolve()]


@pytest.fixture
def invalid_profiles(profile_tree):
    _default_home, profiles_root, _work_home, outside_home, _unserved_home = profile_tree
    (profiles_root / "escape").symlink_to(outside_home, target_is_directory=True)
    return {
        "traversal": "../outside",
        "nonexistent": "missing",
        "non_string": 123,
        "symlink": "escape",
        "unserved": "unserved",
    }


@pytest.mark.parametrize(
    "profile_kind", ["traversal", "nonexistent", "non_string", "symlink", "unserved"]
)
def test_invalid_profile_db_and_handler_fail_closed(
    profile_tree, invalid_profiles, profile_kind, monkeypatch
):
    profile = invalid_profiles[profile_kind]
    get_db = Mock(return_value=object())
    monkeypatch.setattr(server, "_get_db", get_db)

    with pytest.raises(server.TUIProfileSelectionError) as exc_info:
        server._db_for_profile(profile)
    assert str(exc_info.value) == PROFILE_ERROR
    assert get_db.call_count == 0

    calls = []

    def handler(rid, params):
        calls.append((rid, params))
        return {"handled": True}

    response = server._profile_scoped(handler)(9, {"profile": profile})
    assert 4000 <= response["error"]["code"] < 5000
    assert response["error"]["message"] == PROFILE_ERROR
    assert calls == []
    if isinstance(profile, str):
        assert str(profile_tree[3]) not in response["error"]["message"]
        assert profile not in response["error"]["message"]


@pytest.mark.parametrize("rpc_name", ["session.create", "session.resume", "session.delete"])
@pytest.mark.parametrize(
    "profile_kind", ["traversal", "nonexistent", "non_string", "symlink", "unserved"]
)
def test_invalid_profile_session_rpcs_have_no_side_effects(
    profile_tree, invalid_profiles, rpc_name, profile_kind, monkeypatch
):
    profile = invalid_profiles[profile_kind]
    params = {"profile": profile}
    if rpc_name != "session.create":
        params["session_id"] = "session-id"

    launch_db = Mock()
    get_db = Mock(return_value=launch_db)
    load_cfg = Mock(return_value={})
    monkeypatch.setattr(server, "_get_db", get_db)
    monkeypatch.setattr(server, "_load_cfg", load_cfg)
    gateway_prompts = Mock()
    completion_cwd = Mock(return_value=str(profile_tree[0]))
    new_session_key = Mock(return_value="new-key")
    monkeypatch.setattr(server, "_enable_gateway_prompts", gateway_prompts)
    monkeypatch.setattr(server, "_completion_cwd", completion_cwd)
    monkeypatch.setattr(server, "_new_session_key", new_session_key)
    before_sessions = dict(server._sessions)

    response = server.handle_request({"id": 100, "method": rpc_name, "params": params})
    assert response is not None
    assert 4000 <= response["error"]["code"] < 5000
    assert response["error"]["message"] == PROFILE_ERROR
    assert get_db.call_count == 0
    assert load_cfg.call_count == 0
    assert gateway_prompts.call_count == 0
    assert completion_cwd.call_count == 0
    assert new_session_key.call_count == 0
    assert dict(server._sessions) == before_sessions
