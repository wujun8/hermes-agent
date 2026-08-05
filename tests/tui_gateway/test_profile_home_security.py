"""Focused profile-home containment tests for the TUI gateway."""

from pathlib import Path

import pytest

import tui_gateway.server as server


@pytest.fixture
def profile_tree(tmp_path, monkeypatch):
    default_home = tmp_path / ".hermes"
    profiles_root = default_home / "profiles"
    work_home = profiles_root / "work"
    outside_home = tmp_path / "outside"
    default_home.mkdir()
    profiles_root.mkdir()
    work_home.mkdir()
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
    return default_home, profiles_root, work_home, outside_home


def test_valid_profile_is_resolved_to_existing_home(profile_tree):
    _default_home, _profiles_root, work_home, _outside_home = profile_tree

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
    _default_home, _profiles_root, _work_home, _outside_home = profile_tree

    assert server._profile_home(profile) is None


def test_existing_outside_home_and_symlink_escape_are_rejected(profile_tree):
    default_home, profiles_root, _work_home, outside_home = profile_tree
    (profiles_root / "escape").symlink_to(outside_home, target_is_directory=True)

    assert server._profile_home(str(outside_home)) is None
    assert server._profile_home("escape") is None
    assert (outside_home / "config.yaml").read_text(encoding="utf-8") == "outside"
    assert (outside_home / ".env").read_text(encoding="utf-8") == "OUTSIDE_SECRET=sentinel\n"
    assert (outside_home / "state.db").read_bytes() == b"outside-state"
    assert server._profile_home("default") is None
    assert server._profile_home(None) is None


def test_launch_profile_and_non_string_profile_return_none(profile_tree):
    default_home, _profiles_root, work_home, _outside_home = profile_tree

    assert server._profile_home("default") is None
    server._hermes_home = work_home
    assert server._profile_home("WORK") is None
    assert server._profile_home(123) is None  # type: ignore[arg-type]
    server._hermes_home = default_home
