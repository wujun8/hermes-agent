"""Profile identity must survive activation while an agent is still building."""

from tui_gateway import server


def test_lazy_activation_reports_owning_profile(tmp_path, monkeypatch):
    from pathlib import Path

    root = tmp_path / ".hermes"
    root.mkdir()
    secondary = root / "profiles" / "docker"
    secondary.mkdir(parents=True)
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(server, "_current_profile_name", lambda: "default")
    monkeypatch.setattr(server, "_lazy_info_route", lambda *_: {"model": "test-model"})
    monkeypatch.setattr(server, "_project_info_for_cwd", lambda *_: {})
    monkeypatch.setattr(server.git_probe, "branch", lambda *_: "")

    for home, expected in ((root, "default"), (secondary, "docker"), (root, "default")):
        info = server._fallback_session_info({"agent": None, "profile_home": str(home), "cwd": str(tmp_path)})
        assert info["profile_name"] == expected
        assert info["lazy"] is True
        assert info["model"] == "test-model"


def test_live_activation_passes_the_selected_session(monkeypatch):
    agent = object()
    selected = {"agent": agent, "profile_home": "/test/profiles/docker"}
    calls = []

    def session_info(actual_agent, actual_session):
        calls.append((actual_agent, actual_session))
        return {"profile_name": "docker"}

    monkeypatch.setattr(server, "_session_info", session_info)
    assert server._fallback_session_info(selected)["profile_name"] == "docker"
    assert calls == [(agent, selected)]
