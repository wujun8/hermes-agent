"""Hosted folder selection shares operator-configured roots with file and Git reads."""

import subprocess

import pytest
from starlette.testclient import TestClient

import hermes_constants
from hermes_cli import web_server, web_server_files
from hermes_cli.config import atomic_config_write


@pytest.fixture
def hosted_home(monkeypatch, tmp_path):
    root = (tmp_path / "data").resolve()
    root.mkdir()
    (root / "config.yaml").write_text("{}\n", encoding="utf-8")
    monkeypatch.delenv("HERMES_DASHBOARD_FILES_ROOT", raising=False)
    monkeypatch.setenv("HERMES_HOME", str(root))
    monkeypatch.setattr(web_server_files, "_HOSTED_MANAGED_FILES_ROOT", root)
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda **kwargs: root)
    return root


@pytest.fixture
def client(hosted_home, monkeypatch):
    monkeypatch.setattr(web_server.app.state, "auth_required", False, raising=False)
    monkeypatch.setattr(web_server.app.state, "bound_host", None, raising=False)
    client = TestClient(web_server.app, raise_server_exceptions=False)
    client.headers[web_server._SESSION_HEADER_NAME] = web_server._SESSION_TOKEN
    try:
        yield client
    finally:
        client.close()


def _write_roots(home, roots):
    atomic_config_write(home / "config.yaml", {"dashboard": {"files_allowed_roots": roots}})


def _assert_directory_denied(client, directory):
    for endpoint in ("/api/fs/list", "/api/files", "/api/git/review/commit-context"):
        response = client.get(endpoint, params={"path": str(directory)})
        assert response.status_code == 403, response.text
    for endpoint in ("/api/fs/read-text", "/api/files/read"):
        response = client.get(endpoint, params={"path": str(directory / "safe.txt")})
        assert response.status_code == 403, response.text
    response = client.get(
        "/api/git/review/diff", params={"path": str(directory), "file": "safe.txt"}
    )
    assert response.status_code == 403, response.text
    attempted = directory / "new folder"
    response = client.post("/api/files/mkdir", json={"path": str(attempted)})
    assert response.status_code == 403, response.text
    assert not attempted.exists()


def test_hosted_default_and_extra_roots_return_the_matching_boundary(client, hosted_home, tmp_path):
    extra_roots = [tmp_path / "workspace one", tmp_path / "workspace two"]
    for root in extra_roots:
        root.mkdir()
    _write_roots(hosted_home, [str(root) for root in extra_roots])

    default = client.get("/api/files")
    assert default.status_code == 200, default.text
    assert default.json()["path"] == str(hosted_home)
    assert default.json()["locked_root"] == str(hosted_home)

    for root in (hosted_home, *extra_roots):
        nested = root / "nested"
        nested.mkdir()
        safe = nested / "safe.txt"
        safe.write_text("workspace content\n", encoding="utf-8")
        response = client.get("/api/fs/list", params={"path": str(nested)})
        assert response.status_code == 200, response.text
        assert "safe.txt" in {entry["name"] for entry in response.json()["entries"]}
        response = client.get("/api/fs/read-text", params={"path": str(safe)})
        assert response.status_code == 200, response.text
        assert response.json()["text"] == "workspace content\n"

        created_path = nested / "fresh project"
        response = client.post("/api/files/mkdir", json={"path": str(created_path)})
        assert response.status_code == 200, response.text
        assert created_path.is_dir()
        assert response.json()["path"] == str(created_path)
        assert response.json()["locked_root"] == str(root)
        assert response.json()["can_change_path"] is False

        for endpoint, path in (("/api/files", nested), ("/api/files/read", safe)):
            response = client.get(endpoint, params={"path": str(path)})
            assert response.status_code == 200, response.text
            assert response.json()["root"] == str(root)
            assert response.json()["locked_root"] == str(root)
            assert response.json()["can_change_path"] is False


def test_missing_workspace_roots_keep_the_hosted_boundary(client, hosted_home, tmp_path):
    outside = tmp_path / "workspace"
    outside.mkdir()
    (outside / "safe.txt").write_text("outside", encoding="utf-8")
    response = client.get("/api/files")
    assert response.status_code == 200, response.text
    assert response.json()["locked_root"] == str(hosted_home)
    _assert_directory_denied(client, outside)


def test_workspace_roots_use_machine_config_instead_of_active_profile(
    client, hosted_home, monkeypatch, tmp_path
):
    allowed = tmp_path / "machine workspace"
    denied = tmp_path / "profile workspace"
    for root in (allowed, denied):
        root.mkdir()
        (root / "safe.txt").write_text("workspace", encoding="utf-8")
    _write_roots(hosted_home, [str(allowed)])
    profile_home = hosted_home / "profiles" / "work"
    profile_home.mkdir(parents=True)
    (profile_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    _write_roots(profile_home, [str(denied)])
    monkeypatch.setenv("HERMES_HOME", str(profile_home))

    response = client.get("/api/fs/read-text", params={"path": str(allowed / "safe.txt")})
    assert response.status_code == 200, response.text
    response = client.post("/api/files/mkdir", json={"path": str(allowed / "project")})
    assert response.status_code == 200, response.text
    assert response.json()["locked_root"] == str(allowed)
    _assert_directory_denied(client, denied)


def test_explicit_files_root_remains_the_only_boundary(client, hosted_home, monkeypatch, tmp_path):
    forced = tmp_path / "forced"
    workspace = tmp_path / "workspace"
    for root in (forced, workspace, hosted_home):
        root.mkdir(exist_ok=True)
        (root / "safe.txt").write_text("safe", encoding="utf-8")
    monkeypatch.setenv("HERMES_DASHBOARD_FILES_ROOT", str(forced))

    for roots in ([str(workspace)], "invalid list"):
        _write_roots(hosted_home, roots)
        response = client.get("/api/files")
        assert response.status_code == 200, response.text
        assert response.json()["locked_root"] == str(forced)
        response = client.get("/api/fs/read-text", params={"path": str(forced / "safe.txt")})
        assert response.status_code == 200, response.text
        _assert_directory_denied(client, workspace)
        _assert_directory_denied(client, hosted_home)


def test_nonhosted_launch_does_not_apply_workspace_roots(client, monkeypatch, tmp_path):
    local_home = tmp_path / "local-home"
    local_home.mkdir()
    (local_home / "config.yaml").write_text("{}\n", encoding="utf-8")
    _write_roots(local_home, "invalid list")
    monkeypatch.setenv("HERMES_HOME", str(local_home))
    monkeypatch.setattr(hermes_constants, "get_default_hermes_root", lambda **kwargs: local_home)
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    safe = workspace / "safe.txt"
    safe.write_text("local workspace", encoding="utf-8")

    response = client.get("/api/fs/read-text", params={"path": str(safe)})
    assert response.status_code == 200, response.text
    response = client.get("/api/files", params={"path": str(workspace)})
    assert response.status_code == 200, response.text
    assert response.json()["locked_root"] is None
    assert response.json()["can_change_path"] is True
    response = client.post("/api/files/mkdir", json={"path": str(workspace / "project")})
    assert response.status_code == 200, response.text
    assert response.json()["locked_root"] is None


@pytest.mark.platforms("posix")
def test_configured_root_is_canonicalized(client, hosted_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    (workspace / "safe.txt").write_text("canonical content", encoding="utf-8")
    alias = tmp_path / "workspace alias"
    alias.symlink_to(workspace, target_is_directory=True)
    _write_roots(hosted_home, [str(alias)])

    for path in (workspace, alias):
        response = client.get("/api/fs/read-text", params={"path": str(path / "safe.txt")})
        assert response.status_code == 200, response.text
        assert response.json()["path"] == str(workspace / "safe.txt")
        response = client.get("/api/files", params={"path": str(path)})
        assert response.status_code == 200, response.text
        assert response.json()["locked_root"] == str(workspace)
    response = client.post("/api/files/mkdir", json={"path": str(alias / "project")})
    assert response.status_code == 200, response.text
    assert response.json()["path"] == str(workspace / "project")
    assert response.json()["locked_root"] == str(workspace)
    assert (workspace / "project").is_dir()


@pytest.mark.parametrize(
    "invalid_kind",
    ["scalar", "null", "mapping", "non-string", "empty", "whitespace", "relative", "parent"],
)
def test_invalid_workspace_roots_fail_closed(client, hosted_home, tmp_path, invalid_kind):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    safe = workspace / "safe.txt"
    safe.write_text("must not be readable", encoding="utf-8")
    valid = str(workspace)
    values = {
        "scalar": valid,
        "null": None,
        "mapping": {"root": valid},
        "non-string": [valid, 42],
        "empty": [valid, ""],
        "whitespace": [valid, "   "],
        "relative": [valid, "workspace"],
        "parent": [valid, str(workspace / ".." / "outside")],
    }
    _write_roots(hosted_home, values[invalid_kind])

    for endpoint, path in (
        ("/api/files", hosted_home),
        ("/api/fs/list", hosted_home),
        ("/api/fs/read-text", safe),
        ("/api/git/review/commit-context", workspace),
    ):
        response = client.get(endpoint, params={"path": str(path)})
        assert response.status_code == 500, response.text
    attempted = workspace / "project"
    response = client.post("/api/files/mkdir", json={"path": str(attempted)})
    assert response.status_code == 500, response.text
    assert not attempted.exists()


@pytest.mark.platforms("posix")
def test_outside_paths_and_symlink_escapes_stay_denied(client, hosted_home, tmp_path):
    workspace = tmp_path / "workspace"
    outside = tmp_path / "workspace-sibling"
    for root in (workspace, outside):
        root.mkdir()
        (root / "safe.txt").write_text("safe", encoding="utf-8")
    _write_roots(hosted_home, [str(workspace)])
    escape = workspace / "escape"
    escape.symlink_to(outside, target_is_directory=True)

    _assert_directory_denied(client, outside)
    _assert_directory_denied(client, escape)
    response = client.get("/api/fs/list", params={"path": str(tmp_path)})
    assert response.status_code == 403, response.text


def test_workspace_listings_and_reads_still_reject_sensitive_files(client, hosted_home, tmp_path):
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    for name in ("safe.txt", ".env", "auth.json"):
        (workspace / name).write_text("private example", encoding="utf-8")
    _write_roots(hosted_home, [str(workspace)])

    for endpoint in ("/api/fs/list", "/api/files"):
        response = client.get(endpoint, params={"path": str(workspace)})
        assert response.status_code == 200, response.text
        names = {entry["name"] for entry in response.json()["entries"]}
        assert "safe.txt" in names
        assert names.isdisjoint({".env", "auth.json"})
    for name in (".env", "auth.json"):
        for endpoint in ("/api/fs/read-text", "/api/files/read"):
            response = client.get(endpoint, params={"path": str(workspace / name)})
            assert response.status_code == 403, response.text


def test_workspace_git_review_keeps_safe_diffs_and_filters_secrets(client, hosted_home, tmp_path):
    workspace = tmp_path / "repo"
    workspace.mkdir()

    def git(*args):
        subprocess.run(["git", *args], cwd=workspace, check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.com")
    git("config", "user.name", "Test")
    for name in ("safe.txt", ".env", "auth.json"):
        (workspace / name).write_text("old\n", encoding="utf-8")
    git("add", "-A")
    git("commit", "-qm", "initial")
    (workspace / "safe.txt").write_text("SAFE_CHANGE\n", encoding="utf-8")
    for name in (".env", "auth.json"):
        (workspace / name).write_text("PRIVATE_SECRET_CHANGE\n", encoding="utf-8")
    _write_roots(hosted_home, [str(workspace)])

    response = client.get("/api/git/review/commit-context", params={"path": str(workspace)})
    assert response.status_code == 200, response.text
    assert "SAFE_CHANGE" in response.json()["diff"]
    assert "PRIVATE_SECRET_CHANGE" not in response.json()["diff"]
    for endpoint in ("/api/git/review/diff", "/api/git/file-diff"):
        response = client.get(endpoint, params={"path": str(workspace), "file": "safe.txt"})
        assert response.status_code == 200, response.text
        assert "SAFE_CHANGE" in response.json()["diff"]
        for name in (".env", "auth.json"):
            response = client.get(endpoint, params={"path": str(workspace), "file": name})
            assert response.status_code == 403, response.text
