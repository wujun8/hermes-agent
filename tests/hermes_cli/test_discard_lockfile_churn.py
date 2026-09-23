"""Local update policy preserves every user lockfile edit for autostash.

Workspace ownership cannot distinguish npm churn from an intentional lockfile-only change.
The compatibility cleanup hook must leave all dirty bytes intact.
"""

import json
import os
import subprocess

from hermes_cli.update_cmd_git import _discard_lockfile_churn

_GIT_ENV = {
    **os.environ,
    "GIT_AUTHOR_NAME": "t", "GIT_AUTHOR_EMAIL": "t@example.invalid",
    "GIT_COMMITTER_NAME": "t", "GIT_COMMITTER_EMAIL": "t@example.invalid",
}


def _git(repo, *args):
    return subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True, check=True, env=_GIT_ENV).stdout


def _repo(tmp_path):
    _git(tmp_path, "init", "-q")
    (tmp_path / "package.json").write_text(json.dumps({"workspaces": ["apps/*", "ui-tui"]}), encoding="utf-8")
    (tmp_path / "package-lock.json").write_text("lock v1\n", encoding="utf-8")
    for rel in ("apps/desktop", "ui-tui", "vendor/foo"):
        (tmp_path / rel).mkdir(parents=True)
        (tmp_path / rel / "package.json").write_text("{}\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "-qm", "initial")
    return tmp_path


def _still_dirty_after_cleanup(repo, *dirty):
    for rel in dirty:
        (repo / rel).write_text("dirty\n", encoding="utf-8")
    before = {rel: (repo / rel).read_bytes() for rel in dirty}
    _discard_lockfile_churn(["git"], repo)
    assert {rel: (repo / rel).read_bytes() for rel in dirty} == before
    return set(_git(repo, "diff", "--name-only").splitlines())


def test_dirty_workspace_manifest_preserves_root_lock(tmp_path):
    repo = _repo(tmp_path)
    assert _still_dirty_after_cleanup(repo, "apps/desktop/package.json", "package-lock.json") == {
        "apps/desktop/package.json", "package-lock.json",
    }


def test_manifest_outside_workspace_graph_still_preserves_root_lock(tmp_path):
    repo = _repo(tmp_path)
    assert _still_dirty_after_cleanup(repo, "vendor/foo/package.json", "package-lock.json") == {
        "vendor/foo/package.json", "package-lock.json",
    }


def test_unsupported_workspaces_glob_still_preserves_root_lock(tmp_path):
    """Malformed workspace metadata must not permit discarding dirty lockfile bytes."""
    repo = _repo(tmp_path)
    (repo / "package.json").write_text(json.dumps({"workspaces": "apps/*"}), encoding="utf-8")
    _git(repo, "commit", "-qam", "string workspaces")
    assert _still_dirty_after_cleanup(repo, "vendor/foo/package.json", "package-lock.json") == {
        "vendor/foo/package.json", "package-lock.json",
    }
