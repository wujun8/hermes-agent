"""Real-Git regression coverage for release-upgrade transactions.

These tests intentionally build temporary repositories.  The release path is
allowed to rewrite a candidate worktree, but it must never use the
implementation checkout as a fixture.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
import sys
import time
from pathlib import Path

import pytest

from hermes_cli import update_cmd



def _git(repo: Path, *args: str, check: bool = True, input_bytes: bytes | None = None):
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        input=input_bytes,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if check and result.returncode != 0:
        raise AssertionError(
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr.decode(errors='replace')}"
        )
    return result



def _init_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Test")
    _git(repo, "config", "user.email", "hermes-test@example.invalid")
    (repo / "README.txt").write_text("base\n", encoding="utf-8")
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", "base")
    return repo



def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.decode().strip()



def _write_base_metadata(
    repo: Path,
    *,
    tag: str,
    base_sha: str,
    patch_bytes: bytes = b"",
    target_sha: str | None = None,
) -> None:
    patches = repo / "local-patches"
    patches.mkdir(exist_ok=True)
    (patches / "0001-local-maintenance.patch").write_bytes(patch_bytes)
    metadata = {
        "tag": tag,
        "base_sha": base_sha,
        "target_sha": target_sha or base_sha,
        "patch_sha256": hashlib.sha256(patch_bytes).hexdigest(),
        "patch_bytes": len(patch_bytes),
    }
    (patches / ".release_base").write_text(
        json.dumps(metadata, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    (patches / "README.md").write_text("metadata\n", encoding="utf-8")
    _commit(repo, "record release-base metadata")



def _make_maintenance_from_tag(repo: Path, tag: str) -> tuple[str, str]:
    base_sha = _git(repo, "rev-parse", f"refs/tags/{tag}^{{commit}}").stdout.decode().strip()
    _git(repo, "branch", "hermes-release", base_sha)
    _git(repo, "switch", "hermes-release")
    return base_sha, _git(repo, "rev-parse", "HEAD").stdout.decode().strip()



def _commit_upstream_release(repo: Path, tag: str, *, filename: str, content: str) -> str:
    _git(repo, "switch", "main")
    (repo / filename).write_text(content, encoding="utf-8")
    sha = _commit(repo, f"release {tag}")
    _git(repo, "tag", "-a", tag, "-m", tag)
    return sha



def test_package_lock_edit_is_not_discarded_before_stash(tmp_path):
    repo = _init_repo(tmp_path)
    lockfile = repo / "package-lock.json"
    lockfile.write_text('{"lockfileVersion": 3, "intentional": true}\n', encoding="utf-8")

    update_cmd._discard_lockfile_churn(["git"], repo)

    assert json.loads(lockfile.read_text(encoding="utf-8"))["intentional"] is True
    assert "package-lock.json" in _git(repo, "status", "--porcelain").stdout.decode()


def test_unmerged_index_refuses_autostash_without_mutating_user_state(tmp_path, capsys):
    repo = _init_repo(tmp_path)
    _git(repo, "switch", "-c", "feature")
    (repo / "README.txt").write_text("feature side\n", encoding="utf-8")
    _commit(repo, "feature conflict")
    _git(repo, "switch", "main")
    (repo / "README.txt").write_text("main side\n", encoding="utf-8")
    _commit(repo, "main conflict")
    _git(repo, "merge", "feature", check=False)

    before_unmerged = _git(repo, "ls-files", "--unmerged").stdout
    before_bytes = (repo / "README.txt").read_bytes()
    before_status = _git(repo, "status", "--porcelain=v1").stdout

    with pytest.raises(RuntimeError, match="resolve conflicts"):
        update_cmd._stash_local_changes_if_needed(["git"], repo)

    assert _git(repo, "ls-files", "--unmerged").stdout == before_unmerged
    assert (repo / "README.txt").read_bytes() == before_bytes
    assert _git(repo, "status", "--porcelain=v1").stdout == before_status
    assert "resolve conflicts" in capsys.readouterr().out.lower()


def test_managed_eol_helper_and_autostash_preserve_intentional_crlf(tmp_path):
    repo = _init_repo(tmp_path)
    managed = repo / "managed.txt"
    (repo / ".gitattributes").write_text("managed.txt -text\n", encoding="utf-8")
    managed.write_text("one\ntwo\n", encoding="utf-8")
    _commit(repo, "add managed file")
    _git(repo, "config", "core.autocrlf", "true")
    crlf_bytes = b"one\r\ntwo\r\nintentional\r\n"
    managed.write_bytes(crlf_bytes)
    before_status = _git(repo, "status", "--porcelain=v1").stdout

    update_cmd._normalize_managed_eol(["git"], repo)

    assert managed.read_bytes() == crlf_bytes
    assert _git(repo, "status", "--porcelain=v1").stdout == before_status

    stash_ref = update_cmd._stash_local_changes_if_needed(
        ["git"], repo, marker="hermes-test-crlf"
    )
    assert stash_ref
    stashed_bytes = _git(repo, "show", f"{stash_ref}:managed.txt").stdout
    assert stashed_bytes == crlf_bytes
    assert update_cmd._restore_stashed_changes(["git"], repo, stash_ref) is True
    assert managed.read_bytes() == crlf_bytes
    assert _git(repo, "status", "--porcelain=v1").stdout == before_status


def test_repository_lock_cannot_be_stolen_by_unlinking_metadata(tmp_path):
    if update_cmd.fcntl is None:
        pytest.skip("POSIX directory locks are not available")
    repo = _init_repo(tmp_path)
    first = update_cmd.RepositoryUpdateLock(repo)
    second = update_cmd.RepositoryUpdateLock(repo)
    first.acquire()
    assert first.lock_path is not None
    first.lock_path.unlink()
    try:
        with pytest.raises(RuntimeError, match="another hermes update/upgrade"):
            second.acquire()
    finally:
        first.release()
    second.acquire()
    second.release()


@pytest.mark.parametrize("failure_kind", ["write", "fsync"])
def test_repository_lock_metadata_failure_unlocks_and_closes(tmp_path, monkeypatch, failure_kind):
    if update_cmd.fcntl is None:
        pytest.skip("POSIX directory locks are not available")
    repo = _init_repo(tmp_path)
    first = update_cmd.RepositoryUpdateLock(repo)
    with monkeypatch.context() as patcher:
        if failure_kind == "write":
            def fail_write(*_args, **_kwargs):
                raise OSError("injected metadata write failure")

            patcher.setattr(update_cmd, "_atomic_write_bytes", fail_write)
        else:
            patcher.setattr(
                update_cmd.os,
                "fsync",
                lambda _fd: (_ for _ in ()).throw(OSError("injected metadata fsync failure")),
            )
        with pytest.raises(RuntimeError, match="metadata"):
            first.acquire()
    assert first.acquired is False

    second = update_cmd.RepositoryUpdateLock(repo)
    second.acquire()
    second.release()
    first.release()


def test_repository_lock_is_released_when_owner_process_exits(tmp_path):
    repo = _init_repo(tmp_path)
    ready = tmp_path / "child-ready"
    source_root = Path(update_cmd.__file__).resolve().parents[1]
    child_code = (
        "from pathlib import Path; import sys; "
        "from hermes_cli.update_cmd import RepositoryUpdateLock; "
        "lock = RepositoryUpdateLock(Path(sys.argv[1])); lock.acquire(); "
        "Path(sys.argv[2]).write_text('ready')"
    )
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in [str(source_root), env.get("PYTHONPATH", "")] if part
    )
    child = subprocess.Popen(
        [sys.executable, "-c", child_code, str(repo), str(ready)],
        cwd=repo,
        env=env,
    )
    deadline = time.monotonic() + 10
    while not ready.exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    assert ready.exists()
    assert child.wait(timeout=10) == 0

    parent_lock = update_cmd.RepositoryUpdateLock(repo)
    parent_lock.acquire()
    parent_lock.release()


def _write_patch_series(directory: Path, entries: dict[str, bytes]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name, payload in entries.items():
        (directory / name).write_bytes(payload)


def test_conflicting_repo_and_home_patch_series_fail_closed(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    _write_patch_series(repo / "local-patches", {"0001-one.patch": b"repo-only"})
    _write_patch_series(home / "local-patches", {"0001-one.patch": b"home-only"})

    with pytest.raises(RuntimeError, match="Conflicting local patch series") as exc_info:
        update_cmd._local_patches_dir(repo)
    message = str(exc_info.value)
    assert str(repo / "local-patches") in message
    assert str(home / "local-patches") in message
    assert "repo-only" not in message
    assert "home-only" not in message

    with pytest.raises(RuntimeError, match="Conflicting local patch series"):
        update_cmd._snapshot_local_release_patches(repo)


def test_identical_repo_and_home_patch_series_prefer_repo(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    home = tmp_path / "home"
    monkeypatch.setattr(update_cmd, "get_hermes_home", lambda: home)
    entries = {
        "0001-one.patch": b"same-one",
        "0002-two.patch": b"same-two",
    }
    _write_patch_series(repo / "local-patches", entries)
    _write_patch_series(home / "local-patches", entries)

    assert update_cmd._local_patches_dir(repo) == repo / "local-patches"
    assert update_cmd._snapshot_local_release_patches(repo) == [
        ("0001-one.patch", "same-one"),
        ("0002-two.patch", "same-two"),
    ]


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        ("git@github.com:NousResearch/hermes-agent.git", True),
        ("ssh://git@github.com/NousResearch/hermes-agent.git", True),
        ("https://github.com/NousResearch/hermes-agent.git", True),
        ("https://github.com/NousResearch/hermes-agent", True),
        ("https://github.com:443/NousResearch/hermes-agent.git", True),
        ("ssh://git@github.com:22/NousResearch/hermes-agent.git", True),
        ("https://git:secret@github.com/NousResearch/hermes-agent.git", False),
        ("https://github.com:8443/NousResearch/hermes-agent.git", False),
        ("ssh://git@github.com:2222/NousResearch/hermes-agent.git", False),
        ("https://github.com/NousResearch/hermes-agent.git?download=1", False),
        ("https://github.com/NousResearch/hermes-agent.git#main", False),
        ("git@github.com:NousResearch/hermes-agent.git?download=1", False),
        ("https://github.com/NousResearch/other.git", False),
        ("https://gitlab.com/NousResearch/hermes-agent.git", False),
    ],
)
def test_official_origin_url_accepts_only_strict_forms(url, expected):
    assert update_cmd._is_official_origin_url(url) is expected


@pytest.mark.parametrize("artifact_mode", ["missing", "empty", "stale"])
def test_runtime_git_payload_survives_missing_empty_or_stale_patch(tmp_path, artifact_mode):
    repo = _init_repo(tmp_path)
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _git(repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0")
    _git(repo, "branch", "hermes-release", base_sha)
    _git(repo, "switch", "hermes-release")
    (repo / "local.txt").write_text("local commit survives\n", encoding="utf-8")
    local_sha = _commit(repo, "local change after artifact refresh")
    patch_bytes = b"diff --git a/not-the-real-payload b/not-the-real-payload\n"
    _write_base_metadata(repo, tag="v1.0.0", base_sha=base_sha, patch_bytes=patch_bytes)
    patch_path = repo / "local-patches" / "0001-local-maintenance.patch"
    if artifact_mode == "missing":
        patch_path.unlink()
    elif artifact_mode == "empty":
        patch_path.write_bytes(b"")

    payload = update_cmd._generate_runtime_local_payload(
        ["git"], repo, update_cmd._read_release_base_metadata(repo), head_sha=local_sha
    )

    assert b"local commit survives" in payload
    assert b"local-patches" not in payload



def test_local_commit_after_last_patch_refresh_is_in_runtime_payload(tmp_path):
    repo = _init_repo(tmp_path)
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _git(repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0")
    _git(repo, "branch", "hermes-release", base_sha)
    _git(repo, "switch", "hermes-release")
    (repo / "local.txt").write_text("first\n", encoding="utf-8")
    first_sha = _commit(repo, "first local change")
    _write_base_metadata(repo, tag="v1.0.0", base_sha=base_sha, patch_bytes=b"old artifact")
    (repo / "local.txt").write_text("first\nsecond commit\n", encoding="utf-8")
    head_sha = _commit(repo, "local commit after artifact refresh")

    payload = update_cmd._generate_runtime_local_payload(
        ["git"], repo, update_cmd._read_release_base_metadata(repo), head_sha=head_sha
    )

    assert head_sha != first_sha
    assert b"second commit" in payload



def test_exact_tag_resolution_ignores_same_named_branch(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    tag_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _git(repo, "tag", "-a", "v2.0.0", "-m", "release")
    (repo / "branch-only.txt").write_text("branch\n", encoding="utf-8")
    branch_sha = _commit(repo, "same-named branch")
    _git(repo, "branch", "v2.0.0", branch_sha)
    _git(repo, "remote", "add", "origin", str(repo))
    monkeypatch.setattr(update_cmd, "_get_origin_url", lambda _git_cmd, _cwd: "https://github.com/NousResearch/hermes-agent.git")

    resolved = update_cmd._resolve_release_target(["git"], repo, "v2.0.0")

    assert resolved.target_sha == tag_sha
    assert resolved.target_sha != branch_sha
    assert resolved.ref == "refs/hermes-upgrade/tags/v2.0.0"



def test_exact_tag_resolution_rejects_nonofficial_origin(tmp_path, monkeypatch):
    repo = _init_repo(tmp_path)
    _git(repo, "tag", "-a", "v2.0.0", "-m", "release")
    _git(repo, "remote", "add", "origin", str(repo))
    monkeypatch.setattr(update_cmd, "_get_origin_url", lambda _git_cmd, _cwd: "https://evil.example/hermes-agent.git")

    with pytest.raises(RuntimeError, match="official origin"):
        update_cmd._resolve_release_target(["git"], repo, "v2.0.0")



def test_conflict_leaves_live_head_index_and_status_untouched(tmp_path):
    repo = _init_repo(tmp_path)
    base_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _git(repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0")
    _git(repo, "branch", "hermes-release", base_sha)
    _git(repo, "switch", "hermes-release")
    (repo / "README.txt").write_text("maintenance conflicting line\n", encoding="utf-8")
    _commit(repo, "local conflicting change")
    _write_base_metadata(repo, tag="v1.0.0", base_sha=base_sha)
    maintenance_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _git(repo, "switch", "main")
    (repo / "README.txt").write_text("release conflicting line\n", encoding="utf-8")
    target_sha = _commit(repo, "release conflicting change")
    _git(repo, "tag", "-a", "v2.0.0", "-m", "v2.0.0")
    _git(repo, "switch", "hermes-release")
    before_status = _git(repo, "status", "--porcelain=v1", check=True).stdout

    with pytest.raises(RuntimeError, match="candidate"):
        update_cmd._upgrade_release_transaction(["git"], repo, "v2.0.0", target_sha)

    assert _git(repo, "rev-parse", "HEAD").stdout.decode().strip() == maintenance_sha
    assert _git(repo, "status", "--porcelain=v1", check=True).stdout == before_status
    assert _git(repo, "ls-files", "--unmerged").stdout == b""



def test_second_repository_lock_acquisition_fails_without_stealing(tmp_path):
    repo = _init_repo(tmp_path)
    first = update_cmd.RepositoryUpdateLock(repo)
    second = update_cmd.RepositoryUpdateLock(repo)
    first.acquire()
    try:
        with pytest.raises(RuntimeError, match="repository update lock"):
            second.acquire()
        assert first.acquired is True
    finally:
        first.release()
    second.acquire()
    second.release()



def test_binary_mode_symlink_and_rename_payload_survive_two_releases(tmp_path):
    repo = _init_repo(tmp_path)
    (repo / "tracked.txt").write_text("upstream one\n", encoding="utf-8")
    (repo / "rename-me").write_text("rename content\n", encoding="utf-8")
    (repo / "delete-me").write_text("delete\n", encoding="utf-8")
    (repo / "run.sh").write_text("#!/bin/sh\necho base\n", encoding="utf-8")
    (repo / "run.sh").chmod((repo / "run.sh").stat().st_mode | stat.S_IXUSR)
    (repo / "blob.bin").write_bytes(b"base\\x00binary\\xff")
    r1 = _commit(repo, "release one")
    _git(repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0")
    base_sha, _ = _make_maintenance_from_tag(repo, "v1.0.0")
    (repo / "tracked.txt").write_text("upstream one\nlocal addition\n", encoding="utf-8")
    (repo / "blob.bin").write_bytes(b"local\\x00binary\\x00\\xff")
    (repo / "local-added").write_bytes(b"added\\x00")
    (repo / "rename-me-local").write_text("rename content\nlocal\n", encoding="utf-8")
    (repo / "delete-me").unlink()
    (repo / "link-to-local").symlink_to("local-added")
    maintenance_sha = _commit(repo, "local binary mode rename symlink changes")
    _write_base_metadata(repo, tag="v1.0.0", base_sha=base_sha)
    _git(repo, "switch", "main")
    (repo / "upstream-release.txt").write_text("upstream release two\n", encoding="utf-8")
    r2 = _commit(repo, "release two")
    _git(repo, "tag", "-a", "v2.0.0", "-m", "v2.0.0")
    _git(repo, "switch", "hermes-release")

    result = update_cmd._upgrade_release_transaction(["git"], repo, "v2.0.0", r2)

    assert result.candidate_sha == _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    assert (repo / "tracked.txt").read_text(encoding="utf-8") == "upstream one\nlocal addition\n"
    assert (repo / "upstream-release.txt").read_text(encoding="utf-8") == "upstream release two\n"
    assert (repo / "blob.bin").read_bytes() == b"local\\x00binary\\x00\\xff"
    assert (repo / "local-added").read_bytes() == b"added\\x00"
    assert (repo / "rename-me-local").exists()
    assert not (repo / "delete-me").exists()
    assert (repo / "link-to-local").is_symlink()
    assert (repo / "run.sh").stat().st_mode & stat.S_IXUSR
    assert r1 != maintenance_sha

    # A second upstream release must replay the regenerated artifact/runtime
    # payload again, including the binary and symlink state.
    _git(repo, "switch", "main")
    (repo / "upstream-release.txt").write_text("upstream release three\n", encoding="utf-8")
    r3 = _commit(repo, "release three")
    _git(repo, "tag", "-a", "v3.0.0", "-m", "v3.0.0")
    _git(repo, "switch", "hermes-release")
    update_cmd._upgrade_release_transaction(["git"], repo, "v3.0.0", r3)

    assert "upstream release three" in (repo / "upstream-release.txt").read_text(encoding="utf-8")
    assert (repo / "blob.bin").read_bytes() == b"local\\x00binary\\x00\\xff"
    assert (repo / "link-to-local").is_symlink()
    metadata = json.loads((repo / "local-patches" / ".release_base").read_text(encoding="utf-8"))
    assert metadata["tag"] == "v3.0.0"
    assert metadata["base_sha"] == r3
