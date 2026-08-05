"""Real-Git RED coverage for the durable release user-state lifecycle.

These tests deliberately stop the transaction at interruption and finalization
boundaries.  They use temporary repositories so the implementation checkout
and the developer's real stash are never involved.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd



def _git(repo: Path, *args: str, check: bool = True):
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
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


def _install_fsync_probe(monkeypatch, *, failure_path: Path | None = None):
    """Record real fsync paths and optionally fail one directory fsync."""

    events: list[tuple[str, Path | None]] = []
    fd_paths: dict[int, Path] = {}
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        path = fd_paths.get(fd)
        events.append(("fsync", path))
        if failure_path is not None and path == failure_path:
            raise OSError(f"injected directory fsync failure: {path}")
        return real_fsync(fd)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "fsync", tracked_fsync)
    return events, fd_paths



def _commit(repo: Path, message: str) -> str:
    _git(repo, "add", "--all")
    _git(repo, "commit", "-m", message)
    return _git(repo, "rev-parse", "HEAD").stdout.decode().strip()



def _test_candidate_validator(git_cmd, candidate):
    """Explicit seam for non-Hermes-shaped temporary repositories."""
    resolved = update_cmd._git_resolve_commit(git_cmd, candidate, "HEAD")
    assert resolved is not None
    return resolved



def _release_repo(
    tmp_path: Path,
    *,
    conflicting_payload: bool = False,
    target_package_lock: bool = False,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Lifecycle Test")
    _git(repo, "config", "user.email", "hermes-lifecycle@example.invalid")
    (repo / "README.txt").write_text("base\n", encoding="utf-8")
    (repo / "package-lock.json").write_text('{"lockfileVersion": 3}\n', encoding="utf-8")
    base_sha = _commit(repo, "base")
    _git(repo, "tag", "-a", "v1.0.0", "-m", "v1.0.0")

    _git(repo, "branch", "hermes-release", base_sha)
    _git(repo, "switch", "hermes-release")
    if conflicting_payload:
        (repo / "README.txt").write_text("maintenance\n", encoding="utf-8")
        _commit(repo, "committed maintenance conflict")
    patches = repo / "local-patches"
    patches.mkdir()
    (patches / "0001-local-maintenance.patch").write_bytes(b"")
    (patches / ".release_base").write_text(
        json.dumps(
            {
                "tag": "v1.0.0",
                "base_sha": base_sha,
                "target_sha": base_sha,
                "patch_sha256": hashlib.sha256(b"").hexdigest(),
                "patch_bytes": 0,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (patches / "README.md").write_text("metadata\n", encoding="utf-8")
    maintenance_sha = _commit(repo, "record release metadata")

    _git(repo, "switch", "main")
    if conflicting_payload:
        (repo / "README.txt").write_text("release\n", encoding="utf-8")
    else:
        (repo / "release.txt").write_text("release\n", encoding="utf-8")
    if target_package_lock:
        (repo / "package-lock.json").write_bytes(b'{"lockfileVersion": 3, "release": true}\n')
    target_sha = _commit(repo, "release v2")
    _git(repo, "tag", "-a", "v2.0.0", "-m", "v2.0.0")
    _git(repo, "switch", "hermes-release")
    return repo, base_sha, maintenance_sha, target_sha



def _journal_paths(repo: Path) -> list[Path]:
    common = update_cmd._git_common_dir(["git"], repo)
    return sorted((common / "hermes-upgrade-transactions").glob("*/journal.json"))



def _latest_journal(repo: Path) -> tuple[Path, dict]:
    paths = _journal_paths(repo)
    assert paths, "expected a durable release transaction journal"
    path = paths[-1]
    return path, json.loads(path.read_text(encoding="utf-8"))



def _stash_sha_for_marker(repo: Path, marker: str) -> str | None:
    listing = _git(repo, "stash", "list", "--format=%H%x00%gs").stdout.decode(
        errors="replace"
    )
    for line in listing.splitlines():
        sha, separator, subject = line.partition("\0")
        if separator and marker in subject:
            return sha.strip()
    return None



def _assert_exact_checkout(repo: Path, branch: str | None, sha: str) -> None:
    assert _git(repo, "rev-parse", "HEAD").stdout.decode().strip() == sha
    if branch is None:
        assert (
            _git(repo, "rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip()
            == "HEAD"
        )
    else:
        assert (
            _git(repo, "symbolic-ref", "--short", "HEAD").stdout.decode().strip()
            == branch
        )



def _capture_user_state(repo: Path, paths: tuple[str, ...]) -> dict:
    return {
        "bytes": {
            path: (repo / path).read_bytes() if (repo / path).exists() else None
            for path in paths
        },
        "status": _git(repo, "status", "--porcelain=v2").stdout,
        "cached_diff": _git(repo, "diff", "--cached", "--binary").stdout,
        "unstaged_diff": _git(repo, "diff", "--binary").stdout,
        "index_entries": _git(repo, "ls-files", "--stage", "-z").stdout,
        "index_tree": _git(repo, "write-tree").stdout,
    }



def _write_user_state(repo: Path, prefix: str) -> dict:
    _git(repo, "config", "core.autocrlf", "false")
    tracked = repo / "README.txt"
    tracked.write_bytes(f"{prefix} staged\r\nsecond staged line\r\n".encode())
    _git(repo, "add", "--", tracked.name)
    tracked.write_bytes(f"{prefix} working\r\nsecond working line\r\n".encode())
    binary_name = f"{prefix}-binary.bin"
    crlf_name = f"{prefix}-crlf.txt"
    (repo / binary_name).write_bytes(b"\x00" + prefix.encode() + b"\xff\x00\r\n")
    (repo / crlf_name).write_bytes(
        f"{prefix} untracked\r\nintentional CRLF\r\n".encode()
    )
    return _capture_user_state(repo, ("README.txt", binary_name, crlf_name))



def _assert_captured_user_state(
    repo: Path, expected: dict, *, include_index: bool = True
) -> None:
    actual = _capture_user_state(repo, tuple(expected["bytes"]))
    if include_index:
        assert actual == expected
        return
    for key in ("bytes", "status", "cached_diff", "unstaged_diff"):
        assert actual[key] == expected[key]
    expected_readme = next(
        entry for entry in expected["index_entries"].split(b"\0") if entry.endswith(b"\tREADME.txt")
    )
    actual_readme = next(
        entry for entry in actual["index_entries"].split(b"\0") if entry.endswith(b"\tREADME.txt")
    )
    assert actual_readme == expected_readme



def test_keyboard_interrupt_after_stash_restores_exact_branch_and_bytes(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    original_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    lockfile = repo / "package-lock.json"
    lock_bytes = b'{"lockfileVersion": 3, "intentional": true}\n'
    lockfile.write_bytes(lock_bytes)
    user_file = repo / "user-notes.bin"
    user_bytes = b"user\x00bytes\xff\n"
    user_file.write_bytes(user_bytes)

    real_stash = update_cmd._stash_local_changes_if_needed

    def stash_then_interrupt(*args, **kwargs):
        real_stash(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", stash_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        update_cmd._prepare_and_promote_release(
            ["git"], repo, "v2.0.0", target_sha,
            candidate_validator=_test_candidate_validator,
        )

    _assert_exact_checkout(repo, "main", original_sha)
    assert lockfile.read_bytes() == lock_bytes
    assert user_file.read_bytes() == user_bytes
    journal_paths = _journal_paths(repo)
    if journal_paths:
        _path, journal = _latest_journal(repo)
        assert journal["original_head_sha"] == original_sha
        assert journal["stash_pending"] is False or journal["stash_sha"]
        if journal["stash_pending"]:
            assert _git(repo, "cat-file", "-e", f"{journal['stash_sha']}^{{commit}}", check=False).returncode == 0



def test_keyboard_interrupt_after_promotion_keeps_candidate_and_restores_once(
    tmp_path, monkeypatch
):
    repo, _base_sha, maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    original_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    lockfile = repo / "package-lock.json"
    lock_bytes = b'{"lockfileVersion": 3, "intentional": true}\n'
    lockfile.write_bytes(lock_bytes)
    user_file = repo / "user-notes.txt"
    user_file.write_text("preserve me\n", encoding="utf-8")

    real_transaction = update_cmd._upgrade_release_transaction

    def promote_then_interrupt(*args, **kwargs):
        real_transaction(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(update_cmd, "_upgrade_release_transaction", promote_then_interrupt)

    with pytest.raises(KeyboardInterrupt):
        update_cmd._prepare_and_promote_release(
            ["git"], repo, "v2.0.0", target_sha,
            candidate_validator=_test_candidate_validator,
        )

    _assert_exact_checkout(repo, "main", original_sha)
    assert lockfile.read_bytes() == lock_bytes
    assert user_file.read_text(encoding="utf-8") == "preserve me\n"
    promoted_sha = _git(repo, "rev-parse", "refs/heads/hermes-release").stdout.decode().strip()
    assert promoted_sha not in {maintenance_sha, original_sha}
    journal_paths = _journal_paths(repo)
    if journal_paths:
        _path, journal = _latest_journal(repo)
        assert journal["stash_pending"] is False



def test_durable_journal_records_payload_and_stash_identity_at_interrupt(
    tmp_path, monkeypatch
):
    repo, base_sha, maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    original_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    (repo / "package-lock.json").write_bytes(b'{"intentional":true}\n')
    (repo / "untracked.bin").write_bytes(b"untracked\x00\xff")

    real_stash = update_cmd._stash_local_changes_if_needed

    def stash_then_interrupt(*args, **kwargs):
        real_stash(*args, **kwargs)
        raise KeyboardInterrupt()

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", stash_then_interrupt)
    monkeypatch.setattr(update_cmd, "_restore_stashed_changes", lambda *args, **kwargs: False)

    with pytest.raises(KeyboardInterrupt):
        update_cmd._prepare_and_promote_release(
            ["git"], repo, "v2.0.0", target_sha,
            candidate_validator=_test_candidate_validator,
        )

    _path, journal = _latest_journal(repo)
    assert journal["original_branch"] == "main"
    assert journal["original_head_sha"] == original_sha
    assert journal["maintenance_old_sha"] == maintenance_sha
    assert journal["base_sha"] == base_sha
    assert journal["target_sha"] == target_sha
    assert journal["stash_marker"]
    assert journal["stash_pending"] is True
    assert journal["stash_sha"] == _stash_sha_for_marker(repo, journal["stash_marker"])
    payload_path = Path(journal["payload_path"])
    assert payload_path.exists()
    assert stat.S_ISREG(payload_path.lstat().st_mode)
    payload = payload_path.read_bytes()
    assert hashlib.sha256(payload).hexdigest() == journal["payload_sha256"]
    assert len(payload) == journal["payload_bytes"]
    assert "phase" in journal
    assert journal["backup_ref"]



def test_upgrade_transaction_leaves_journal_for_outer_finalizer(tmp_path):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    result = update_cmd._upgrade_release_transaction(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )

    path, journal = _latest_journal(repo)
    assert path.exists()
    assert journal["candidate_sha"] == result.candidate_sha
    assert journal["candidate_cleanup"] is True
    assert journal["stash_pending"] is False



def test_delayed_restore_hides_intentional_files_from_pipeline_and_restores_after_failure(
    tmp_path,
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    original_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    lockfile = repo / "package-lock.json"
    lock_bytes = b'{"lockfileVersion": 3, "intentional": true}\n'
    lockfile.write_bytes(lock_bytes)
    user_file = repo / "user-untracked.txt"
    user_bytes = b"keep this exact file\n"
    user_file.write_bytes(user_bytes)

    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert not lockfile.exists() or lockfile.read_bytes() != lock_bytes
    assert not user_file.exists()

    observed = {}

    def mocked_node_or_build_step():
        observed["lock"] = lockfile.read_bytes() if lockfile.exists() else None
        observed["untracked"] = user_file.exists()
        return []

    mocked_node_or_build_step()
    assert observed["lock"] != lock_bytes
    assert observed["untracked"] is False
    assert result.context is not None

    with pytest.raises(RuntimeError, match="post-promotion pipeline failure"):
        try:
            raise RuntimeError("post-promotion pipeline failure")
        finally:
            assert update_cmd._finalize_release_upgrade(
                ["git"], repo, result.context
            ) is True

    _assert_exact_checkout(repo, "main", original_sha)
    assert lockfile.read_bytes() == lock_bytes
    assert user_file.read_bytes() == user_bytes
    assert not _journal_paths(repo)



def test_detached_failure_restores_exact_sha_and_status(tmp_path):
    repo, base_sha, _maintenance_sha, target_sha = _release_repo(
        tmp_path, conflicting_payload=True
    )
    _git(repo, "checkout", "--detach", base_sha)
    (repo / "user-only.txt").write_text("detached user state\n", encoding="utf-8")
    before_status = _git(repo, "status", "--porcelain=v1").stdout

    with pytest.raises(RuntimeError, match="candidate"):
        update_cmd._prepare_and_promote_release(
            ["git"], repo, "v2.0.0", target_sha,
            candidate_validator=_test_candidate_validator,
        )

    _assert_exact_checkout(repo, None, base_sha)
    assert _git(repo, "status", "--porcelain=v1").stdout == before_status
    assert (repo / "user-only.txt").read_text(encoding="utf-8") == "detached user state\n"



def test_stash_restore_conflict_keeps_immutable_stash_and_journal(tmp_path):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(
        tmp_path, target_package_lock=True
    )
    lockfile = repo / "package-lock.json"
    lockfile.write_bytes(b'{"lockfileVersion": 3, "user": true}\n')
    original_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()

    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    _path, before = _latest_journal(repo)
    assert before["stash_pending"] is True
    stash_sha = before["stash_sha"]

    restored = update_cmd._finalize_release_upgrade(
        ["git"], repo, result.context
    )
    assert restored is False
    _assert_exact_checkout(repo, "hermes-release", _git(repo, "rev-parse", "HEAD").stdout.decode().strip())
    assert _git(repo, "cat-file", "-e", f"{stash_sha}^{{commit}}", check=False).returncode == 0
    _path, after = _latest_journal(repo)
    assert after["stash_pending"] is True
    assert after["stash_sha"] == stash_sha
    assert after["original_head_sha"] == original_sha



def test_verified_release_finalizer_is_idempotent_and_preserves_post_finalization_state(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    before = _write_user_state(repo, "before")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context

    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
    _assert_captured_user_state(repo, before, include_index=False)
    assert context.journal["phase"] == "finalized"
    assert context.journal["final_state_verified"] is True
    assert not context.journal_path.exists()

    after = _write_user_state(repo, "after")
    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert calls == []

    _assert_captured_user_state(repo, after)
    assert context.journal["phase"] == "finalized"
    assert context.journal["final_state_verified"] is True



def test_verified_finalizer_retries_only_journal_ack_after_unlink_failure(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _write_user_state(repo, "before")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    journal_path = context.journal_path
    real_unlink = Path.unlink
    fail_once = True

    def fail_journal_unlink(path, *args, **kwargs):
        nonlocal fail_once
        if path == journal_path and fail_once:
            fail_once = False
            raise OSError("injected journal unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_journal_unlink)
    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert journal_path.exists()
    assert context.journal["phase"] == "finalized"
    assert context.journal["final_state_verified"] is True
    persisted = json.loads(journal_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "finalized"
    assert persisted["final_state_verified"] is True

    after = _write_user_state(repo, "after-unlink-failure")
    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert calls == []

    assert not journal_path.exists()
    _assert_captured_user_state(repo, after)

    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        calls: list[list[str]] = []

        def record_third_run(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_third_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert calls == []

    _assert_captured_user_state(repo, after)



@pytest.mark.parametrize("verified_marker", [False, None])
def test_inconsistent_finalized_marker_fails_closed_without_git_or_state_mutation(
    tmp_path, monkeypatch, verified_marker
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    update_cmd._journal_update(context, "finalized", finalized=False)
    if verified_marker is None:
        context.journal.pop("final_state_verified", None)
        update_cmd._write_transaction_journal(
            context.common_dir, context.journal, path=context.journal_path
        )
    else:
        assert context.journal["final_state_verified"] is False

    expected = _write_user_state(repo, "inconsistent")
    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
        assert calls == []

    _assert_captured_user_state(repo, expected)
    assert context.journal["phase"] == "finalized"
    assert context.journal_path.exists()
    journal = json.loads(context.journal_path.read_text(encoding="utf-8"))
    assert journal["phase"] == "finalized"
    if verified_marker is None:
        assert "final_state_verified" not in journal
    else:
        assert journal["final_state_verified"] is False


def test_existing_transaction_root_is_fsynced_after_child_mkdir_before_journal_and_stash(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    common = update_cmd._git_common_dir(["git"], repo)
    transactions = common / "hermes-upgrade-transactions"
    transactions.mkdir()
    events, _fd_paths = _install_fsync_probe(monkeypatch)
    real_mkdir = Path.mkdir

    def record_transaction_mkdir(path, *args, **kwargs):
        if path.parent == transactions:
            events.append(("mkdir", path))
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", record_transaction_mkdir)
    real_stash = update_cmd._stash_local_changes_if_needed

    def record_stash(*args, **kwargs):
        events.append(("stash", None))
        return real_stash(*args, **kwargs)

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", record_stash)

    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    child = next(path for kind, path in events if kind == "mkdir")
    child_mkdir_index = next(i for i, (kind, path) in enumerate(events) if kind == "mkdir" and path == child)
    root_fsync_index = next(
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path == transactions
    )
    journal_temp_index = next(
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync"
        and path is not None
        and path.parent == child
        and path.name.startswith(".journal.json.")
    )
    stash_index = next(i for i, (kind, _path) in enumerate(events) if kind == "stash")

    assert child.parent == transactions
    assert child_mkdir_index < root_fsync_index < journal_temp_index < stash_index
    assert update_cmd._finalize_release_upgrade(["git"], repo, result.context) is True


def test_new_transaction_root_fsyncs_common_dir_before_child_and_root_entries(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    common = update_cmd._git_common_dir(["git"], repo)
    transactions = common / "hermes-upgrade-transactions"
    assert not transactions.exists()
    events, _fd_paths = _install_fsync_probe(monkeypatch)
    real_mkdir = Path.mkdir

    def record_transaction_mkdir(path, *args, **kwargs):
        if path == transactions or path.parent == transactions:
            events.append(("mkdir", path))
        return real_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "mkdir", record_transaction_mkdir)
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    root_mkdir_index = next(
        i for i, (kind, path) in enumerate(events)
        if kind == "mkdir" and path == transactions
    )
    child = next(
        path for kind, path in events
        if kind == "mkdir" and path is not None and path.parent == transactions
    )
    child_mkdir_index = next(i for i, (kind, path) in enumerate(events) if kind == "mkdir" and path == child)
    common_fsync_index = next(
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path == common
    )
    root_fsync_index = next(
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path == transactions
    )

    assert root_mkdir_index < common_fsync_index < child_mkdir_index < root_fsync_index
    assert update_cmd._finalize_release_upgrade(["git"], repo, result.context) is True


def test_transaction_atomic_replaces_fsync_each_temp_file_and_child_directory(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, _target_sha = _release_repo(tmp_path)
    common = update_cmd._git_common_dir(["git"], repo)
    (common / "hermes-upgrade-transactions").mkdir()
    events, _fd_paths = _install_fsync_probe(monkeypatch)

    context = update_cmd._create_release_upgrade_context(
        ["git"], repo,
        original_branch="hermes-release",
        original_head_sha="a" * 40,
        maintenance_old_sha="b" * 40,
        release_tag="v2.0.0",
        base_sha="c" * 40,
        target_sha="d" * 40,
        payload=b"transaction payload\n",
    )
    child = context.transaction_dir
    transaction_temp_paths = [
        path
        for kind, path in events
        if kind == "fsync"
        and path is not None
        and path.parent == child
        and path.name.startswith(".")
    ]
    journal_temps = [path for path in transaction_temp_paths if path.name.startswith(".journal.json.")]
    payload_temps = [
        path
        for path in transaction_temp_paths
        if path.name.startswith(".runtime-local-maintenance.patch.")
    ]
    child_fsync_indices = [
        i for i, (kind, path) in enumerate(events) if kind == "fsync" and path == child
    ]

    assert len(journal_temps) == 2
    assert len(payload_temps) == 1
    assert len(child_fsync_indices) >= 3
    for temp_path in journal_temps + payload_temps:
        temp_index = next(
            i for i, (kind, path) in enumerate(events)
            if kind == "fsync" and path == temp_path
        )
        assert any(index > temp_index for index in child_fsync_indices)


def test_transactions_root_fsync_failure_aborts_before_stash_and_preserves_real_git_state(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    expected = _write_user_state(repo, "root-fsync-failure")
    common = update_cmd._git_common_dir(["git"], repo)
    transactions = common / "hermes-upgrade-transactions"
    transactions.mkdir()
    _events, _fd_paths = _install_fsync_probe(monkeypatch, failure_path=transactions)
    stash_calls: list[object] = []
    candidate_calls: list[object] = []

    def unexpected_stash(*_args, **_kwargs):
        stash_calls.append(True)
        raise AssertionError("stash helper must not run before transaction root durability")

    def unexpected_candidate(*_args, **_kwargs):
        candidate_calls.append(True)
        raise AssertionError("candidate mutator must not run before transaction root durability")

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", unexpected_stash)
    monkeypatch.setattr(update_cmd, "_upgrade_release_transaction", unexpected_candidate)
    git_commands: list[list[str]] = []
    real_run = update_cmd.subprocess.run

    def record_run(command, *args, **kwargs):
        git_commands.append([str(part) for part in command])
        return real_run(command, *args, **kwargs)

    with monkeypatch.context() as patcher:
        patcher.setattr(update_cmd.subprocess, "run", record_run)
        with pytest.raises(OSError, match="directory fsync failure"):
            update_cmd._prepare_and_promote_release(
                ["git"], repo, "v2.0.0", target_sha,
                candidate_validator=_test_candidate_validator,
            )

    assert stash_calls == []
    assert candidate_calls == []
    assert _capture_user_state(repo, tuple(expected["bytes"])) == expected
    mutating_commands = {
        "add", "apply", "branch", "checkout", "clean", "commit", "reset",
        "stash", "switch", "update-ref", "worktree",
    }
    assert not any(len(command) > 1 and command[1] in mutating_commands for command in git_commands)
    children = [path for path in transactions.iterdir() if path.is_dir()]
    assert children
    assert all(not any(child.iterdir()) for child in children)


def test_transaction_child_fsync_failure_after_journal_replace_aborts_before_stash(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    expected = _write_user_state(repo, "child-fsync-failure")
    common = update_cmd._git_common_dir(["git"], repo)
    transactions = common / "hermes-upgrade-transactions"
    transactions.mkdir()
    transaction_id = "child-fsync-failure"
    child = transactions / transaction_id
    monkeypatch.setattr(update_cmd.uuid, "uuid4", lambda: type("UUID", (), {"hex": transaction_id})())
    _events, _fd_paths = _install_fsync_probe(monkeypatch, failure_path=child)
    stash_calls: list[object] = []

    def unexpected_stash(*_args, **_kwargs):
        stash_calls.append(True)
        raise AssertionError("stash helper must not run after child journal fsync failure")

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", unexpected_stash)

    with pytest.raises(OSError, match="directory fsync failure"):
        update_cmd._prepare_and_promote_release(
            ["git"], repo, "v2.0.0", target_sha,
            candidate_validator=_test_candidate_validator,
        )

    assert stash_calls == []
    assert _capture_user_state(repo, tuple(expected["bytes"])) == expected
    assert child.is_dir()
    assert (child / "journal.json").is_file()
    assert not (child / "runtime-local-maintenance.patch").exists()


def test_final_ack_fsyncs_child_after_known_entries_then_root_after_rmdir(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    update_cmd._mark_release_finalized(context)
    child = context.transaction_dir
    transactions = child.parent
    journal_path = context.journal_path
    payload_path = child / "runtime-local-maintenance.patch"
    events, _fd_paths = _install_fsync_probe(monkeypatch)
    real_unlink = Path.unlink
    real_rmdir = Path.rmdir

    def record_unlink(path, *args, **kwargs):
        if path in {journal_path, payload_path}:
            events.append(("unlink", path))
        return real_unlink(path, *args, **kwargs)

    def record_rmdir(path, *args, **kwargs):
        if path == child:
            events.append(("rmdir", path))
        return real_rmdir(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", record_unlink)
    monkeypatch.setattr(Path, "rmdir", record_rmdir)

    assert update_cmd._acknowledge_finalized_release(context) is True
    unlink_indices = [i for i, (kind, _path) in enumerate(events) if kind == "unlink"]
    child_fsync_index = next(
        i for i, (kind, path) in enumerate(events) if kind == "fsync" and path == child
    )
    rmdir_index = next(i for i, (kind, _path) in enumerate(events) if kind == "rmdir")
    root_fsync_index = next(
        i for i, (kind, path) in enumerate(events)
        if kind == "fsync" and path == transactions and i > rmdir_index
    )

    assert set(path for kind, path in events if kind == "unlink") == {journal_path, payload_path}
    assert max(unlink_indices) < child_fsync_index < rmdir_index < root_fsync_index
    assert not child.exists()


def test_final_ack_does_not_delete_unknown_transaction_evidence(tmp_path):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    update_cmd._mark_release_finalized(context)
    unknown = context.transaction_dir / "operator-evidence.txt"
    unknown.write_text("keep this evidence\n", encoding="utf-8")

    assert update_cmd._acknowledge_finalized_release(context) is False
    assert unknown.read_text(encoding="utf-8") == "keep this evidence\n"
    assert context.transaction_dir.is_dir()
    assert not context.journal_path.exists()
    assert not (context.transaction_dir / "runtime-local-maintenance.patch").exists()


def test_final_ack_child_fsync_failure_retries_ack_only_and_preserves_post_finalize_state(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, "ack-child-fsync")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    child = context.transaction_dir
    fd_paths: dict[int, Path] = {}
    events: list[tuple[str, Path | None]] = []
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync
    armed = False
    failed = False

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        nonlocal failed
        path = fd_paths.get(fd)
        events.append(("fsync", path))
        if armed and not failed and path == child:
            failed = True
            raise OSError(f"injected final acknowledgment fsync failure: {path}")
        return real_fsync(fd)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "fsync", tracked_fsync)
    real_mark = update_cmd._mark_release_finalized

    def mark_and_arm(mark_context):
        nonlocal armed
        real_mark(mark_context)
        armed = True

    monkeypatch.setattr(update_cmd, "_mark_release_finalized", mark_and_arm)

    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert failed is True
    assert context.journal["phase"] == "finalized"
    assert context.journal["final_state_verified"] is True
    assert not context.journal_path.exists()
    post_finalize = _write_user_state(repo, "post-finalize")

    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        git_calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert git_calls == []

    assert not context.transaction_dir.exists()
    assert _capture_user_state(repo, tuple(post_finalize["bytes"])) == post_finalize


def test_terminal_marker_fsync_failure_latches_without_exposing_terminal(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, "terminal-marker-uncertain")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    child = context.transaction_dir
    fd_paths: dict[int, Path] = {}
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync
    armed = False
    failed = False

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        nonlocal failed
        path = fd_paths.get(fd)
        if armed and not failed and path == child:
            failed = True
            raise OSError(f"injected terminal marker fsync failure: {path}")
        return real_fsync(fd)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "fsync", tracked_fsync)
    real_mark = update_cmd._mark_release_finalized

    def arm_before_mark(mark_context):
        nonlocal armed
        armed = True
        return real_mark(mark_context)

    monkeypatch.setattr(update_cmd, "_mark_release_finalized", arm_before_mark)

    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert failed is True
    assert context.journal["phase"] == "finalizing"
    assert context.final_marker_write_uncertain is True
    assert context.final_marker_candidate["phase"] == "finalized"
    persisted = json.loads(context.journal_path.read_text(encoding="utf-8"))
    assert persisted["phase"] == "finalized"
    assert persisted["final_state_verified"] is True

    post_failure = _write_user_state(repo, "after-terminal-marker-failure")
    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        git_calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert git_calls == []

    _assert_captured_user_state(repo, post_failure)
    assert context.journal["phase"] == "finalized"
    assert context.journal["final_state_verified"] is True
    assert not context.transaction_dir.exists()


def test_uncertain_terminal_marker_retry_failure_stays_latched_and_git_free(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, "uncertain-retry-failure")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    child = context.transaction_dir
    fd_paths: dict[int, Path] = {}
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync
    armed = False
    failures = 0

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        nonlocal failures
        path = fd_paths.get(fd)
        if armed and path == child and failures < 2:
            failures += 1
            raise OSError(f"injected terminal marker fsync failure {failures}: {path}")
        return real_fsync(fd)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "fsync", tracked_fsync)
    real_mark = update_cmd._mark_release_finalized

    def arm_before_mark(mark_context):
        nonlocal armed
        armed = True
        return real_mark(mark_context)

    monkeypatch.setattr(update_cmd, "_mark_release_finalized", arm_before_mark)
    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert context.final_marker_write_uncertain is True
    assert context.journal["phase"] == "finalizing"
    post_failure = _write_user_state(repo, "after-persistent-retry-failure")

    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        git_calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
        assert git_calls == []
        assert context.final_marker_write_uncertain is True
        assert context.journal["phase"] == "finalizing"
        assert context.journal_path.exists()
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert git_calls == []

    _assert_captured_user_state(repo, post_failure)
    assert not context.transaction_dir.exists()


def test_uncertain_terminal_marker_replace_failure_reconciles_prior_journal_git_free(
    tmp_path, monkeypatch
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, "replace-did-not-happen")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    real_replace = update_cmd.os.replace
    armed = False
    failed = False

    def guarded_replace(source, destination):
        nonlocal failed
        if armed and Path(destination) == context.journal_path:
            failed = True
            raise OSError("injected terminal marker replace failure")
        return real_replace(source, destination)

    monkeypatch.setattr(update_cmd.os, "replace", guarded_replace)
    real_mark = update_cmd._mark_release_finalized

    def arm_before_mark(mark_context):
        nonlocal armed
        armed = True
        return real_mark(mark_context)

    monkeypatch.setattr(update_cmd, "_mark_release_finalized", arm_before_mark)
    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert failed is True
    assert context.final_marker_write_uncertain is True
    assert context.journal["phase"] == "finalizing"
    persisted = json.loads(context.journal_path.read_text(encoding="utf-8"))
    assert persisted == context.journal
    armed = False

    post_failure = _write_user_state(repo, "after-replace-failure")
    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        git_calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert git_calls == []

    assert context.journal["phase"] == "finalized"
    assert context.journal["final_state_verified"] is True
    _assert_captured_user_state(repo, post_failure)
    assert not context.transaction_dir.exists()


@pytest.mark.parametrize(
    "tamper",
    ["missing", "tampered", "different-transaction", "invalid-finalized", "escape", "symlink"],
)
def test_uncertain_terminal_marker_tampering_fails_closed_without_ack_or_git(
    tmp_path, monkeypatch, tamper
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, f"tamper-{tamper}")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    child = context.transaction_dir
    fd_paths: dict[int, Path] = {}
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync
    armed = False
    failed = False

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        nonlocal failed
        path = fd_paths.get(fd)
        if armed and not failed and path == child:
            failed = True
            raise OSError("injected terminal marker fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "fsync", tracked_fsync)
    real_mark = update_cmd._mark_release_finalized

    def arm_before_mark(mark_context):
        nonlocal armed
        armed = True
        return real_mark(mark_context)

    monkeypatch.setattr(update_cmd, "_mark_release_finalized", arm_before_mark)
    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert failed is True
    assert context.final_marker_write_uncertain is True
    assert context.journal["phase"] == "finalizing"

    if tamper == "missing":
        context.journal_path.unlink()
    elif tamper == "symlink":
        evidence = context.transaction_dir / "journal-evidence.json"
        context.journal_path.replace(evidence)
        context.journal_path.symlink_to(evidence.name)
    else:
        disk = json.loads(context.journal_path.read_text(encoding="utf-8"))
        if tamper == "tampered":
            disk["payload_sha256"] = "0" * 64
        elif tamper == "different-transaction":
            disk["transaction_id"] = "attacker-controlled"
        elif tamper == "invalid-finalized":
            disk["phase"] = "finalized"
            disk["state"] = "finalized"
            disk["final_state_verified"] = False
        elif tamper == "escape":
            disk["payload_path"] = str(tmp_path / "outside-payload")
        context.journal_path.write_text(
            json.dumps(disk, sort_keys=True, indent=2) + "\n", encoding="utf-8"
        )

    with monkeypatch.context() as patcher:
        git_calls: list[list[str]] = []

        def forbidden_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            raise AssertionError("uncertain reconciliation invoked Git/subprocess")

        def forbidden_ack(*args, **kwargs):
            raise AssertionError("uncertain reconciliation acknowledged tampered evidence")

        patcher.setattr(update_cmd.subprocess, "run", forbidden_run)
        patcher.setattr(update_cmd, "_acknowledge_finalized_release", forbidden_ack)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
        assert git_calls == []

    assert context.final_marker_write_uncertain is True
    assert context.journal["phase"] == "finalizing"
    assert child.exists()
    if tamper == "missing":
        assert not context.journal_path.exists()
    elif tamper == "symlink":
        assert context.journal_path.is_symlink()


@pytest.mark.parametrize("write_exception", [BaseException, KeyboardInterrupt])
def test_terminal_marker_write_baseexception_latches_before_propagation(
    tmp_path, monkeypatch, write_exception
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, "terminal-marker-baseexception")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    update_cmd._journal_update(context, "finalizing")
    real_write = update_cmd._write_transaction_journal
    armed = True

    def fail_terminal_write(common_dir, payload, *, path=None):
        if armed and payload.get("phase") == "finalized":
            raise write_exception("injected terminal marker write exception")
        return real_write(common_dir, payload, path=path)

    monkeypatch.setattr(update_cmd, "_write_transaction_journal", fail_terminal_write)
    with pytest.raises(write_exception):
        update_cmd._mark_release_finalized(context)

    assert context.final_marker_write_uncertain is True
    assert context.final_marker_candidate["phase"] == "finalized"
    assert context.journal["phase"] == "finalizing"
    assert json.loads(context.journal_path.read_text(encoding="utf-8")) == context.journal

    armed = False
    with monkeypatch.context() as patcher:
        real_run = subprocess.run
        git_calls: list[list[str]] = []

        def record_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            return real_run(command, *args, **kwargs)

        patcher.setattr(update_cmd.subprocess, "run", record_run)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is True
        assert git_calls == []

    assert not context.transaction_dir.exists()


@pytest.mark.parametrize("candidate_tamper", ["transaction-id", "payload-escape", "marker", "prior"])
def test_uncertain_terminal_marker_candidate_mismatch_fails_closed(
    tmp_path, monkeypatch, candidate_tamper
):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    _git(repo, "switch", "main")
    _write_user_state(repo, f"candidate-{candidate_tamper}")
    result = update_cmd._prepare_and_promote_release(
        ["git"], repo, "v2.0.0", target_sha,
        candidate_validator=_test_candidate_validator,
    )
    assert result.context is not None
    context = result.context
    child = context.transaction_dir
    fd_paths: dict[int, Path] = {}
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync
    armed = False
    failed = False

    def tracked_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        fd_paths[fd] = Path(path)
        return fd

    def tracked_fsync(fd):
        nonlocal failed
        path = fd_paths.get(fd)
        if armed and not failed and path == child:
            failed = True
            raise OSError("injected terminal marker fsync failure")
        return real_fsync(fd)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "fsync", tracked_fsync)
    real_mark = update_cmd._mark_release_finalized

    def arm_before_mark(mark_context):
        nonlocal armed
        armed = True
        return real_mark(mark_context)

    monkeypatch.setattr(update_cmd, "_mark_release_finalized", arm_before_mark)
    assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
    assert context.final_marker_write_uncertain is True

    if candidate_tamper == "transaction-id":
        context.final_marker_candidate["transaction_id"] = "attacker-controlled"
    elif candidate_tamper == "payload-escape":
        context.final_marker_candidate["payload_path"] = str(tmp_path / "outside-payload")
    elif candidate_tamper == "marker":
        context.final_marker_candidate["final_state_verified"] = False
    else:
        context.journal["target_sha"] = "0" * 40

    with monkeypatch.context() as patcher:
        git_calls: list[list[str]] = []

        def forbidden_run(command, *args, **kwargs):
            git_calls.append([str(part) for part in command])
            raise AssertionError("candidate reconciliation invoked Git/subprocess")

        def forbidden_ack(*args, **kwargs):
            raise AssertionError("candidate reconciliation acknowledged mismatched evidence")

        patcher.setattr(update_cmd.subprocess, "run", forbidden_run)
        patcher.setattr(update_cmd, "_acknowledge_finalized_release", forbidden_ack)
        assert update_cmd._finalize_release_upgrade(["git"], repo, context) is False
        assert git_calls == []

    assert context.final_marker_write_uncertain is True
    assert context.journal_path.exists()
    assert child.exists()


def test_directory_fsync_closes_fd_on_success_and_required_failure(tmp_path, monkeypatch):
    if os.name != "posix":
        pytest.skip("POSIX directory fsync semantics")
    real_open = update_cmd.os.open
    real_fsync = update_cmd.os.fsync
    real_close = update_cmd.os.close

    for fail in (False, True):
        opened: list[int] = []
        closed: list[int] = []

        def tracked_open(path, flags, *args):
            fd = real_open(path, flags, *args)
            opened.append(fd)
            return fd

        def tracked_close(fd):
            closed.append(fd)
            return real_close(fd)

        def tracked_fsync(fd):
            if fail:
                raise OSError("injected fsync failure")
            return real_fsync(fd)

        with monkeypatch.context() as patcher:
            patcher.setattr(update_cmd.os, "open", tracked_open)
            patcher.setattr(update_cmd.os, "close", tracked_close)
            patcher.setattr(update_cmd.os, "fsync", tracked_fsync)
            if fail:
                with pytest.raises(OSError, match="injected fsync failure"):
                    update_cmd._fsync_directory(tmp_path, required=True)
            else:
                update_cmd._fsync_directory(tmp_path, required=True)

        assert opened == closed


def test_directory_fsync_windows_path_is_explicit_best_effort(tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(
        update_cmd,
        "_directory_fsync_is_windows",
        lambda: True,
        raising=False,
    )

    def fail_open(*_args, **_kwargs):
        raise OSError("directory handles unsupported")

    monkeypatch.setattr(update_cmd.os, "open", fail_open)
    with caplog.at_level("WARNING"):
        update_cmd._fsync_directory(tmp_path, required=True)
    assert "best effort" in caplog.text.lower()
