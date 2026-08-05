"""Real-Git RED coverage for the durable release user-state lifecycle.

These tests deliberately stop the transaction at interruption and finalization
boundaries.  They use temporary repositories so the implementation checkout
and the developer's real stash are never involved.
"""

from __future__ import annotations

import hashlib
import json
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
