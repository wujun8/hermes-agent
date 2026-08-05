"""Real-Git failure-boundary tests for release-upgrade transactions.

Each injected failure is narrow: the wrapper delegates every unrelated command to
real Git and returns one failure only for the named operation.  The fixtures are
intentionally small and use the private candidate-validator seam because they are
not Hermes-shaped repositories; production's strict default remains covered by
the candidate-validation tests.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from hermes_cli import update_cmd


_GIT_USER = {
    "user.email": "upgrade-faults@example.invalid",
    "user.name": "Hermes Upgrade Faults",
}


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if check:
        assert result.returncode == 0, (
            f"git {' '.join(args)} failed ({result.returncode}): "
            f"{result.stderr or result.stdout}"
        )
    return result


def _candidate_validator(git_cmd: list[str], candidate: Path) -> str:
    resolved = update_cmd._git_resolve_commit(git_cmd, candidate, "HEAD")
    assert resolved is not None
    return resolved


def _install_git_fault(monkeypatch, predicate, message: str):
    """Record real Git calls and fail exactly one matching command."""

    real_run = subprocess.run
    calls: list[list[str]] = []
    injected = False

    def run(command, *args, **kwargs):
        nonlocal injected
        normalized = list(command) if isinstance(command, (list, tuple)) else [command]
        calls.append([str(part) for part in normalized])
        if not injected and predicate([str(part) for part in normalized], kwargs.get("cwd")):
            injected = True
            stdout = b"" if kwargs.get("text") is not True else ""
            stderr = b"injected release-upgrade fault\n" if kwargs.get("text") is not True else message
            return subprocess.CompletedProcess(command, 1, stdout=stdout, stderr=stderr)
        return real_run(command, *args, **kwargs)

    # update_cmd and the test helper share the subprocess module object; the
    # wrapper remains honest because every non-matching call delegates to the
    # saved real runner.
    monkeypatch.setattr(update_cmd.subprocess, "run", run)
    return calls


def _install_git_recorder(monkeypatch):
    real_run = subprocess.run
    calls: list[list[str]] = []

    def run(command, *args, **kwargs):
        normalized = list(command) if isinstance(command, (list, tuple)) else [command]
        calls.append([str(part) for part in normalized])
        return real_run(command, *args, **kwargs)

    monkeypatch.setattr(update_cmd.subprocess, "run", run)
    return calls


def _journal_paths(repo: Path) -> list[Path]:
    common = Path(_run_git(repo, "rev-parse", "--git-common-dir").stdout.strip())
    if not common.is_absolute():
        common = (repo / common).resolve()
    root = common / "hermes-upgrade-transactions"
    if not root.exists():
        return []
    return sorted(root.glob("*/journal.json"))


def _read_one_journal(repo: Path) -> dict:
    paths = _journal_paths(repo)
    assert len(paths) == 1, f"expected one durable journal, found {paths}"
    return json.loads(paths[0].read_text(encoding="utf-8"))


def _show_ref(repo: Path, ref: str) -> str | None:
    result = _run_git(repo, "rev-parse", "--verify", f"{ref}^{{commit}}", check=False)
    return result.stdout.strip() if result.returncode == 0 else None


def _assert_user_state(repo: Path, fixture: SimpleNamespace) -> None:
    assert (repo / fixture.user_path).read_bytes() == fixture.user_bytes
    assert (repo / fixture.untracked_path).read_bytes() == fixture.untracked_bytes
    assert _run_git(repo, "status", "--porcelain").stdout == fixture.status
    assert _run_git(repo, "write-tree").stdout.strip() == fixture.index_tree


def _assert_user_bytes_and_status(repo: Path, fixture: SimpleNamespace) -> None:
    assert (repo / fixture.user_path).read_bytes() == fixture.user_bytes
    assert (repo / fixture.untracked_path).read_bytes() == fixture.untracked_bytes
    assert _run_git(repo, "status", "--porcelain").stdout == fixture.status


def _assert_original_maintenance_checkout(repo: Path, fixture: SimpleNamespace) -> None:
    assert _run_git(repo, "symbolic-ref", "--quiet", "--short", "HEAD").stdout.strip() == "hermes-release"
    assert _run_git(repo, "rev-parse", "HEAD").stdout.strip() == fixture.old_sha
    assert _show_ref(repo, "refs/heads/hermes-release") == fixture.old_sha
    _assert_user_state(repo, fixture)


def _prepare_fixture(tmp_path: Path, *, conflict: bool = False) -> SimpleNamespace:
    repo = tmp_path / ("release-fault-conflict" if conflict else "release-faults")
    repo.mkdir()

    def git(*args: str, check: bool = True):
        return _run_git(repo, *args, check=check)

    git("init", "-b", "main")
    for key, value in _GIT_USER.items():
        git("config", key, value)

    (repo / "tracked.txt").write_text("base tracked\n", encoding="utf-8")
    (repo / "user.txt").write_text("clean user\n", encoding="utf-8")
    (repo / "conflict.txt").write_text("base conflict\n", encoding="utf-8")
    git("add", "tracked.txt", "user.txt", "conflict.txt")
    git("commit", "-m", "release base")
    base_sha = git("rev-parse", "HEAD").stdout.strip()
    git("tag", "v1.0.0")

    git("switch", "-c", "hermes-release")
    patches = repo / "local-patches"
    patches.mkdir()
    (repo / "maintenance.txt").write_text("maintenance payload\n", encoding="utf-8")
    (patches / ".release_base").write_text(
        json.dumps({"format_version": 1, "tag": "v1.0.0", "base_sha": base_sha}) + "\n",
        encoding="utf-8",
    )
    git("add", "maintenance.txt", "local-patches/.release_base")
    git("commit", "-m", "local maintenance")
    old_sha = git("rev-parse", "HEAD").stdout.strip()

    git("switch", "main")
    (repo / "release.txt").write_text("release payload\n", encoding="utf-8")
    if conflict:
        (repo / "conflict.txt").write_text("release conflict\n", encoding="utf-8")
    git("add", "release.txt", "conflict.txt")
    git("commit", "-m", "release target")
    git("tag", "v1.2.3")
    target_sha = git("rev-parse", "v1.2.3").stdout.strip()

    git("switch", "hermes-release")
    user_path = "conflict.txt" if conflict else "user.txt"
    user_bytes = b"user version that must survive\n"
    (repo / user_path).write_bytes(user_bytes)
    untracked_path = "user-untracked.bin"
    untracked_bytes = b"\x00user bytes\xff\n"
    (repo / untracked_path).write_bytes(untracked_bytes)
    status = git("status", "--porcelain").stdout
    index_tree = git("write-tree").stdout.strip()

    return SimpleNamespace(
        repo=repo,
        base_sha=base_sha,
        old_sha=old_sha,
        target_sha=target_sha,
        user_path=user_path,
        user_bytes=user_bytes,
        untracked_path=untracked_path,
        untracked_bytes=untracked_bytes,
        status=status,
        index_tree=index_tree,
    )


def _prepare(fixture: SimpleNamespace):
    return update_cmd._prepare_and_promote_release(
        ["git"],
        fixture.repo,
        "v1.2.3",
        fixture.target_sha,
        candidate_validator=_candidate_validator,
    )


def test_fault_f1_worktree_add_failure_restores_real_live_state(tmp_path, monkeypatch):
    fixture = _prepare_fixture(tmp_path)
    calls = _install_git_fault(
        monkeypatch,
        lambda command, _cwd: len(command) >= 3 and command[1:3] == ["worktree", "add"],
        "synthetic worktree-add failure",
    )

    with pytest.raises(RuntimeError, match="isolated upgrade candidate"):
        _prepare(fixture)

    _assert_original_maintenance_checkout(fixture.repo, fixture)
    assert not _journal_paths(fixture.repo)
    assert _show_ref(fixture.repo, "refs/heads/hermes-upgrade-candidate") is None
    assert any(command[1:3] == ["worktree", "add"] for command in calls)
    assert _show_ref(fixture.repo, "refs/hermes-upgrade/backups") is None


def test_fault_f2_candidate_commit_failure_retains_candidate_evidence(tmp_path, monkeypatch):
    fixture = _prepare_fixture(tmp_path)
    real_run = subprocess.run
    seen_candidate = {}

    def fail_after_payload(git_cmd, candidate, message):
        seen_candidate["path"] = candidate
        assert (candidate / "maintenance.txt").read_text(encoding="utf-8") == "maintenance payload\n"
        staged = real_run(
            git_cmd + ["diff", "--cached", "--name-only"],
            cwd=candidate,
            capture_output=True,
            text=True,
            check=True,
        )
        assert "maintenance.txt" in staged.stdout.splitlines()
        raise RuntimeError("synthetic candidate commit failure")

    monkeypatch.setattr(update_cmd, "_commit_candidate_changes", fail_after_payload)

    with pytest.raises(RuntimeError, match="synthetic candidate commit failure"):
        _prepare(fixture)

    _assert_original_maintenance_checkout(fixture.repo, fixture)
    journal = _read_one_journal(fixture.repo)
    assert journal["phase"] != "finalized"
    assert journal["candidate_created"] is True
    assert journal["candidate_cleanup"] is False
    assert journal["candidate_path"] == str(seen_candidate["path"])
    assert journal["candidate_branch"].startswith("hermes-upgrade-candidate/")
    assert journal["candidate_sha"] is None
    assert journal["backup_created"] is True
    assert _show_ref(fixture.repo, journal["backup_ref"]) == fixture.old_sha
    assert Path(journal["candidate_path"]).exists()
    assert _show_ref(fixture.repo, f"refs/heads/{journal['candidate_branch']}") is not None


def test_fault_f3_compare_and_swap_failure_preserves_ref_and_evidence(tmp_path, monkeypatch):
    fixture = _prepare_fixture(tmp_path)
    calls = _install_git_fault(
        monkeypatch,
        lambda command, _cwd: (
            len(command) == 5
            and command[1] == "update-ref"
            and command[2] == "refs/heads/hermes-release"
            and command[4] == fixture.old_sha
        ),
        "synthetic compare-and-swap failure",
    )

    with pytest.raises(RuntimeError, match="changed before promotion"):
        _prepare(fixture)

    _assert_original_maintenance_checkout(fixture.repo, fixture)
    journal = _read_one_journal(fixture.repo)
    assert journal["phase"] != "finalized"
    assert journal["maintenance_old_sha"] == fixture.old_sha
    assert journal["candidate_sha"] is not None
    assert journal["backup_created"] is True
    assert _show_ref(fixture.repo, journal["backup_ref"]) == fixture.old_sha
    assert _show_ref(fixture.repo, "refs/heads/hermes-release") == fixture.old_sha
    assert Path(journal["candidate_path"]).exists()
    assert any(
        command[1:2] == ["update-ref"]
        and command[2:3] == ["refs/heads/hermes-release"]
        for command in calls
    )


def test_fault_f4_post_cas_reset_failure_retains_actionable_durable_recovery(
    tmp_path, monkeypatch, capsys
):
    fixture = _prepare_fixture(tmp_path)
    calls = _install_git_fault(
        monkeypatch,
        lambda command, _cwd: (
            len(command) == 4
            and command[1:3] == ["reset", "--hard"]
            and command[3] != "HEAD"
        ),
        "synthetic post-CAS reset failure",
    )

    with pytest.raises(RuntimeError, match="Promotion did not verify") as exc_info:
        _prepare(fixture)

    output = capsys.readouterr().out + str(exc_info.value)
    journal_paths = _journal_paths(fixture.repo)
    assert len(journal_paths) == 1
    journal = json.loads(journal_paths[0].read_text(encoding="utf-8"))
    assert journal["maintenance_old_sha"] == fixture.old_sha
    assert journal["old_sha"] == fixture.old_sha
    assert journal["candidate_sha"] is not None
    assert journal["backup_ref"]
    assert journal["stash_sha"]
    assert journal["stash_pending"] is False
    assert journal["phase"] != "finalized"
    assert _show_ref(fixture.repo, journal["backup_ref"]) == fixture.old_sha
    assert _show_ref(fixture.repo, "refs/heads/hermes-release") == journal["candidate_sha"]
    assert _run_git(fixture.repo, "rev-parse", "HEAD").stdout.strip() == journal["candidate_sha"]
    assert _run_git(fixture.repo, "symbolic-ref", "--short", "HEAD").stdout.strip() == "hermes-release"
    _assert_user_bytes_and_status(fixture.repo, fixture)
    candidate_tree = _run_git(
        fixture.repo, "rev-parse", f"{journal['candidate_sha']}^{{tree}}"
    ).stdout.strip()
    assert _run_git(fixture.repo, "write-tree").stdout.strip() == candidate_tree
    assert "git update-ref" in output
    assert journal["old_sha"] in output
    assert journal["candidate_sha"] in output
    assert journal["backup_ref"] in output
    assert any(command[1:3] == ["reset", "--hard"] for command in calls)


def test_fault_f5_cleanup_failure_keeps_journal_and_retries_fail_closed(tmp_path, monkeypatch):
    fixture = _prepare_fixture(tmp_path)
    calls = _install_git_fault(
        monkeypatch,
        lambda command, _cwd: (
            len(command) == 5
            and command[1:4] == ["worktree", "remove", "--force"]
        ),
        "synthetic candidate cleanup failure",
    )

    result = _prepare(fixture)
    assert result.context is not None
    assert result.candidate_path is not None
    assert result.context.journal["candidate_cleanup"] is False
    assert _show_ref(fixture.repo, result.backup_ref) == fixture.old_sha

    assert update_cmd._finalize_release_upgrade(
        ["git"], fixture.repo, result.context
    ) is False
    _assert_user_bytes_and_status(fixture.repo, fixture)
    journal = _read_one_journal(fixture.repo)
    assert journal["phase"] != "finalized"
    assert journal["candidate_cleanup"] is False
    assert journal["candidate_path"] == str(result.candidate_path)
    assert journal["candidate_branch"]
    assert journal["candidate_sha"] == result.candidate_sha
    assert journal["backup_ref"] == result.backup_ref
    assert "stash_pending" in journal
    assert Path(journal["candidate_path"]).exists()
    assert _show_ref(fixture.repo, f"refs/heads/{journal['candidate_branch']}") == result.candidate_sha

    journals_before = _journal_paths(fixture.repo)
    with pytest.raises(RuntimeError, match="unfinished release transaction"):
        _prepare(fixture)
    assert _journal_paths(fixture.repo) == journals_before
    assert len([c for c in calls if c[1:3] == ["worktree", "add"]]) == 1


def test_fault_f6_stash_conflict_keeps_immutable_stash_and_does_not_double_apply_or_drop(
    tmp_path, monkeypatch, capsys
):
    fixture = _prepare_fixture(tmp_path, conflict=True)
    calls = _install_git_recorder(monkeypatch)

    result = _prepare(fixture)
    assert result.context is not None
    context = result.context
    stash_sha = context.journal["stash_sha"]
    assert stash_sha

    assert update_cmd._finalize_release_upgrade(["git"], fixture.repo, context) is False
    first_output = capsys.readouterr().out
    journal = _read_one_journal(fixture.repo)
    assert journal["phase"] == "stash-restore-conflict"
    assert journal["stash_sha"] == stash_sha
    assert journal["stash_pending"] is True
    assert journal["stash_apply_attempted"] is True
    assert journal["stash_applied"] is False
    assert journal["checkout_restored"] is True
    assert _run_git(fixture.repo, "symbolic-ref", "--short", "HEAD").stdout.strip() == journal["original_branch"]
    assert _run_git(fixture.repo, "rev-parse", "HEAD").stdout.strip() == journal["candidate_sha"]
    assert _show_ref(fixture.repo, stash_sha) == stash_sha
    assert f"git stash apply {stash_sha}" in first_output
    assert stash_sha in first_output

    # A retry after a conflict must leave the immutable stash for an operator;
    # it must not silently drop it merely because apply was already attempted.
    assert update_cmd._finalize_release_upgrade(["git"], fixture.repo, context) is False
    second_output = capsys.readouterr().out
    assert _show_ref(fixture.repo, stash_sha) == stash_sha
    journal_after = _read_one_journal(fixture.repo)
    assert journal_after["stash_pending"] is True
    assert journal_after["stash_apply_attempted"] is True
    assert journal_after["stash_applied"] is False
    assert journal_after["phase"] != "finalized"
    assert sum(command[1:3] == ["stash", "apply"] for command in calls) == 1
    assert sum(command[1:3] == ["stash", "drop"] for command in calls) == 0
    assert "git stash apply" not in second_output
