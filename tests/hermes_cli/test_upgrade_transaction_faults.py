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


def _run_git_bytes(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        capture_output=True,
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


@pytest.mark.parametrize("hide_immutable_verification", [False, True])
def test_release_nonzero_stash_push_requires_independent_immutable_verification(
    tmp_path, monkeypatch, hide_immutable_verification
):
    """A nonzero push is usable only after release-owned SHA verification."""
    fixture = _prepare_fixture(tmp_path)
    real_run = subprocess.run
    calls: list[list[str]] = []

    def run(command, *args, **kwargs):
        normalized = [str(part) for part in command]
        calls.append(normalized)
        if (
            hide_immutable_verification
            and len(normalized) >= 2
            and normalized[1] == "cat-file"
        ):
            stdout = "" if kwargs.get("text") is True else b""
            stderr = "hidden immutable stash" if kwargs.get("text") is True else b"hidden immutable stash"
            return subprocess.CompletedProcess(command, 1, stdout, stderr)
        result = real_run(command, *args, **kwargs)
        if len(normalized) >= 3 and normalized[1:3] == ["stash", "push"]:
            return subprocess.CompletedProcess(command, 1, result.stdout, result.stderr)
        return result

    monkeypatch.setattr(update_cmd.subprocess, "run", run)

    if hide_immutable_verification:
        with pytest.raises(RuntimeError, match="immutable stash"):
            _prepare(fixture)
        journal = _assert_capture_uncertain_journal(fixture.repo)
        stash_sha = _stash_sha_for_marker(fixture.repo, journal["stash_marker"])
        assert stash_sha is not None
        assert _show_ref(fixture.repo, stash_sha) == stash_sha
        _assert_no_uncertain_capture_mutators(calls)
    else:
        result = _prepare(fixture)
        assert result.context is not None
        journal = result.context.journal
        assert journal["stash_capture_confirmed"] is True
        assert journal["stash_capture_uncertain"] is False
        assert journal["stash_sha"] == _stash_sha_for_marker(
            fixture.repo, journal["stash_marker"]
        )
        assert any(
            len(command) >= 2 and command[1] == "cat-file" for command in calls
        )


def _capture_fixture(tmp_path: Path) -> SimpleNamespace:
    fixture = _prepare_fixture(tmp_path)
    repo = fixture.repo
    _run_git(repo, "config", "core.autocrlf", "false")
    staged_bytes = b"staged user bytes\r\nintentional CRLF\r\n"
    working_bytes = b"working user bytes\r\nintentional CRLF\r\n"
    (repo / fixture.user_path).write_bytes(staged_bytes)
    _run_git(repo, "add", "--", fixture.user_path)
    (repo / fixture.user_path).write_bytes(working_bytes)
    fixture.untracked_bytes = b"\x00untracked binary\xff\r\n"
    (repo / fixture.untracked_path).write_bytes(fixture.untracked_bytes)
    untracked_text_path = "user-untracked.txt"
    untracked_text_bytes = b"untracked text\r\nintentional CRLF\r\n"
    (repo / untracked_text_path).write_bytes(untracked_text_bytes)
    return SimpleNamespace(
        fixture=fixture,
        staged_bytes=staged_bytes,
        working_bytes=working_bytes,
        untracked_text_path=untracked_text_path,
        untracked_text_bytes=untracked_text_bytes,
        status_v2=_run_git_bytes(repo, "status", "--porcelain=v2").stdout,
        cached_diff=_run_git_bytes(repo, "diff", "--cached", "--binary").stdout,
        index_entries=_run_git_bytes(repo, "ls-files", "--stage", "-z").stdout,
        index_tree=_run_git_bytes(repo, "write-tree").stdout.strip(),
    )


def _assert_capture_fixture_unchanged(snapshot: SimpleNamespace) -> None:
    repo = snapshot.fixture.repo
    assert (repo / snapshot.fixture.user_path).read_bytes() == snapshot.working_bytes
    assert (repo / snapshot.fixture.untracked_path).read_bytes() == snapshot.fixture.untracked_bytes
    assert (repo / snapshot.untracked_text_path).read_bytes() == snapshot.untracked_text_bytes
    assert _run_git_bytes(repo, "status", "--porcelain=v2").stdout == snapshot.status_v2
    assert _run_git_bytes(repo, "diff", "--cached", "--binary").stdout == snapshot.cached_diff
    assert _run_git_bytes(repo, "ls-files", "--stage", "-z").stdout == snapshot.index_entries
    assert _run_git_bytes(repo, "write-tree").stdout.strip() == snapshot.index_tree


def _context_from_journal(repo: Path) -> update_cmd.ReleaseUpgradeContext:
    journal_path = _journal_paths(repo)[0]
    common_dir = update_cmd._git_common_dir(["git"], repo)
    return update_cmd.ReleaseUpgradeContext(
        root=repo,
        common_dir=common_dir,
        transaction_dir=journal_path.parent,
        journal_path=journal_path,
        journal=json.loads(journal_path.read_text(encoding="utf-8")),
    )


def _stash_sha_for_marker(repo: Path, marker: str) -> str | None:
    listing = _run_git(repo, "stash", "list", "--format=%H%x00%gs").stdout
    for line in listing.splitlines():
        commit, separator, subject = line.partition("\x00")
        if separator and marker in subject:
            return commit.strip()
    return None


def _assert_no_uncertain_capture_mutators(calls: list[list[str]]) -> None:
    assert not any(command[1:3] == ["reset", "--hard"] for command in calls)
    assert not any(command[1:2] == ["clean"] for command in calls)
    assert not any(command[1:2] == ["checkout"] for command in calls)
    assert not any(command[1:3] in (["stash", "apply"], ["stash", "drop"]) for command in calls)


def _assert_capture_uncertain_journal(repo: Path) -> dict:
    journal = _read_one_journal(repo)
    assert journal["local_state_present"] is True
    assert journal["stash_capture_required"] is True
    assert journal["stash_capture_confirmed"] is False
    assert journal["stash_capture_uncertain"] is True
    assert journal["stash_sha"] is None
    assert journal["phase"] == "stash-capture-uncertain"
    return journal


def test_stash_helper_failure_before_git_mutation_preserves_bytes_index_and_journal(
    tmp_path, monkeypatch
):
    snapshot = _capture_fixture(tmp_path)
    calls = _install_git_recorder(monkeypatch)

    def fail_before_git_mutation(*_args, **_kwargs):
        raise RuntimeError("injected stash capture failure")

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", fail_before_git_mutation)
    with pytest.raises(RuntimeError, match="injected stash capture failure"):
        _prepare(snapshot.fixture)

    _assert_capture_fixture_unchanged(snapshot)
    _assert_capture_uncertain_journal(snapshot.fixture.repo)
    _assert_no_uncertain_capture_mutators(calls)

    before = (
        snapshot.fixture.user_path,
        snapshot.fixture.repo / snapshot.untracked_text_path,
        _run_git_bytes(snapshot.fixture.repo, "status", "--porcelain=v2").stdout,
        _run_git_bytes(snapshot.fixture.repo, "diff", "--cached", "--binary").stdout,
    )
    assert update_cmd._finalize_release_upgrade(
        ["git"], snapshot.fixture.repo, _context_from_journal(snapshot.fixture.repo)
    ) is False
    _assert_capture_fixture_unchanged(snapshot)
    assert (
        snapshot.fixture.user_path,
        snapshot.fixture.repo / snapshot.untracked_text_path,
        _run_git_bytes(snapshot.fixture.repo, "status", "--porcelain=v2").stdout,
        _run_git_bytes(snapshot.fixture.repo, "diff", "--cached", "--binary").stdout,
    ) == before


def test_nonzero_real_stash_push_fails_closed_without_reset_clean_or_journal_deletion(
    tmp_path, monkeypatch
):
    snapshot = _capture_fixture(tmp_path)
    calls = _install_git_fault(
        monkeypatch,
        lambda command, _cwd: command[1:3] == ["stash", "push"],
        "injected nonzero stash push",
    )

    with pytest.raises(subprocess.CalledProcessError):
        _prepare(snapshot.fixture)

    _assert_capture_fixture_unchanged(snapshot)
    _assert_capture_uncertain_journal(snapshot.fixture.repo)
    _assert_no_uncertain_capture_mutators(calls)


def test_successful_stash_with_unavailable_identity_keeps_stash_evidence_and_is_idempotent(
    tmp_path, monkeypatch, capsys
):
    snapshot = _capture_fixture(tmp_path)
    real_stash = update_cmd._stash_local_changes_if_needed

    def stash_but_hide_identity(*args, **kwargs):
        assert real_stash(*args, **kwargs)
        return None

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", stash_but_hide_identity)
    monkeypatch.setattr(update_cmd, "_refresh_transaction_stash_identity", lambda *_args: None)

    with pytest.raises(RuntimeError, match="Stash capture"):
        _prepare(snapshot.fixture)

    repo = snapshot.fixture.repo
    journal = _assert_capture_uncertain_journal(repo)
    stash_sha = _stash_sha_for_marker(repo, journal["stash_marker"])
    assert stash_sha is not None
    assert _show_ref(repo, stash_sha) == stash_sha
    assert _run_git(repo, "status", "--porcelain=v2").stdout == ""
    assert _run_git_bytes(repo, "show", f"{stash_sha}:{snapshot.fixture.user_path}").stdout == snapshot.working_bytes
    assert _run_git_bytes(repo, "show", f"{stash_sha}^3:{snapshot.fixture.untracked_path}").stdout == snapshot.fixture.untracked_bytes
    assert "stash" in capsys.readouterr().out.lower()

    before_stash = _show_ref(repo, stash_sha)
    assert update_cmd._finalize_release_upgrade(
        ["git"], repo, _context_from_journal(repo)
    ) is False
    assert _show_ref(repo, stash_sha) == before_stash
    assert _read_one_journal(repo)["phase"] == "stash-capture-uncertain"


def test_post_push_exception_refreshes_unique_stash_sha_and_restores_exact_state(
    tmp_path, monkeypatch
):
    snapshot = _capture_fixture(tmp_path)
    calls = _install_git_recorder(monkeypatch)
    real_stash = update_cmd._stash_local_changes_if_needed

    def stash_then_raise(*args, **kwargs):
        assert real_stash(*args, **kwargs)
        raise RuntimeError("injected post-push exception")

    monkeypatch.setattr(update_cmd, "_stash_local_changes_if_needed", stash_then_raise)
    with pytest.raises(RuntimeError, match="injected post-push exception"):
        _prepare(snapshot.fixture)

    _assert_capture_fixture_unchanged(snapshot)
    assert not _journal_paths(snapshot.fixture.repo)
    assert _run_git(snapshot.fixture.repo, "stash", "list").stdout == ""
    assert sum(command[1:3] == ["stash", "apply"] for command in calls) == 1
    assert sum(command[1:3] == ["stash", "drop"] for command in calls) == 1


def test_no_local_state_failure_keeps_existing_safe_cleanup_contract(tmp_path, monkeypatch):
    fixture = _prepare_fixture(tmp_path)
    repo = fixture.repo
    _run_git(repo, "switch", "main")
    original_sha = _run_git(repo, "rev-parse", "HEAD").stdout.strip()
    _run_git(repo, "clean", "-fd")
    _run_git(repo, "reset", "--hard", "HEAD")
    calls = _install_git_recorder(monkeypatch)
    monkeypatch.setattr(
        update_cmd,
        "_upgrade_release_transaction",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("later candidate failure")),
    )

    with pytest.raises(RuntimeError, match="later candidate failure"):
        _prepare(fixture)

    assert _run_git(repo, "symbolic-ref", "--short", "HEAD").stdout.strip() == "main"
    assert _run_git(repo, "rev-parse", "HEAD").stdout.strip() == original_sha
    assert _run_git(repo, "status", "--porcelain=v2").stdout == ""
    assert not _journal_paths(repo)
    assert any(command[1:3] == ["reset", "--hard"] for command in calls)
    assert any(command[1:2] == ["clean"] for command in calls)
