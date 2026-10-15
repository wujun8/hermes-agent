"""Ordinary updates preserve user EOL bytes, Git configuration and staged state.

Historical managed checkouts may use autocrlf=true with an LF index and CRLF
worktree. Those bytes can also be intentional user edits, so the compatibility
hook leaves them alone and the ordinary update protects them through autostash.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from hermes_cli.update_cmd import _normalize_managed_eol
from types import SimpleNamespace
from hermes_cli import main as hermes_main
from hermes_cli import update_cmd

pytestmark = pytest.mark.skipif(shutil.which("git") is None, reason="needs git")

GIT_CMD = ["git"]


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )


def _managed_repo(tmp_path: Path, files: dict[str, bytes]) -> Path:
    """A checkout in the broken state: LF in the index, CRLF in the worktree,
    ``core.autocrlf=true`` still set — i.e. what Git for Windows leaves behind.

    The CRLF is written by git's own checkout rather than by hand, which is how a
    real managed install gets there. It matters: git's stat cache then records the
    CRLF size, so the churn reads *clean* under ``autocrlf=true`` and is only
    visible once the pin is evaluated.
    """
    repo = tmp_path / "managed"
    repo.mkdir(parents=True)
    _git(repo, "init")
    for name, body in files.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    _git(repo, "-c", "core.autocrlf=false", "add", "-A")
    _git(repo, "commit", "-m", "init")
    _git(repo, "config", "core.autocrlf", "true")
    for name in files:
        (repo / name).unlink()
    _git(repo, "checkout", "--", ".")
    # Deterministic dirtiness: whether `git diff` content-checks an entry (and
    # so sees the CRLF churn) or trusts the stat cache depends on racy-git
    # detection — entries whose mtime equals the index timestamp get content-
    # compared, later ones read clean. On a fast runner a large checkout
    # straddles that boundary nondeterministically (CI flake: 92/661 of 1200
    # dirty). Bump every worktree mtime past the index write so ALL entries
    # are stat-stale and git must content-compare each one.
    import os as _os
    import time as _time

    bumped = _time.time() + 5
    for name in files:
        _os.utime(repo / name, (bumped, bumped))
    return repo


def _dirty(repo: Path) -> set[str]:
    out = subprocess.run(
        ["git", "-c", "core.autocrlf=false", "diff", "--name-only"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    )
    return {line for line in out.stdout.splitlines() if line}


def _autocrlf(repo: Path) -> str:
    out = subprocess.run(
        ["git", "config", "--local", "--get", "core.autocrlf"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=False,
    )
    return out.stdout.strip()


def _repository_snapshot(repo: Path) -> dict:
    paths = _git_bytes(repo, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    porcelain = _git_bytes(repo, "--no-optional-locks", "status", "--porcelain=v1", "-z")
    return {
        "worktree": {name: (repo / name).read_bytes() for name in paths.decode().split("\0") if name},
        "config": (repo / ".git/config").read_bytes(),
        "index": (repo / ".git/index").read_bytes(),
        "index_entries": _git_bytes(repo, "ls-files", "--stage", "-z"),
        "porcelain": porcelain,
    }


def test_crlf_invisible_under_autocrlf_true_is_preserved(tmp_path: Path) -> None:
    """A clean status never authorizes replacing CRLF bytes or pinning config."""
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\ny = 2\n"})
    as_git_sees_it = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, capture_output=True, text=True, check=True
    )
    assert as_git_sees_it.stdout.strip() == ""
    assert b"\r\n" in (repo / "a.py").read_bytes()

    before = _repository_snapshot(repo)
    _normalize_managed_eol(GIT_CMD, repo, checkout_move=None)

    assert _repository_snapshot(repo) == before
    assert _dirty(repo) == {"a.py"}
    assert b"\r\n" in (repo / "a.py").read_bytes()
    assert _autocrlf(repo) == "true"


def test_crlf_bytes_and_original_autocrlf_config_are_preserved(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n", "b.md": b"# Title\n\nbody\n"})

    before = _repository_snapshot(repo)
    _normalize_managed_eol(GIT_CMD, repo, checkout_move=None)

    assert _repository_snapshot(repo) == before
    assert _dirty(repo) == {"a.py", "b.md"}
    assert b"\r\n" in (repo / "a.py").read_bytes()
    assert _autocrlf(repo) == "true"


def test_real_edits_survive_even_when_line_endings_also_flipped(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"churn.py": b"y = 2\n", "both.py": b"z = 3\n"})
    # A genuine edit that ALSO got renormalized must not be discarded.
    (repo / "both.py").write_bytes(b"z = 3\r\nz += 1\r\n")

    before = _repository_snapshot(repo)
    _normalize_managed_eol(GIT_CMD, repo, checkout_move=None)

    assert _repository_snapshot(repo) == before
    assert _dirty(repo) == {"both.py", "churn.py"}
    assert (repo / "both.py").read_bytes() == b"z = 3\r\nz += 1\r\n"
    assert _autocrlf(repo) == "true"


def test_no_config_pin_is_written_when_there_is_no_crlf(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    (repo / "a.py").write_bytes(b"x = 1\n")

    before = _repository_snapshot(repo)
    _normalize_managed_eol(GIT_CMD, repo, checkout_move=None)

    assert _repository_snapshot(repo) == before
    assert _dirty(repo) == set()
    assert _autocrlf(repo) == "true"


@pytest.mark.platforms("posix")  # shim needs a POSIX shell
def test_pin_is_withheld_when_the_churn_cannot_be_cleared(tmp_path: Path) -> None:
    """If normalization can't finish, the checkout is left as found — pinning
    anyway would surface churn we failed to clear."""
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    # Real git everywhere except the restore, which fails. Stands in for any
    # reason it cannot finish: a git too old for --pathspec-from-file, an
    # unwritable working tree, a file locked by a running process.
    shim = tmp_path / "git-no-checkout"
    shim.write_text(
        '#!/bin/sh\nfor a in "$@"; do [ "$a" = checkout ] && exit 1; done\nexec git "$@"\n'
    )
    shim.chmod(0o755)

    _normalize_managed_eol([str(shim)], repo)

    assert _autocrlf(repo) == "true"
    assert b"\r\n" in (repo / "a.py").read_bytes()


def test_already_pinned_checkout_is_untouched(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    _git(repo, "config", "core.autocrlf", "false")

    _normalize_managed_eol(GIT_CMD, repo)

    # Nothing to repair from this function's perspective: autocrlf is not true,
    # so it must not start restoring files behind the user's back.
    assert _dirty(repo) == {"a.py"}


def test_autocrlf_input_is_left_alone(tmp_path: Path) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\n"})
    _git(repo, "config", "core.autocrlf", "input")

    _normalize_managed_eol(GIT_CMD, repo)

    assert _autocrlf(repo) == "input"


def test_large_crlf_tree_preserves_every_file_config_index_and_porcelain(tmp_path: Path) -> None:
    """Even a 1200-file historical checkout is never normalized behind the user."""
    files = {f"pkg/mod_{i:04d}.py": f"VALUE = {i}\n".encode() for i in range(1200)}
    repo = _managed_repo(tmp_path, files)
    assert len(_dirty(repo)) == len(files)

    before = _repository_snapshot(repo)
    _normalize_managed_eol(GIT_CMD, repo, checkout_move=None)

    assert _repository_snapshot(repo) == before
    assert len(_dirty(repo)) == len(files)
    assert _autocrlf(repo) == "true"


def test_non_repo_is_ignored(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    _normalize_managed_eol(GIT_CMD, tmp_path)

    assert capsys.readouterr().out == ""


def _git_bytes(cwd: Path, *args: str) -> bytes:
    return subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
    ).stdout

def _make_local_update_repos(tmp_path: Path) -> tuple[Path, Path]:
    """Create a local checkout plus a local-only origin with one update."""
    origin = tmp_path / "origin.git"
    repo = tmp_path / "checkout"
    upstream = tmp_path / "upstream-work"

    origin.mkdir()
    repo.mkdir()
    _git(origin, "init", "--bare")
    _git(origin, "symbolic-ref", "HEAD", "refs/heads/main")
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "tests@example.invalid")
    _git(repo, "config", "user.name", "Hermes EOL Tests")
    (repo / "tracked.py").write_bytes(b"VALUE = 1\n")
    _git(repo, "add", "tracked.py")
    _git(repo, "commit", "-m", "initial")
    _git(repo, "remote", "add", "origin", str(origin))
    _git(repo, "push", "-u", "origin", "main")

    subprocess.run(
        ["git", "clone", str(origin), str(upstream)],
        check=True,
        capture_output=True,
        text=True,
    )
    _git(upstream, "config", "user.email", "tests@example.invalid")
    _git(upstream, "config", "user.name", "Hermes EOL Tests")
    (upstream / "remote.txt").write_bytes(b"remote update\n")
    _git(upstream, "add", "remote.txt")
    _git(upstream, "commit", "-m", "remote update")
    _git(upstream, "push", "origin", "main")
    return repo, origin

def _patch_ordinary_update_runtime(monkeypatch, repo: Path) -> None:
    """Exercise real Git/stash while isolating the PM completion and live fleet."""
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", repo)
    monkeypatch.setattr(hermes_main, "_run_pre_update_backup", lambda *_a, **_k: None)
    monkeypatch.setattr(hermes_main, "_pause_windows_gateways_for_update", lambda *_a, **_k: None)
    monkeypatch.setattr(hermes_main, "_resume_windows_gateways_after_update", lambda *_a, **_k: None)
    monkeypatch.setattr(update_cmd._posix_pause, "arm_pause", lambda **_kwargs: None)
    monkeypatch.setattr(hermes_main, "_installed_desktop_apps", list)
    monkeypatch.setattr(hermes_main, "_desktop_dist_exists", lambda _root: False)
    monkeypatch.setattr(hermes_main, "_desktop_packaged_executable", lambda _root: None)
    monkeypatch.setattr(
        hermes_main,
        "_get_origin_url",
        lambda _git_cmd, _repo: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(hermes_main, "_resolve_update_branch", lambda _args: "main")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_FILES", ("tracked.py",))
    monkeypatch.setattr(
        update_cmd, "_critical_module_import_failures", lambda *_a, **_k: {}
    )
    monkeypatch.setattr(
        update_cmd, "_validate_critical_modules_import", lambda *_a, **_k: (True, None, None)
    )

    def complete(request):
        # PM's child owns dependency builds and fleet restarts after this Git boundary.
        assert request["expected_sha"] == _git_bytes(repo, "rev-parse", "HEAD").decode().strip()

    monkeypatch.setattr(update_cmd, "_complete_source_update", complete)
    monkeypatch.setattr(
        "hermes_cli.update_inventory.collect_runtime_inventory",
        lambda **_kwargs: SimpleNamespace(runtimes=[], to_dict=dict),
    )
    monkeypatch.setattr(
        "hermes_cli.update_inventory.record_plan_in_receipt", lambda *_a, **_k: None
    )
    monkeypatch.setattr("hermes_cli.config.load_config", dict)

def _run_ordinary_update(repo: Path, monkeypatch, *, dirty: bool) -> tuple[list, list, list]:
    _patch_ordinary_update_runtime(monkeypatch, repo)
    tracked = repo / "tracked.py"
    user_bytes = b"VALUE = 1\r\n# intentional CRLF edit\r\n"
    untracked_bytes = b"unrelated\x00bytes\r\n"
    if dirty:
        _git(repo, "config", "core.autocrlf", "false")
        tracked.write_bytes(user_bytes)
        (repo / "unrelated.bin").write_bytes(untracked_bytes)

    events: list[str] = []
    git_calls: list[list[str]] = []
    from hermes_cli import update_custody

    real_run_git = update_custody.run_git

    def recording_run_git(git_cmd, args, **kwargs):
        git_calls.append([str(git_cmd[0]), *(str(part) for part in args)])
        return real_run_git(git_cmd, args, **kwargs)

    original_eol = update_cmd._normalize_managed_eol

    def eol_noop(*args, **kwargs):
        events.append("eol")
        return original_eol(*args, **kwargs)

    original_stash = update_cmd._stash_local_changes_if_needed
    original_restore = update_cmd._restore_stashed_changes
    original_post_pull_guard = update_cmd._validate_post_pull_critical_files_syntax

    def stash(*args, **kwargs):
        events.append("stash")
        return original_stash(*args, **kwargs)

    def restore(*args, **kwargs):
        events.append("restore")
        return original_restore(*args, **kwargs)

    def post_pull_guard(*args, **kwargs):
        events.append("post-pull")
        return original_post_pull_guard(*args, **kwargs)

    monkeypatch.setattr(update_custody, "run_git", recording_run_git)
    monkeypatch.setattr(update_cmd, "_normalize_managed_eol", eol_noop)
    monkeypatch.setattr(
        update_cmd,
        "_validate_post_pull_critical_files_syntax",
        post_pull_guard,
    )
    monkeypatch.setattr(hermes_main, "_stash_local_changes_if_needed", stash)
    monkeypatch.setattr(hermes_main, "_restore_stashed_changes", restore)

    args = SimpleNamespace(
        release_tag=None,
        branch=None,
        yes=True,
        force=False,
        force_venv=False,
    )
    update_cmd._cmd_update_impl(args, gateway_mode=False)
    return events, git_calls, [user_bytes, untracked_bytes]

def test_normalize_managed_eol_preserves_crlf_worktree_index_and_porcelain_exactly(
    tmp_path: Path,
) -> None:
    repo = _managed_repo(tmp_path, {"a.py": b"x = 1\ny = 2\n"})
    path = repo / "a.py"
    intentional = b"x = 1\r\ny = 2\r\n# intentional\r\n"
    path.write_bytes(intentional)

    before = {
        "worktree": path.read_bytes(),
        "index_diff": _git_bytes(repo, "diff", "--cached", "--binary", "--full-index"),
        "porcelain": _git_bytes(repo, "status", "--porcelain=v1", "-z"),
        "autocrlf": _autocrlf(repo),
    }

    update_cmd._normalize_managed_eol(GIT_CMD, repo)

    assert path.read_bytes() == before["worktree"]
    assert _git_bytes(repo, "diff", "--cached", "--binary", "--full-index") == before["index_diff"]
    assert _git_bytes(repo, "status", "--porcelain=v1", "-z") == before["porcelain"]
    assert _autocrlf(repo) == before["autocrlf"] == "true"

def test_normalize_managed_eol_never_runs_checkout_reset_or_clean(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def fail_if_called(cmd, *args, **kwargs):
        calls.append([str(part) for part in cmd])
        raise AssertionError(f"EOL compatibility hook ran Git: {cmd!r}")

    monkeypatch.setattr(update_cmd.subprocess, "run", fail_if_called)
    update_cmd._normalize_managed_eol(GIT_CMD, tmp_path)

    assert calls == []

def test_ordinary_update_autostash_round_trip_preserves_crlf_and_untracked_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _origin = _make_local_update_repos(tmp_path)

    events, git_calls, expected = _run_ordinary_update(repo, monkeypatch, dirty=True)

    assert (repo / "remote.txt").read_bytes() == b"remote update\n"
    assert (repo / "tracked.py").read_bytes() == expected[0]
    assert (repo / "unrelated.bin").read_bytes() == expected[1]
    assert events == ["eol", "stash", "post-pull", "restore"]

    stash_apply = [call for call in git_calls if call[1:3] == ["stash", "apply"]]
    stash_drop = [call for call in git_calls if call[1:3] == ["stash", "drop"]]
    assert len(stash_apply) == 1
    assert len(stash_drop) == 1
    assert _git_bytes(repo, "stash", "list") == b""

    stash_push_index = next(i for i, call in enumerate(git_calls) if call[1:3] == ["stash", "push"])
    mutators = {call[1] for call in git_calls[:stash_push_index] if len(call) > 1}
    assert not (mutators & {"checkout", "reset", "clean"})

def test_ordinary_update_clean_checkout_still_completes_without_autostash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo, _origin = _make_local_update_repos(tmp_path)

    events, git_calls, _expected = _run_ordinary_update(repo, monkeypatch, dirty=False)

    assert (repo / "remote.txt").read_bytes() == b"remote update\n"
    assert events == ["eol", "stash", "post-pull"]
    assert not any(call[1:3] == ["stash", "push"] for call in git_calls)
    assert _git_bytes(repo, "stash", "list") == b""

def test_normalize_managed_eol_leaves_other_autocrlf_modes_and_bytes_untouched(
    tmp_path: Path,
) -> None:
    for mode in ("false", "input"):
        repo = _managed_repo(tmp_path / mode, {"a.py": b"x = 1\n"})
        _git(repo, "config", "core.autocrlf", mode)
        before = (repo / "a.py").read_bytes(), _autocrlf(repo)

        update_cmd._normalize_managed_eol(GIT_CMD, repo)

        assert ((repo / "a.py").read_bytes(), _autocrlf(repo)) == before
