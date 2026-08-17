"""Fail-closed release-candidate validation and gitlink coverage."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from unittest.mock import Mock

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


def _candidate_validator(git_cmd, candidate):
    resolved = update_cmd._git_resolve_commit(git_cmd, candidate, "HEAD")
    assert resolved is not None
    return resolved


def _release_repo(tmp_path: Path, *, target_gitlink: bool = False):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Candidate Test")
    _git(repo, "config", "user.email", "hermes-candidate@example.invalid")
    (repo / "README.txt").write_text("base\n", encoding="utf-8")
    base_sha = _commit(repo, "base")
    _git(repo, "tag", "v1.0.0")

    _git(repo, "branch", "hermes-release", base_sha)
    _git(repo, "switch", "hermes-release")
    (repo / "local.txt").write_text("local maintenance\n", encoding="utf-8")
    patches = repo / "local-patches"
    patches.mkdir()
    (patches / "0001-local-maintenance.patch").write_bytes(b"")
    (patches / ".release_base").write_text(
        json.dumps(
            {
                "tag": "v1.0.0",
                "base_sha": base_sha,
                "target_sha": base_sha,
            },
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    (patches / "README.md").write_text("metadata\n", encoding="utf-8")
    maintenance_sha = _commit(repo, "record release metadata")

    _git(repo, "switch", "main")
    (repo / "release.txt").write_text("release\n", encoding="utf-8")
    target_sha = _commit(repo, "release")
    if target_gitlink:
        _git(
            repo,
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{base_sha},vendor/linked-submodule",
        )
        target_sha = _git(repo, "commit", "-m", "release with gitlink").stdout
        target_sha = _git(repo, "rev-parse", "HEAD").stdout.decode().strip()
    _git(repo, "tag", "v2.0.0")
    _git(repo, "switch", "hermes-release")
    return repo, base_sha, maintenance_sha, target_sha


def test_critical_file_manifest_is_strict_and_includes_updater_modules():
    assert "hermes_cli/main.py" in update_cmd._UPDATE_CRITICAL_FILES
    assert "hermes_cli/update_cmd.py" in update_cmd._UPDATE_CRITICAL_FILES
    assert "hermes_cli.main" in update_cmd._UPDATE_CRITICAL_MODULES
    assert "hermes_cli.update_cmd" in update_cmd._UPDATE_CRITICAL_MODULES


@pytest.mark.parametrize("kind", ["missing", "directory", "symlink"])
def test_syntax_guard_rejects_missing_nonregular_or_symlink_file(tmp_path, monkeypatch, kind):
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_FILES", ("required.py",))
    required = tmp_path / "required.py"
    if kind == "directory":
        required.mkdir()
    elif kind == "symlink":
        outside = tmp_path / "outside.py"
        outside.write_text("VALUE = 1\n", encoding="utf-8")
        required.symlink_to(outside)

    ok, failing_path, error = update_cmd._validate_critical_files_syntax(tmp_path)

    assert ok is False
    assert failing_path == str(required)
    assert error
    assert kind in error.lower() or "regular" in error.lower()


def test_syntax_guard_rejects_regular_file_that_resolves_outside_candidate(tmp_path, monkeypatch):
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_FILES", ("required.py",))
    outside = tmp_path.parent / "candidate-validation-outside.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        (tmp_path / "required.py").symlink_to(outside)
        ok, failing_path, error = update_cmd._validate_critical_files_syntax(tmp_path)
    finally:
        outside.unlink(missing_ok=True)

    assert ok is False
    assert failing_path == str(tmp_path / "required.py")
    assert "symlink" in error.lower() or "outside" in error.lower()


def test_syntax_guard_rejects_syntax_error(tmp_path, monkeypatch):
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_FILES", ("required.py",))
    required = tmp_path / "required.py"
    required.write_text("def broken(:\n", encoding="utf-8")

    ok, failing_path, error = update_cmd._validate_critical_files_syntax(tmp_path)

    assert ok is False
    assert failing_path == str(required)
    assert "syntax" in error.lower()


def _write_consumer(root: Path, source: str = "VALUE = 1\n") -> None:
    (root / "consumer.py").write_text(source, encoding="utf-8")


def test_import_probe_executes_candidate_module_and_requires_origin(tmp_path, monkeypatch):
    _write_consumer(tmp_path)
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert (ok, module, error) == (True, None, None)


def test_import_probe_rejects_wrong_origin_module(tmp_path, monkeypatch):
    outside = tmp_path.parent / "wrong-origin-consumer.py"
    outside.write_text("VALUE = 1\n", encoding="utf-8")
    try:
        (tmp_path / "consumer.py").symlink_to(outside)
        monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))
        ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)
    finally:
        outside.unlink(missing_ok=True)

    assert ok is False
    assert module == "consumer"
    assert error and "candidate" in error.lower()


def test_import_probe_rejects_import_exception_and_traceback(tmp_path, monkeypatch):
    _write_consumer(tmp_path, "raise RuntimeError('candidate import exploded')\n")
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "consumer"
    assert error and "candidate import exploded" in error


@pytest.mark.parametrize("failure", [OSError("cannot launch"), subprocess.TimeoutExpired("python", 1)])
def test_import_probe_launch_failure_and_timeout_fail_closed(tmp_path, monkeypatch, failure):
    _write_consumer(tmp_path)
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_MODULES", ("consumer",))

    def fail_probe(*_args, **_kwargs):
        raise failure

    monkeypatch.setattr(update_cmd.subprocess, "run", fail_probe)
    ok, module, error = update_cmd._validate_critical_modules_import(tmp_path)

    assert ok is False
    assert module == "candidate import probe"
    assert error and ("launch" in error.lower() or "timed out" in error.lower())


def test_release_transaction_defaults_to_strict_validation(tmp_path):
    repo, _base_sha, maintenance_sha, target_sha = _release_repo(tmp_path)

    with pytest.raises(RuntimeError, match="Candidate validation failed"):
        update_cmd._upgrade_release_transaction(["git"], repo, "v2.0.0", target_sha)

    assert _git(repo, "rev-parse", "HEAD").stdout.decode().strip() == maintenance_sha
    assert _git(repo, "rev-parse", "refs/heads/hermes-release").stdout.decode().strip() == maintenance_sha
    assert _git(repo, "status", "--porcelain=v1").stdout == b""


def test_release_transaction_calls_explicit_test_validator_seam(tmp_path):
    repo, _base_sha, _maintenance_sha, target_sha = _release_repo(tmp_path)
    validator = Mock(side_effect=_candidate_validator)

    result = update_cmd._upgrade_release_transaction(
        ["git"], repo, "v2.0.0", target_sha, candidate_validator=validator
    )

    assert result.candidate_sha
    validator.assert_called_once()


def test_gitlink_target_is_rejected_before_payload_apply_or_promotion(tmp_path):
    repo, _base_sha, maintenance_sha, target_sha = _release_repo(
        tmp_path, target_gitlink=True
    )

    with pytest.raises(RuntimeError, match="gitlink|submodule"):
        update_cmd._upgrade_release_transaction(
            ["git"], repo, "v2.0.0", target_sha,
            candidate_validator=_candidate_validator,
        )

    assert _git(repo, "rev-parse", "HEAD").stdout.decode().strip() == maintenance_sha
    assert _git(repo, "rev-parse", "refs/heads/hermes-release").stdout.decode().strip() == maintenance_sha
    assert _git(repo, "status", "--porcelain=v1").stdout == b""


def test_incremental_merge_reuses_recorded_rerere_resolution(tmp_path):
    repo = tmp_path / "rerere-repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.name", "Hermes Rerere Test")
    _git(repo, "config", "user.email", "hermes-rerere@example.invalid")
    conflict = repo / "conflict.txt"
    conflict.write_text("base\n", encoding="utf-8")
    _commit(repo, "base")

    _git(repo, "switch", "-c", "maintenance")
    conflict.write_text("local\n", encoding="utf-8")
    maintenance_sha = _commit(repo, "local change")

    _git(repo, "switch", "main")
    conflict.write_text("upstream\n", encoding="utf-8")
    target_sha = _commit(repo, "upstream change")

    first = tmp_path / "candidate-first"
    _git(repo, "worktree", "add", "-b", "candidate-first", str(first), maintenance_sha)
    with pytest.raises(RuntimeError, match="conflict.txt"):
        update_cmd._merge_release_into_candidate(
            ["git"],
            first,
            maintenance_sha=maintenance_sha,
            target_sha=target_sha,
            release_tag="v2.0.0",
        )
    (first / "conflict.txt").write_text("local\nupstream\n", encoding="utf-8")
    _git(first, "add", "conflict.txt")
    _git(first, "-c", "rerere.enabled=true", "rerere")
    _git(repo, "worktree", "remove", "--force", str(first))
    _git(repo, "branch", "-D", "candidate-first")

    second = tmp_path / "candidate-second"
    _git(repo, "worktree", "add", "-b", "candidate-second", str(second), maintenance_sha)
    candidate_sha = update_cmd._merge_release_into_candidate(
        ["git"],
        second,
        maintenance_sha=maintenance_sha,
        target_sha=target_sha,
        release_tag="v2.0.0",
    )

    assert (second / "conflict.txt").read_text(encoding="utf-8") == "local\nupstream\n"
    parents = (
        _git(second, "rev-list", "--parents", "-n", "1", candidate_sha)
        .stdout.decode()
        .split()
    )
    assert parents == [candidate_sha, maintenance_sha, target_sha]
    assert _git(repo, "config", "--get", "rerere.enabled", check=False).returncode == 1


def test_payload_apply_reports_actual_unmerged_paths(monkeypatch, tmp_path):
    def fake_run(command, **_kwargs):
        if command[-2:] == ["ls-files", "--unmerged"]:
            return subprocess.CompletedProcess(
                command,
                0,
                stdout=(
                    "100644 aaa 1\tconflict-a.py\n"
                    "100644 bbb 2\tconflict-a.py\n"
                    "100644 ccc 3\tconflict-b.py\n"
                ),
                stderr="",
            )
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="Applied patch ... cleanly.\n",
            stderr="",
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError) as exc_info:
        update_cmd._apply_payload_to_candidate(
            ["git"], tmp_path, b"payload", label="v2.0.0"
        )

    message = str(exc_info.value)
    assert "conflict-a.py" in message
    assert "conflict-b.py" in message


def test_payload_apply_reports_apply_error_without_unmerged_index(
    monkeypatch, tmp_path
):
    def fake_run(command, **_kwargs):
        if command[-2:] == ["ls-files", "--unmerged"]:
            return subprocess.CompletedProcess(command, 0, stdout="", stderr="")
        return subprocess.CompletedProcess(
            command,
            1,
            stdout="",
            stderr="error: patch failed: hermes_cli/update_cmd.py:42\n",
        )

    monkeypatch.setattr(update_cmd.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="patch failed.*hermes_cli/update_cmd.py:42"):
        update_cmd._apply_payload_to_candidate(
            ["git"], tmp_path, b"payload", label="v2.0.0"
        )
