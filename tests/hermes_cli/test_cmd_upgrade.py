"""Tests for ``hermes upgrade`` release-tag update path."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hermes_cli import main as hm


def _git_completed(cmd, stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def test_cmd_upgrade_resolves_latest_release_and_delegates_to_update_impl():
    args = SimpleNamespace(check=False, gateway=False)

    with patch("hermes_cli.config.is_managed", return_value=False), \
         patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch.object(hm, "_fetch_latest_release_tag", return_value="v1.2.3"), \
         patch.object(hm, "_install_hangup_protection", return_value={}) as protect, \
         patch.object(hm, "_finalize_update_output") as finalize, \
         patch.object(hm, "_cmd_update_impl") as update_impl:
        hm.cmd_upgrade(args)

    assert args.release_tag == "v1.2.3"
    assert args.branch is None
    protect.assert_called_once_with(gateway_mode=False)
    update_impl.assert_called_once_with(args, gateway_mode=False)
    finalize.assert_called_once()


def test_cmd_upgrade_check_compares_head_to_latest_release_tag(capsys):
    from hermes_cli import update_cmd

    calls = []
    target = update_cmd.ReleaseTarget(
        tag="v1.2.3",
        target_sha="1" * 40,
        ref="refs/hermes-upgrade/tags/v1.2.3",
    )

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[-4:] == ["merge-base", "--is-ancestor", target.target_sha, "HEAD"]:
            return _git_completed(cmd, returncode=1)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch.object(hm, "_fetch_latest_release_tag", return_value="v1.2.3"), \
         patch.object(update_cmd, "_resolve_release_target", return_value=target) as resolve_target, \
         patch("subprocess.run", side_effect=fake_run):
        hm._cmd_upgrade_check()

    out = capsys.readouterr().out
    assert "Latest Release: v1.2.3" in out
    assert "Release upgrade available: v1.2.3" in out
    resolve_target.assert_called_once_with(["git"], hm.PROJECT_ROOT, "v1.2.3")
    assert calls == [["git", "merge-base", "--is-ancestor", target.target_sha, "HEAD"]]


def test_cmd_upgrade_check_reports_already_on_latest_release(capsys):
    from hermes_cli import update_cmd

    target = update_cmd.ReleaseTarget(
        tag="v1.2.3",
        target_sha="2" * 40,
        ref="refs/hermes-upgrade/tags/v1.2.3",
    )
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        if cmd[-4:] == ["merge-base", "--is-ancestor", target.target_sha, "HEAD"]:
            return _git_completed(cmd, returncode=0)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch.object(hm, "_fetch_latest_release_tag", return_value="v1.2.3"), \
         patch.object(update_cmd, "_resolve_release_target", return_value=target) as resolve_target, \
         patch("subprocess.run", side_effect=fake_run):
        hm._cmd_upgrade_check()

    resolve_target.assert_called_once_with(["git"], hm.PROJECT_ROOT, "v1.2.3")
    assert calls == [["git", "merge-base", "--is-ancestor", target.target_sha, "HEAD"]]
    assert "Already includes the latest Release (v1.2.3)" in capsys.readouterr().out


@pytest.mark.parametrize("transaction_error", [False, True])
def test_cmd_upgrade_release_orchestration_uses_transaction_and_releases_lock(
    tmp_path, monkeypatch, transaction_error
):
    """Release orchestration uses the immutable target and always releases its repo lock."""
    from hermes_cli import update_cmd

    repo = tmp_path / ("transaction-error" if transaction_error else "transaction-success")
    repo.mkdir()
    (repo / ".git").mkdir()
    args = SimpleNamespace(
        release_tag="v1.2.3",
        yes=True,
        force=False,
        force_venv=False,
    )
    expected_git_cmd = ["git"]
    target = update_cmd.ReleaseTarget(
        tag=args.release_tag,
        target_sha="3" * 40,
        ref="refs/hermes-upgrade/tags/v1.2.3",
    )
    maintenance_sha = "4" * 40
    lock_instances = []
    subprocess_calls = []

    class FakeRepositoryUpdateLock:
        def __init__(self, repo_root, git_cmd):
            self.repo_root = repo_root
            self.git_cmd = git_cmd
            self.acquire_calls = 0
            self.release_calls = 0
            lock_instances.append(self)

        def acquire(self):
            self.acquire_calls += 1
            return self

        def release(self):
            self.release_calls += 1

    class StopAfterSuccessfulTransaction(RuntimeError):
        pass

    def fake_run(cmd, **kwargs):
        subprocess_calls.append(cmd)
        if "--abbrev-ref" in cmd:
            return _git_completed(cmd, stdout="hermes-release\n")
        if "merge-base" in cmd:
            assert target.target_sha in cmd
            return _git_completed(cmd, returncode=1)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    transaction = Mock(
        side_effect=(
            RuntimeError("transaction failed")
            if transaction_error
            else None
        ),
        return_value=SimpleNamespace(target_sha=target.target_sha),
    )
    resolve_target = Mock(return_value=target)
    legacy_replay = Mock(
        side_effect=AssertionError("legacy release replay must not run")
    )

    with monkeypatch.context() as patcher:
        patcher.setattr(hm, "PROJECT_ROOT", repo)
        patcher.setattr(hm, "_is_windows", lambda: False)
        patcher.setattr(hm, "_run_pre_update_backup", lambda _args: None)
        patcher.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
        patcher.setattr(
            hm,
            "_get_origin_url",
            lambda _git_cmd, _repo: "https://github.com/NousResearch/hermes-agent.git",
        )
        patcher.setattr(
            update_cmd, "_capture_head_sha", lambda _git_cmd, _repo: maintenance_sha
        )
        patcher.setattr(
            hm,
            "_clear_bytecode_cache",
            lambda _repo: (_ for _ in ()).throw(StopAfterSuccessfulTransaction()),
        )
        patcher.setattr("hermes_cli.config.load_config", lambda: {})
        patcher.setattr(update_cmd, "RepositoryUpdateLock", FakeRepositoryUpdateLock)
        patcher.setattr(update_cmd, "_resolve_release_target", resolve_target)
        patcher.setattr(update_cmd, "_git_resolve_commit", lambda *args: maintenance_sha)
        patcher.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
        patcher.setattr(update_cmd, "_prepare_and_promote_release", transaction)
        patcher.setattr(update_cmd, "_upgrade_release_with_local_patches", legacy_replay)
        patcher.setattr("subprocess.run", fake_run)

        if transaction_error:
            with pytest.raises(SystemExit) as exc_info:
                update_cmd._cmd_update_impl(args, gateway_mode=False)
            assert exc_info.value.code == 1
        else:
            with pytest.raises(StopAfterSuccessfulTransaction):
                update_cmd._cmd_update_impl(args, gateway_mode=False)

    assert len(lock_instances) == 1
    assert lock_instances[0].repo_root == repo
    assert lock_instances[0].git_cmd == expected_git_cmd
    assert lock_instances[0].acquire_calls == 1
    assert lock_instances[0].release_calls == 1
    resolve_target.assert_called_once_with(expected_git_cmd, repo, args.release_tag)
    transaction.assert_called_once_with(
        expected_git_cmd,
        repo,
        args.release_tag,
        target.target_sha,
        input_fn=None,
    )
    legacy_replay.assert_not_called()
    assert subprocess_calls == [
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        [
            "git",
            "merge-base",
            "--is-ancestor",
            target.target_sha,
            "refs/heads/hermes-release",
        ],
    ]
    assert list(repo.iterdir()) == [repo / ".git"]


def test_cmd_upgrade_replays_release_through_isolated_transaction(tmp_path):
    """The release seam promotes a real candidate without touching this checkout."""
    from hermes_cli import update_cmd

    repo = tmp_path / "repo"
    repo.mkdir()

    def git(*args):
        return subprocess.run(
            ["git", *args],
            cwd=repo,
            capture_output=True,
            text=True,
            check=True,
        )

    git("init", "-b", "main")
    git("config", "user.email", "tests@example.invalid")
    git("config", "user.name", "Hermes Tests")
    (repo / "README.md").write_text("base\n", encoding="utf-8")
    git("add", "README.md")
    git("commit", "-m", "base")
    base_sha = git("rev-parse", "HEAD").stdout.strip()
    git("tag", "v1.0.0")

    git("branch", "hermes-release", base_sha)
    git("switch", "hermes-release")
    (repo / "local.txt").write_text("local maintenance\n", encoding="utf-8")
    patches = repo / "local-patches"
    patches.mkdir()
    (patches / ".release_base").write_text(
        '{"base_sha": "' + base_sha + '", "tag": "v1.0.0"}\n',
        encoding="utf-8",
    )
    git("add", "local.txt", "local-patches/.release_base")
    git("commit", "-m", "local maintenance")

    git("switch", "main")
    (repo / "upstream.txt").write_text("release payload\n", encoding="utf-8")
    git("add", "upstream.txt")
    git("commit", "-m", "release")
    git("tag", "v1.2.3")
    target_sha = git("rev-parse", "v1.2.3").stdout.strip()
    git("switch", "hermes-release")

    with patch.object(update_cmd, "_upgrade_release_with_local_patches", side_effect=AssertionError("legacy release replay must not run")):
        result = update_cmd._prepare_and_promote_release(
            ["git"], repo, "v1.2.3", target_sha
        )

    assert result.target_sha == target_sha
    assert (repo / "upstream.txt").read_text(encoding="utf-8") == "release payload\n"
    assert (repo / "local.txt").read_text(encoding="utf-8") == "local maintenance\n"
    assert git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "hermes-release"
