"""Tests for ``hermes upgrade`` release-tag update path."""

import subprocess
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from hermes_cli import main as hm


def _git_completed(cmd, stdout="", returncode=0, stderr=""):
    return subprocess.CompletedProcess(cmd, returncode, stdout=stdout, stderr=stderr)


def _test_candidate_validator(git_cmd, candidate):
    """Explicit seam for the non-Hermes-shaped release fixture."""
    from hermes_cli import update_cmd

    resolved = update_cmd._git_resolve_commit(git_cmd, candidate, "HEAD")
    assert resolved is not None
    return resolved


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


@pytest.mark.parametrize(
    ("transaction_error", "post_promotion_signal"),
    [
        (False, None),
        (True, None),
        (False, "called_process_error"),
        (False, "system_exit"),
        (False, "keyboard_interrupt"),
    ],
)
def test_cmd_upgrade_release_orchestration_uses_transaction_and_releases_lock(
    tmp_path, monkeypatch, transaction_error, post_promotion_signal
):
    """Release orchestration uses the immutable target and always releases its repo lock."""
    from hermes_cli import update_cmd

    repo = tmp_path / (
        f"transaction-error-{post_promotion_signal}"
        if transaction_error
        else f"transaction-success-{post_promotion_signal}"
    )
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
    lifecycle_events = []
    transaction_context = object()

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
            lifecycle_events.append("lock-release")

    class StopAfterSuccessfulTransaction(RuntimeError):
        pass

    def post_promotion_stop(_repo):
        if post_promotion_signal is None:
            raise StopAfterSuccessfulTransaction()
        if post_promotion_signal == "called_process_error":
            raise subprocess.CalledProcessError(1, ["post-promotion"])
        if post_promotion_signal == "system_exit":
            raise SystemExit(23)
        raise KeyboardInterrupt()

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
        return_value=SimpleNamespace(
            target_sha=target.target_sha, context=transaction_context
        ),
    )
    resolve_target = Mock(return_value=target)
    legacy_replay = Mock(
        side_effect=AssertionError("legacy release replay must not run")
    )

    def finalize_release(*args, **kwargs):
        assert args[2] is transaction_context
        assert lock_instances[0].release_calls == 0
        lifecycle_events.append("finalize")
        return True

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
        patcher.setattr(hm, "_clear_bytecode_cache", post_promotion_stop)
        patcher.setattr("hermes_cli.config.load_config", lambda: {})
        patcher.setattr(update_cmd, "RepositoryUpdateLock", FakeRepositoryUpdateLock)
        patcher.setattr(update_cmd, "_resolve_release_target", resolve_target)
        patcher.setattr(update_cmd, "_git_resolve_commit", lambda *args: maintenance_sha)
        patcher.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
        patcher.setattr(update_cmd, "_prepare_and_promote_release", transaction)
        patcher.setattr(update_cmd, "_finalize_release_upgrade", finalize_release)
        patcher.setattr(update_cmd, "_upgrade_release_with_local_patches", legacy_replay)
        patcher.setattr("subprocess.run", fake_run)

        if transaction_error or post_promotion_signal == "called_process_error":
            with pytest.raises(SystemExit) as exc_info:
                update_cmd._cmd_update_impl(args, gateway_mode=False)
            assert exc_info.value.code == 1
        elif post_promotion_signal == "system_exit":
            with pytest.raises(SystemExit) as exc_info:
                update_cmd._cmd_update_impl(args, gateway_mode=False)
            assert exc_info.value.code == 23
        elif post_promotion_signal == "keyboard_interrupt":
            with pytest.raises(KeyboardInterrupt):
                update_cmd._cmd_update_impl(args, gateway_mode=False)
        else:
            with pytest.raises(StopAfterSuccessfulTransaction):
                update_cmd._cmd_update_impl(args, gateway_mode=False)

    assert len(lock_instances) == 1
    assert lock_instances[0].repo_root == repo
    assert lock_instances[0].git_cmd == expected_git_cmd
    assert lock_instances[0].acquire_calls == 1
    assert lock_instances[0].release_calls == 1
    assert lifecycle_events == (
        ["lock-release"] if transaction_error else ["finalize", "lock-release"]
    )
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


def test_windows_release_post_promotion_failure_never_uses_zip_fallback(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import update_cmd

    repo = tmp_path / "release-post-promotion-error"
    repo.mkdir()
    (repo / ".git").mkdir()
    args = SimpleNamespace(
        release_tag="v1.2.3",
        yes=True,
        force=False,
        force_venv=False,
    )
    target = update_cmd.ReleaseTarget(
        tag=args.release_tag,
        target_sha="3" * 40,
        ref="refs/hermes-upgrade/tags/v1.2.3",
    )
    lock_instances = []
    lifecycle_events = []
    transaction_context = object()

    class FakeRepositoryUpdateLock:
        def __init__(self, repo_root, git_cmd):
            self.repo_root = repo_root
            self.git_cmd = git_cmd
            self.release_calls = 0
            lock_instances.append(self)

        def acquire(self):
            return self

        def release(self):
            self.release_calls += 1
            lifecycle_events.append("lock-release")

    def fake_run(cmd, **kwargs):
        if "--abbrev-ref" in cmd:
            return _git_completed(cmd, stdout="hermes-release\n")
        if "merge-base" in cmd:
            return _git_completed(cmd, returncode=1)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    transaction = Mock(
        return_value=SimpleNamespace(target_sha=target.target_sha, context=transaction_context)
    )
    zip_fallback = Mock(side_effect=AssertionError("release must not use ZIP fallback"))

    def fail_after_promotion(*_args, **_kwargs):
        raise subprocess.CalledProcessError(1, ["post-promotion-build"])

    def finalize_release(*args, **kwargs):
        assert args[2] is transaction_context
        assert lock_instances[0].release_calls == 0
        lifecycle_events.append("finalize")
        return True

    with monkeypatch.context() as patcher:
        patcher.setattr(update_cmd.sys, "platform", "win32")
        patcher.setattr(hm, "PROJECT_ROOT", repo)
        patcher.setattr(hm, "_is_windows", lambda: False)
        patcher.setattr(hm, "_run_pre_update_backup", lambda _args: None)
        patcher.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
        patcher.setattr(hm, "_resume_windows_gateways_after_update", lambda _state: None)
        patcher.setattr(
            hm,
            "_get_origin_url",
            lambda _git_cmd, _repo: "https://github.com/NousResearch/hermes-agent.git",
        )
        patcher.setattr(update_cmd, "_capture_head_sha", lambda *_args: "4" * 40)
        patcher.setattr(hm, "_clear_bytecode_cache", fail_after_promotion)
        patcher.setattr("hermes_cli.config.load_config", lambda: {})
        patcher.setattr(update_cmd, "RepositoryUpdateLock", FakeRepositoryUpdateLock)
        patcher.setattr(update_cmd, "_resolve_release_target", Mock(return_value=target))
        patcher.setattr(update_cmd, "_git_resolve_commit", lambda *args: "4" * 40)
        patcher.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
        patcher.setattr(update_cmd, "_prepare_and_promote_release", transaction)
        patcher.setattr(update_cmd, "_finalize_release_upgrade", finalize_release)
        patcher.setattr(update_cmd, "_update_via_zip", zip_fallback)
        patcher.setattr("subprocess.run", fake_run)

        with pytest.raises(SystemExit) as exc_info:
            update_cmd._cmd_update_impl(args, gateway_mode=False)

    assert exc_info.value.code == 1
    assert "post-promotion" in capsys.readouterr().out.lower()
    zip_fallback.assert_not_called()
    assert lifecycle_events == ["finalize", "lock-release"]
    assert lock_instances[0].release_calls == 1


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
            ["git"], repo, "v1.2.3", target_sha,
            candidate_validator=_test_candidate_validator,
        )

    assert result.target_sha == target_sha
    assert (repo / "upstream.txt").read_text(encoding="utf-8") == "release payload\n"
    assert (repo / "local.txt").read_text(encoding="utf-8") == "local maintenance\n"
    assert git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip() == "hermes-release"


def _finalization_context(tmp_path):
    return SimpleNamespace(journal_path=tmp_path / "release-finalization.json")


def test_release_finalization_false_raises_dedicated_failure_with_journal_path(
    tmp_path, monkeypatch
):
    from hermes_cli import update_cmd

    context = _finalization_context(tmp_path)
    finalizer = Mock(return_value=False)
    monkeypatch.setattr(update_cmd, "_finalize_release_upgrade", finalizer)

    with pytest.raises(
        update_cmd.ReleaseFinalizationIncompleteError,
        match=f"{context.journal_path}",
    ):
        update_cmd._finalize_release_upgrade_for_orchestration(
            ["git"], tmp_path, context
        )

    finalizer.assert_called_once_with(
        ["git"], tmp_path, context, input_fn=None
    )


@pytest.mark.parametrize(
    "primary_factory",
    [
        lambda: RuntimeError("primary transaction failure"),
        KeyboardInterrupt,
    ],
)
def test_release_finalization_false_preserves_active_primary_exception(
    tmp_path, monkeypatch, primary_factory
):
    from hermes_cli import update_cmd
    import sys

    context = _finalization_context(tmp_path)
    monkeypatch.setattr(update_cmd, "_finalize_release_upgrade", Mock(return_value=False))
    primary = primary_factory()

    with pytest.raises(type(primary)) as exc_info:
        try:
            raise primary
        except BaseException:
            update_cmd._finalize_release_upgrade_for_orchestration(
                ["git"], tmp_path, context, primary_exc_info=sys.exc_info()
            )
            raise

    assert exc_info.value is primary
    assert str(exc_info.value) == str(primary)


@pytest.mark.parametrize(
    "finalizer_control",
    [KeyboardInterrupt(), SystemExit(17)],
)
def test_release_finalizer_process_control_preserves_primary_and_lock_order(
    tmp_path, monkeypatch, finalizer_control
):
    from hermes_cli import update_cmd
    import sys

    context = _finalization_context(tmp_path)
    monkeypatch.setattr(
        update_cmd,
        "_finalize_release_upgrade",
        Mock(side_effect=finalizer_control),
    )
    primary = RuntimeError("primary remains authoritative")

    with pytest.raises(RuntimeError) as exc_info:
        try:
            raise primary
        except BaseException:
            update_cmd._finalize_release_upgrade_for_orchestration(
                ["git"], tmp_path, context, primary_exc_info=sys.exc_info()
            )
            raise

    assert exc_info.value is primary


@pytest.mark.parametrize(
    "finalizer_control",
    [KeyboardInterrupt(), SystemExit(23)],
)
def test_release_finalizer_process_control_propagates_without_primary(
    tmp_path, monkeypatch, finalizer_control
):
    from hermes_cli import update_cmd

    context = _finalization_context(tmp_path)
    monkeypatch.setattr(
        update_cmd,
        "_finalize_release_upgrade",
        Mock(side_effect=finalizer_control),
    )

    with pytest.raises(type(finalizer_control)) as exc_info:
        update_cmd._finalize_release_upgrade_for_orchestration(
            ["git"], tmp_path, context
        )

    assert exc_info.value is finalizer_control


def test_release_finalization_true_remains_successful(tmp_path, monkeypatch):
    from hermes_cli import update_cmd

    context = _finalization_context(tmp_path)
    finalizer = Mock(return_value=True)
    monkeypatch.setattr(update_cmd, "_finalize_release_upgrade", finalizer)

    assert (
        update_cmd._finalize_release_upgrade_for_orchestration(
            ["git"], tmp_path, context
        )
        is True
    )
    finalizer.assert_called_once()


def _run_mocked_top_level_release_update(
    tmp_path,
    monkeypatch,
    *,
    finalizer_result=True,
    post_success_exception=None,
    state_sink=None,
):
    """Run the real update owner with every unrelated update operation inert."""
    from hermes_cli import backup, config, gateway, model_catalog, profiles, update_cmd
    from hermes_cli import managed_uv
    from tools import skills_sync

    repo = tmp_path / "release-owner-repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    args = SimpleNamespace(
        release_tag="v1.2.3",
        yes=True,
        force=False,
        force_venv=False,
    )
    target = update_cmd.ReleaseTarget(
        tag=args.release_tag,
        target_sha="3" * 40,
        ref="refs/hermes-upgrade/tags/v1.2.3",
    )
    maintenance_sha = "4" * 40
    context = SimpleNamespace(journal_path=tmp_path / "release-finalization.json")
    events = []
    lock_instances = []

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
            events.append("lock-release")

    def fake_run(cmd, **kwargs):
        if "--abbrev-ref" in cmd:
            return _git_completed(cmd, stdout="hermes-release\n")
        if "merge-base" in cmd:
            return _git_completed(cmd, returncode=1)
        raise AssertionError(f"unexpected subprocess call: {cmd!r}")

    transaction = Mock(
        return_value=SimpleNamespace(target_sha=target.target_sha, context=context)
    )
    def finalize_release(*args, **kwargs):
        events.append("finalize")
        return finalizer_result

    finalizer = Mock(side_effect=finalize_release)
    clear_bytecode = Mock(return_value=0)
    if post_success_exception is not None:
        clear_bytecode.side_effect = post_success_exception

    monkeypatch.setattr(hm, "PROJECT_ROOT", repo)
    monkeypatch.setattr(hm, "_is_windows", lambda: False)
    monkeypatch.setattr(hm, "_run_pre_update_backup", lambda _args: None)
    monkeypatch.setattr(hm, "_pause_windows_gateways_for_update", lambda: None)
    monkeypatch.setattr(hm, "_resume_windows_gateways_after_update", lambda _state: None)
    monkeypatch.setattr(
        hm,
        "_get_origin_url",
        lambda _git_cmd, _repo: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(hm, "_clear_bytecode_cache", clear_bytecode)
    monkeypatch.setattr(hm, "_record_bytecode_fingerprint", lambda: None)
    monkeypatch.setattr(hm, "_reload_updated_runtime_modules", lambda: None)
    monkeypatch.setattr(hm, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hm, "_clear_lazy_refresh_incomplete_marker", lambda: None)
    monkeypatch.setattr(hm, "_upgrade_pip_before_lazy_refresh", lambda *args, **kwargs: None)
    monkeypatch.setattr(hm, "_refresh_active_lazy_features", lambda *args, **kwargs: True)
    monkeypatch.setattr(hm, "_refresh_active_memory_provider_dependencies", lambda: None)
    monkeypatch.setattr(hm, "_is_termux_env", lambda *_args, **_kwargs: False)
    monkeypatch.setattr(hm, "_install_python_dependencies_with_optional_fallback", lambda *args, **kwargs: None)
    monkeypatch.setattr(hm, "_build_web_ui", lambda *args, **kwargs: True)
    monkeypatch.setattr(hm, "_desktop_packaged_executable", lambda *_args: None)
    monkeypatch.setattr(hm, "_desktop_dist_exists", lambda *_args: False)
    monkeypatch.setattr(hm, "_resolve_node_runtime_npm", lambda: None)
    monkeypatch.setattr(managed_uv, "ensure_uv", lambda: "/fake/uv")
    monkeypatch.setattr(managed_uv, "update_managed_uv", lambda: None)

    monkeypatch.setattr(update_cmd, "_is_fork", lambda _origin: False)
    monkeypatch.setattr(update_cmd, "_resolve_release_target", Mock(return_value=target))
    monkeypatch.setattr(update_cmd, "_git_resolve_commit", lambda *_args: maintenance_sha)
    monkeypatch.setattr(update_cmd, "_capture_head_sha", lambda *_args: maintenance_sha)
    monkeypatch.setattr(update_cmd, "_prepare_and_promote_release", transaction)
    monkeypatch.setattr(update_cmd, "_invalidate_update_cache", lambda: None)
    monkeypatch.setattr(update_cmd, "_finalize_release_upgrade", finalizer)
    monkeypatch.setattr(update_cmd, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_write_lazy_refresh_incomplete_marker", lambda: None)
    monkeypatch.setattr(update_cmd, "_upgrade_pip_before_lazy_refresh", lambda *args, **kwargs: None)
    monkeypatch.setattr(update_cmd, "_refresh_active_lazy_features", lambda *args, **kwargs: True)
    monkeypatch.setattr(update_cmd, "_refresh_active_memory_provider_dependencies", lambda: None)
    monkeypatch.setattr(update_cmd, "_validate_critical_modules_import", lambda *_args: (True, None, None))
    monkeypatch.setattr(update_cmd, "_update_node_dependencies", lambda: [])
    monkeypatch.setattr(update_cmd, "_print_fts_optimize_available_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_print_curator_first_run_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_print_curator_recent_run_notice", lambda: None)
    monkeypatch.setattr(update_cmd, "_ensure_fhs_path_guard", lambda: None)
    monkeypatch.setattr(update_cmd, "_ensure_acp_launcher", lambda: None)
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *_args: None)
    monkeypatch.setattr(update_cmd.shutil, "which", lambda _name: None)
    monkeypatch.setattr(update_cmd, "_warn_incomplete_gateway_fleet_restart", lambda *_args: None)
    monkeypatch.setattr(update_cmd, "_ensure_uv_for_termux", lambda *_args: None)

    monkeypatch.setattr(config, "load_config", lambda: {})
    monkeypatch.setattr(config, "get_missing_env_vars", lambda **_kwargs: [])
    monkeypatch.setattr(config, "get_missing_config_fields", lambda: [])
    monkeypatch.setattr(config, "check_config_version", lambda: (1, 1))
    monkeypatch.setattr(config, "migrate_config", lambda **_kwargs: None)
    monkeypatch.setattr(backup, "restore_cron_jobs_if_emptied", lambda *_args: None)
    monkeypatch.setattr(model_catalog, "seed_cache_from_checkout", lambda *_args: False)
    monkeypatch.setattr(skills_sync, "sync_skills", lambda **_kwargs: {
        "copied": [],
        "updated": [],
        "user_modified": [],
        "cleaned": [],
        "relocated": [],
    })
    monkeypatch.setattr(profiles, "list_profiles", lambda: [])
    monkeypatch.setattr(profiles, "backfill_profile_envs", lambda **_kwargs: [])
    monkeypatch.setattr(gateway, "supports_systemd_services", lambda: False)
    monkeypatch.setattr(gateway, "is_macos", lambda: False)
    monkeypatch.setattr(gateway, "_get_service_pids", lambda: set())
    monkeypatch.setattr(gateway, "find_gateway_pids", lambda **_kwargs: [])
    monkeypatch.setattr(gateway, "find_profile_gateway_processes", lambda **_kwargs: [])
    monkeypatch.setattr("subprocess.run", fake_run)
    monkeypatch.setattr(update_cmd, "RepositoryUpdateLock", FakeRepositoryUpdateLock)

    state = {
        "repo": repo,
        "context": context,
        "events": events,
        "finalizer": finalizer,
        "lock": None,
        "clear_bytecode": clear_bytecode,
    }
    if state_sink is not None:
        state_sink.update(state)
    try:
        update_cmd._cmd_update_impl(args, gateway_mode=False)
    finally:
        assert len(lock_instances) == 1
        state["lock"] = lock_instances[0]
        if state_sink is not None:
            state_sink["lock"] = lock_instances[0]
    return state


def test_cmd_update_impl_finalization_false_fails_without_release_banners(
    tmp_path, monkeypatch, capsys
):
    from hermes_cli import update_cmd

    state = {}
    with pytest.raises(update_cmd.ReleaseFinalizationIncompleteError) as exc_info:
        _run_mocked_top_level_release_update(
            tmp_path, monkeypatch, finalizer_result=False, state_sink=state
        )

    assert str(exc_info.value).endswith(str(state["context"].journal_path) + ".")
    output = capsys.readouterr().out
    assert "✓ Code updated!" not in output
    assert "✓ Update complete!" not in output
    assert state["finalizer"].call_count == 1
    assert state["events"] == ["finalize", "lock-release"]
    assert state["lock"].release_calls == 1


def test_cmd_update_impl_finalization_true_emits_each_release_banner_once(
    tmp_path, monkeypatch, capsys
):
    state = _run_mocked_top_level_release_update(
        tmp_path, monkeypatch, finalizer_result=True
    )

    output = capsys.readouterr().out
    assert output.count("✓ Code updated!") == 1
    assert output.count("✓ Update complete!") == 1
    assert state["finalizer"].call_count == 1
    assert state["events"] == ["finalize", "lock-release"]
    assert state["lock"].release_calls == 1


@pytest.mark.parametrize(
    "primary",
    [RuntimeError("top-level update failed"), KeyboardInterrupt()],
)
def test_cmd_update_impl_primary_exception_wins_finalization_false(
    tmp_path, monkeypatch, capsys, primary
):
    state = {}
    with pytest.raises(type(primary)) as exc_info:
        _run_mocked_top_level_release_update(
            tmp_path,
            monkeypatch,
            finalizer_result=False,
            post_success_exception=primary,
            state_sink=state,
        )

    assert exc_info.value is primary
    output = capsys.readouterr().out
    assert "✓ Code updated!" not in output
    assert "✓ Update complete!" not in output
    assert state["finalizer"].call_count == 1
    assert state["events"] == ["finalize", "lock-release"]
    assert state["lock"].release_calls == 1
