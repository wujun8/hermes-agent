"""Tests for ``hermes upgrade`` release-tag update path."""

import subprocess
from types import SimpleNamespace
from unittest.mock import patch

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
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        joined = " ".join(str(part) for part in cmd)
        if "fetch origin --tags" in joined:
            return _git_completed(cmd)
        if "rev-parse --verify v1.2.3^{commit}" in joined:
            return _git_completed(cmd, stdout="newsha\n")
        if "merge-base --is-ancestor v1.2.3 HEAD" in joined:
            return _git_completed(cmd, returncode=1)
        return _git_completed(cmd)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch.object(hm, "_fetch_latest_release_tag", return_value="v1.2.3"), \
         patch("subprocess.run", side_effect=fake_run):
        hm._cmd_upgrade_check()

    out = capsys.readouterr().out
    assert "Latest Release: v1.2.3" in out
    assert "Release upgrade available: v1.2.3" in out
    assert any(cmd[:4] == ["git", "fetch", "origin", "--tags"] for cmd in calls)


def test_cmd_upgrade_check_reports_already_on_latest_release(capsys):
    def fake_run(cmd, **kwargs):
        joined = " ".join(str(part) for part in cmd)
        if "rev-parse --verify v1.2.3^{commit}" in joined:
            return _git_completed(cmd, stdout="samesha\n")
        if "merge-base --is-ancestor v1.2.3 HEAD" in joined:
            return _git_completed(cmd, returncode=0)
        return _git_completed(cmd)

    with patch("hermes_cli.config.detect_install_method", return_value="git"), \
         patch.object(hm, "_fetch_latest_release_tag", return_value="v1.2.3"), \
         patch("subprocess.run", side_effect=fake_run):
        hm._cmd_upgrade_check()

    assert "Already includes the latest Release (v1.2.3)" in capsys.readouterr().out


def test_cmd_upgrade_merges_release_into_local_maintenance_branch(capsys):
    """Release upgrades keep local patches on a branch instead of detached HEAD."""
    args = SimpleNamespace(release_tag="v1.2.3", gateway=False, yes=True)
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        joined = " ".join(str(part) for part in cmd)
        if "rev-parse --abbrev-ref HEAD" in joined:
            return _git_completed(cmd, stdout="main\n")
        if "rev-parse --verify v1.2.3^{commit}" in joined:
            return _git_completed(cmd, stdout="release-sha\n")
        if "rev-parse --verify hermes-release" in joined:
            return _git_completed(cmd, returncode=128, stderr="missing branch\n")
        if "merge-base --is-ancestor v1.2.3 HEAD" in joined:
            return _git_completed(cmd, returncode=1)
        if joined.endswith("rev-parse HEAD"):
            return _git_completed(cmd, stdout="local-sha\n")
        if "status --porcelain" in joined or "ls-files --unmerged" in joined:
            return _git_completed(cmd)
        return _git_completed(cmd)

    with patch("shutil.which", return_value=None), \
         patch.object(hm, "_validate_critical_files_syntax", return_value=(True, None, None)), \
         patch.object(hm, "_clear_bytecode_cache", return_value=0), \
         patch.object(hm, "_install_python_dependencies_with_optional_fallback"), \
         patch.object(hm, "_refresh_active_lazy_features"), \
         patch.object(hm, "_update_node_dependencies"), \
         patch.object(hm, "_build_web_ui"), \
         patch.object(hm, "_print_curator_first_run_notice"), \
         patch.object(hm, "_print_curator_recent_run_notice"), \
         patch.object(hm, "_ensure_fhs_path_guard"), \
         patch.object(hm, "_get_origin_url", return_value="https://github.com/NousResearch/hermes-agent.git"), \
         patch("hermes_cli.config.get_missing_env_vars", return_value=[]), \
         patch("hermes_cli.config.get_missing_config_fields", return_value=[]), \
         patch("hermes_cli.config.check_config_version", return_value=(1, 1)), \
         patch("subprocess.run", side_effect=fake_run):
        hm._cmd_update_impl(args, gateway_mode=False)

    commands = [" ".join(str(part) for part in cmd) for cmd in calls]
    assert "git checkout -B hermes-release" in commands
    assert "git merge --no-edit v1.2.3" in commands
    assert "git checkout --detach v1.2.3" not in commands
    assert "Target Release: v1.2.3" in capsys.readouterr().out
