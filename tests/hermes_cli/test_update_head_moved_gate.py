"""Tests for the post-pull HEAD-movement gate in ``hermes update``.

Issue #79678: a detached/pinned checkout can report "N new commit(s)"
against origin, run the ff-only merge successfully, and still sit on the
old commit afterward (the branch-switch step re-detaches to the raw SHA).
Before this guard ``hermes update`` printed "✓ Code updated!" and
reinstalled deps + rebuilt the desktop app against the stale tree — no
error, no warning. The gate compares the pre-pull and post-pull HEAD SHA
and fails loudly when the update was a no-op.
"""

from types import SimpleNamespace

import pytest

from hermes_cli import main as hermes_main
import hermes_cli.main_web_build as main_web_build
import hermes_cli.main_install_repair as main_install_repair
from hermes_cli import update_cmd

pytestmark = pytest.mark.usefixtures("isolated_source_completion")

# The current updater verifies full Git object names before moving the checkout.
PRE_SHA, POST_SHA = "abc123" + "0" * 34, "def456" + "0" * 34


def _make_head_pinned_side_effect(sha=PRE_SHA, target_sha=POST_SHA):
    """Simulate a detached checkout pinned to ``sha``: HEAD never moves."""

    def side_effect(cmd, **kwargs):
        joined = " ".join(str(c) for c in cmd)

        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="HEAD\n", stderr="")

        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

        if "rev-parse" in cmd:
            ref = str(cmd[-1]).removesuffix("^{commit}")
            resolved = {
                "HEAD": sha, "main": sha, "refs/heads/main": sha,
                "origin/main": target_sha, "refs/remotes/origin/main": target_sha,
                sha: sha, target_sha: target_sha,
            }
            if ref in resolved:
                return SimpleNamespace(returncode=0, stdout=f"{resolved[ref]}\n", stderr="")

        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect

def _patch_update_deps(monkeypatch, tmp_path, run_side_effect):
    """Patch the hermes_cli.main helpers ``_cmd_update_impl`` touches.

    ``_m()`` in update_cmd.py lazily returns hermes_cli.main, so patching
    attributes on that module is the canonical test surface (matches
    tests/hermes_cli/test_cmd_update.py).
    """
    monkeypatch.setattr(hermes_main.subprocess, "run", run_side_effect)
    # Windows custody and macOS fetch watching can bypass subprocess.run.
    # Patch the common Git entry point and its two module-level imports.
    from hermes_cli import gitlock, update_custody, update_cmd_maint

    def run_git(git_cmd, args, **kwargs):
        return run_side_effect([*git_cmd, *args], **kwargs)

    for module in (update_custody, gitlock, update_cmd._commit):
        monkeypatch.setattr(module, "run_git", run_git)
    monkeypatch.setattr(hermes_main, "PROJECT_ROOT", tmp_path)
    (tmp_path / ".git").mkdir()  # pass the "is a git repo" gate
    monkeypatch.setattr(
        hermes_main, "_resolve_update_branch", lambda args: "main"
    )
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: False)
    monkeypatch.setattr(main_install_repair, "_is_windows", lambda: False)
    monkeypatch.setattr(
        hermes_main, "_get_origin_url",
        lambda *a, **k: "https://github.com/NousResearch/hermes-agent.git",
    )
    monkeypatch.setattr(update_cmd, "_is_fork", lambda *a, **k: False)
    monkeypatch.setattr(
        hermes_main, "_stash_local_changes_if_needed", lambda *a, **k: None
    )
    monkeypatch.setattr(hermes_main, "_clear_bytecode_cache", lambda *a, **k: 0)
    monkeypatch.setattr(
        hermes_main, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        main_web_build, "_record_bytecode_fingerprint", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_run_pre_update_backup", lambda *a, **k: None
    )
    monkeypatch.setattr(
        hermes_main, "_pause_windows_gateways_for_update", lambda: None
    )
    monkeypatch.setattr(
        hermes_main, "_resume_windows_gateways_after_update", lambda *a, **k: None
    )
    # The shared isolated_source_completion fixture bypasses PM preparation and
    # product builds. Keep its in-process tail away from real maintenance too.
    monkeypatch.setattr(
        update_cmd_maint, "_run_post_update_maintenance", lambda **kwargs: True,
    )
    monkeypatch.setattr(update_cmd._posix_pause, "arm_pause", lambda **kwargs: None)
    monkeypatch.setattr(hermes_main, "_write_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(hermes_main, "_clear_update_incomplete_marker", lambda: None)
    monkeypatch.setattr(main_install_repair, "_clear_update_incomplete_marker", lambda: None)
    # Gateway restart path (called after a successful update).
    monkeypatch.setattr(update_cmd, "_finish_dashboard_update_cleanup", lambda *a, **k: None)
    # Keep the (now surfaced — #78574) gateway auto-restart phase away from
    # this machine's real gateways: discovery returns nothing, systemd is
    # unsupported, so the phase is a clean no-op for both snapshots.
    import hermes_cli.gateway as hermes_gateway

    monkeypatch.setattr(
        hermes_gateway, "find_gateway_pids", lambda all_profiles=False: []
    )
    monkeypatch.setattr(
        hermes_gateway, "supports_systemd_services", lambda: False
    )
    monkeypatch.setattr(
        hermes_gateway, "find_profile_gateway_processes", lambda *a, **k: []
    )

def test_update_fails_loudly_when_head_pinned(monkeypatch, tmp_path, capsys):
    """A detached/pinned HEAD that never moves must fail loudly, not print
    '✓ Code updated!' against the stale tree."""
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)
    _patch_update_deps(monkeypatch, tmp_path, _make_head_pinned_side_effect())

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(args)

    assert exc_info.value.code == 1
    out = capsys.readouterr().out
    assert "Code did not move" in out
    assert "✓ Code updated!" not in out


def _make_head_moved_side_effect(pre_sha=PRE_SHA, post_sha=POST_SHA):
    """Simulate git commands where HEAD advances from pre_sha to post_sha."""
    advanced = False

    def side_effect(cmd, **kwargs):
        nonlocal advanced
        joined = " ".join(str(c) for c in cmd)

        if "merge" in cmd and "--ff-only" in cmd:
            advanced = True

        # git rev-parse --abbrev-ref HEAD  (get current branch)
        if "rev-parse" in joined and "--abbrev-ref" in joined:
            return SimpleNamespace(returncode=0, stdout="main\n", stderr="")

        # git rev-list HEAD..origin/main --count  (behind count)
        if "rev-list" in joined:
            return SimpleNamespace(returncode=0, stdout="3\n", stderr="")

        # Local refs advance only after the merge; resolving the fetched target
        # (including origin/main^{commit}) must not move the simulated HEAD.
        if "rev-parse" in cmd:
            ref = str(cmd[-1]).removesuffix("^{commit}")
            head_sha = post_sha if advanced else pre_sha
            resolved = {
                "HEAD": head_sha, "main": head_sha, "refs/heads/main": head_sha,
                "origin/main": post_sha, "refs/remotes/origin/main": post_sha,
                pre_sha: pre_sha, post_sha: post_sha,
            }
            if ref in resolved:
                return SimpleNamespace(returncode=0, stdout=f"{resolved[ref]}\n", stderr="")

        # Everything else (merge, checkout, etc.) succeeds quietly.
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    return side_effect

def test_update_success_when_head_moves(monkeypatch, tmp_path, capsys):
    """When the pull advances HEAD, the update proceeds normally."""
    args = SimpleNamespace(branch=None, yes=False, force=False, force_venv=False)
    _patch_update_deps(monkeypatch, tmp_path, _make_head_moved_side_effect())

    hermes_main.cmd_update(args)  # completes normally (no SystemExit)

    out = capsys.readouterr().out
    assert "✓ Code updated!" in out
    assert "Code did not move" not in out

def test_windows_does_not_zip_fallback_after_git_update_then_subprocess_failure(
    monkeypatch, tmp_path, capsys
):
    """A later subprocess failure must not redo an already successful git update."""
    args = SimpleNamespace(branch=None, yes=False, force=True, force_venv=True)
    base_run = _make_head_moved_side_effect()
    git_update_finished = False
    late_failure_raised = False

    def run_side_effect(cmd, **kwargs):
        nonlocal git_update_finished, late_failure_raised
        if "merge" in cmd and "--ff-only" in cmd:
            git_update_finished = True
        if (
            git_update_finished
            and not late_failure_raised
            and "rev-parse" in cmd
            and "--abbrev-ref" in cmd
        ):
            # Fail the post-merge check once; the ZIP safety probe must still
            # be able to observe that HEAD advanced and refuse a fallback.
            late_failure_raised = True
            raise hermes_main.subprocess.CalledProcessError(1, cmd)
        return base_run(cmd, **kwargs)

    _patch_update_deps(monkeypatch, tmp_path, run_side_effect)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    zip_calls = []

    def fail_zip(*zip_args, **zip_kwargs):
        zip_calls.append((zip_args, zip_kwargs))
        raise AssertionError("_update_via_zip called after git update succeeded")

    monkeypatch.setattr(update_cmd, "_update_via_zip", fail_zip)

    with pytest.raises(SystemExit) as exc_info:
        hermes_main.cmd_update(args)

    assert exc_info.value.code == 1
    assert git_update_finished and late_failure_raised
    assert zip_calls == []
    out = capsys.readouterr().out
    assert "✗ Git update failed." in out
    assert "Details:" in out
    assert "Falling back to ZIP download" not in out

def test_windows_early_git_failure_falls_back_to_zip(
    monkeypatch, tmp_path, capsys
):
    """An early check=True Git failure must invoke the Windows ZIP fallback."""
    args = SimpleNamespace(branch=None, yes=False, force=True, force_venv=True)
    base_run = _make_head_moved_side_effect()

    def run_side_effect(cmd, **kwargs):
        if "rev-parse" in cmd and "--abbrev-ref" in cmd:
            assert kwargs["check"] is True
            raise hermes_main.subprocess.CalledProcessError(1, cmd)
        return base_run(cmd, **kwargs)

    _patch_update_deps(monkeypatch, tmp_path, run_side_effect)
    monkeypatch.setattr(hermes_main, "_is_windows", lambda: True)
    zip_calls = []

    def update_via_zip(*zip_args, **zip_kwargs):
        zip_calls.append((zip_args, zip_kwargs))

    monkeypatch.setattr(update_cmd, "_update_via_zip", update_via_zip)

    hermes_main.cmd_update(args)

    assert len(zip_calls) == 1
    out = capsys.readouterr().out
    assert "Falling back to ZIP download" in out
    assert "UnboundLocalError" not in out
