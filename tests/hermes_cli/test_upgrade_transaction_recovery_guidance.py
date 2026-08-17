"""Security and phase coverage for unfinished release transaction guidance."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from hermes_cli import update_cmd


SHA_OLD = "1" * 40
SHA_CANDIDATE = "2" * 40
SHA_TARGET = "3" * 40
SHA_STASH = "4" * 40
TX_ID = "a" * 32
MARKER = "hermes-update-autostash-test-123"


def _journal(tmp_path: Path, phase: str, **overrides) -> tuple[Path, dict, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    transaction_dir = repo / ".git" / "hermes-upgrade-transactions" / TX_ID
    journal_path = transaction_dir / "journal.json"
    candidate_path = repo.parent / f"hermes-upgrade-candidate-{TX_ID}"
    journal = {
        "version": 2,
        "transaction_id": TX_ID,
        "phase": phase,
        "state": phase,
        "original_branch": "main",
        "original_ref": "refs/heads/main",
        "original_head_sha": SHA_OLD,
        "maintenance_branch": "hermes-release",
        "maintenance_ref": "refs/heads/hermes-release",
        "maintenance_old_sha": SHA_OLD,
        "old_sha": SHA_OLD,
        "current_sha": SHA_CANDIDATE,
        "target_sha": SHA_TARGET,
        "backup_ref": f"refs/hermes-upgrade/backups/{TX_ID}",
        "backup_created": True,
        "candidate_branch": f"hermes-upgrade-candidate/{TX_ID}",
        "candidate_path": str(candidate_path),
        "candidate_sha": SHA_CANDIDATE,
        "candidate_created": True,
        "candidate_cleanup": phase == "finalizing",
        "stash_marker": MARKER,
        "stash_sha": SHA_STASH,
        "local_state_present": True,
        "stash_capture_required": True,
        "stash_capture_confirmed": True,
        "stash_capture_uncertain": False,
        "stash_pending": True,
        "stash_apply_attempted": False,
        "stash_applied": False,
        "payload_path": "SECRET_PAYLOAD_BYTES.patch",
        "payload_sha256": "SECRET_PAYLOAD_HASH",
        "payload_bytes": 987654,
        "environment": "SECRET_ENV_VALUE",
    }
    journal.update(overrides)
    return journal_path, journal, repo


def _commands(guidance: str) -> list[str]:
    return [line[2:] for line in guidance.splitlines() if line.startswith("$ ")]


def test_uncertain_capture_is_inspection_only_and_preserves_journal(tmp_path):
    journal_path, journal, repo = _journal(
        tmp_path,
        "stash-capture-uncertain",
        stash_sha=None,
        stash_capture_confirmed=False,
        stash_capture_uncertain=True,
        stash_pending=False,
    )

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert f"journal: {journal_path}" in guidance
    assert "phase: stash-capture-uncertain" in guidance
    assert f"repo root: {repo}" in guidance
    assert "original checkout: branch main" in guidance
    assert f"original HEAD SHA: {SHA_OLD}" in guidance
    assert "DO NOT run destructive checkout/reset/clean/stash apply/drop cleanup" in guidance
    assert "$ git status --short" in guidance
    assert "$ git stash list --format='%gd %H %s'" in guidance
    assert MARKER in guidance
    assert any(command.startswith("git show --stat ") for command in commands)
    assert not any(
        any(token in command for token in ("reset --hard", "clean", "checkout", "stash apply", "stash drop", "mv --"))
        for command in commands
    )
    assert f"{SHA_STASH}" not in guidance
    assert "SECRET_PAYLOAD" not in guidance
    assert "SECRET_ENV_VALUE" not in guidance


def test_confirmed_pre_promotion_uses_immutable_stash_sha(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "stashed")

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert "$ git stash apply --index " + SHA_STASH in guidance
    assert "stash@{" not in guidance
    assert "git checkout --force main" in commands
    assert "git reset --hard" not in "\n".join(commands)
    assert "only after verifying the original checkout identity" in guidance
    assert f"$ mv -- {journal_path} {journal_path}.resolved" in guidance


def test_post_cas_guidance_contains_exact_compare_and_swap_and_pinned_reset(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "promotion-needs-recovery")

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert (
        f"git update-ref refs/heads/hermes-release {SHA_OLD} {SHA_CANDIDATE}"
        in commands
    )
    assert f"git checkout --force hermes-release" in commands
    assert f"git reset --hard {SHA_OLD}" in commands
    assert f"git checkout --force main" in commands
    assert f"git reset --hard {SHA_OLD}" in commands
    assert f"git show --stat refs/hermes-upgrade/backups/{TX_ID}" in guidance
    assert "only after confirmed stash capture" in guidance


def test_candidate_cleanup_guidance_is_optional_and_shell_quotes_valid_path(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "candidate-cleanup-failed")
    candidate_path = repo.parent / "candidate path with spaces"
    journal["candidate_path"] = str(candidate_path)

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert "git worktree list --porcelain" in commands
    assert f"git -C '{candidate_path}' status --short" in commands
    assert "OPTIONAL final cleanup" in guidance
    assert f"git worktree remove --force -- '{candidate_path}'" in guidance
    assert f"git branch -D -- hermes-upgrade-candidate/{TX_ID}" in guidance
    assert "after live checkout and user-file/stash recovery are verified" in guidance


def test_stash_pending_conflict_repeats_only_immutable_apply(tmp_path):
    journal_path, journal, repo = _journal(
        tmp_path,
        "stash-restore-failed",
        stash_apply_attempted=True,
        stash_applied=False,
    )

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert f"git stash apply --index {SHA_STASH}" in commands
    assert "git stash drop" not in guidance
    assert "stash@{0}" not in guidance
    assert "verify files and index before any stash reflog cleanup" in guidance


def test_malformed_journal_values_are_unavailable_and_not_injected(tmp_path):
    journal_path, journal, repo = _journal(
        tmp_path,
        "promoted",
        original_branch="main\n$(SECRET_BRANCH)",
        original_ref="refs/heads/main;printf SECRET_REF",
        original_head_sha="not-a-sha",
        maintenance_branch="hermes-release\nrm -rf /",
        maintenance_ref="refs/heads/hermes-release;echo SECRET_MAINT",
        maintenance_old_sha="bad-old",
        old_sha="bad-old",
        current_sha="bad-current",
        candidate_sha="bad-candidate",
        backup_ref="refs/hermes-upgrade/backups/../../secret",
        candidate_branch="candidate\n$(SECRET_CANDIDATE)",
        candidate_path=str(tmp_path / ".." / "outside;echo SECRET_PATH"),
        stash_marker="marker\n$(SECRET_MARKER)",
        stash_sha="bad-stash",
        stash_capture_confirmed=False,
        stash_capture_uncertain=True,
    )

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert "unavailable (invalid journal value)" in guidance
    for secret in (
        "SECRET_BRANCH",
        "SECRET_REF",
        "SECRET_MAINT",
        "SECRET_CANDIDATE",
        "SECRET_PATH",
        "SECRET_MARKER",
        "SECRET_PAYLOAD",
        "SECRET_ENV_VALUE",
    ):
        assert secret not in guidance
    assert "$(" not in guidance
    assert not any(";" in command or "\n" in command for command in commands)
    assert not any(
        token in command
        for command in commands
        for token in ("update-ref", "checkout", "reset", "clean", "stash apply", "worktree remove", "branch -D")
    )


def test_missing_capture_flags_are_reported_unavailable(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "stash-capture-uncertain")
    for key in ("stash_capture_required", "stash_capture_confirmed", "stash_capture_uncertain", "stash_pending"):
        journal.pop(key)
    journal["stash_sha"] = None

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )

    for label in ("required", "confirmed", "uncertain", "pending"):
        assert f"stash capture {label}: unavailable (invalid journal value)" in guidance or (
            label == "pending" and "stash pending: unavailable (invalid journal value)" in guidance
        )


def test_candidate_path_equal_to_live_repo_cannot_be_removed(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "candidate-cleanup-failed")
    journal["candidate_path"] = str(repo)

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )

    assert "Candidate status command unavailable (invalid candidate path)." in guidance
    assert "git worktree remove --force" not in guidance
    assert "git branch -D" not in guidance


def test_unknown_phase_fails_closed_without_recovery_mutators(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "untrusted-phase")

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )
    commands = _commands(guidance)

    assert "phase: unavailable (invalid journal value)" in guidance
    assert not any(
        any(token in command for token in ("update-ref", "checkout", "reset", "clean", "stash apply", "worktree remove", "branch -D"))
        for command in commands
    )


def test_refusal_raises_with_guidance_and_still_blocks_second_transaction(tmp_path, monkeypatch):
    journal_path, journal, repo = _journal(tmp_path, "candidate-validated")
    journal_path.parent.mkdir(parents=True)
    journal_path.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        update_cmd,
        "_find_unfinished_release_transaction",
        lambda _git_cmd, _cwd: (journal_path, journal),
    )

    with pytest.raises(RuntimeError) as exc_info:
        update_cmd._reject_unfinished_release_transaction(["git"], repo)

    message = str(exc_info.value)
    assert "unfinished release transaction" in message
    assert "phase: candidate-validated" in message
    assert f"git update-ref refs/heads/hermes-release {SHA_OLD} {SHA_CANDIDATE}" not in message
    assert "git stash apply --index" in message

    with pytest.raises(RuntimeError, match="unfinished release transaction"):
        update_cmd._reject_unfinished_release_transaction(["git"], repo)


def test_real_git_journal_is_refused_with_guidance(tmp_path):
    journal_path, journal, repo = _journal(tmp_path, "candidate-validated")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    journal_path.parent.mkdir(parents=True)
    journal_path.write_text(json.dumps(journal) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError) as exc_info:
        update_cmd._reject_unfinished_release_transaction(["git"], repo)

    message = str(exc_info.value)
    assert f"journal: {journal_path}" in message
    assert "phase: candidate-validated" in message
    assert "$ git stash apply --index " + SHA_STASH in message
    assert "git update-ref" not in message


@pytest.mark.parametrize(
    "phase",
    [
        "journal-created",
        "stash-capture-uncertain",
        "stashed",
        "candidate-created",
        "candidate-validated",
        "promoted",
        "candidate-cleanup-failed",
        "stash-restore-failed",
    ],
)
def test_every_unfinished_phase_has_common_identity_and_phase_matrix(tmp_path, phase):
    uncertain = phase in {"journal-created", "stash-capture-uncertain"}
    conflict = phase == "stash-restore-failed"
    journal_path, journal, repo = _journal(
        tmp_path,
        phase,
        stash_capture_confirmed=not uncertain,
        stash_capture_uncertain=uncertain,
        stash_sha=None if uncertain else SHA_STASH,
        stash_pending=not uncertain,
        stash_apply_attempted=conflict,
        stash_applied=False,
        candidate_cleanup=False,
    )

    guidance = update_cmd._format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=repo
    )

    assert f"phase: {phase}" in guidance
    assert f"journal: {journal_path}" in guidance
    assert f"repo root: {repo}" in guidance
    assert f"original HEAD SHA: {SHA_OLD}" in guidance
    assert f"maintenance old SHA: {SHA_OLD}" in guidance
    assert f"candidate SHA: {SHA_CANDIDATE}" in guidance
    assert f"backup ref: refs/hermes-upgrade/backups/{TX_ID}" in guidance
    assert f"candidate branch: hermes-upgrade-candidate/{TX_ID}" in guidance
    assert "stash capture required: True" in guidance
    assert f"stash marker: {MARKER}" in guidance
    if uncertain:
        assert "DO NOT run destructive" in guidance
    else:
        assert f"git stash apply --index {SHA_STASH}" in guidance
