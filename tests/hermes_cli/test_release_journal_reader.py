"""Focused safety coverage for bounded release-transaction journal reads."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from hermes_cli import update_cmd


_TX_ID = "a" * 32


def _journal_paths(tmp_path: Path) -> tuple[Path, Path, Path]:
    common = tmp_path / "common"
    transactions = common / "hermes-upgrade-transactions"
    transaction = transactions / _TX_ID
    transaction.mkdir(parents=True)
    journal = transaction / "journal.json"
    journal.touch()
    return transactions, transaction, journal


def _read_journal(transactions: Path, transaction: Path, journal: Path) -> dict:
    return update_cmd._read_release_transaction_journal(
        transaction,
        journal_path=journal,
        expected_transactions_dir=transactions,
    )


def _exact_size_object(size: int) -> bytes:
    prefix = b'{"padding":"'
    suffix = b'"}'
    assert size >= len(prefix) + len(suffix)
    return prefix + (b"x" * (size - len(prefix) - len(suffix))) + suffix


def test_exact_limit_is_accepted_and_oversize_is_rejected_before_read(
    tmp_path, monkeypatch
):
    transactions, transaction, journal = _journal_paths(tmp_path)
    journal.write_bytes(_exact_size_object(update_cmd._MAX_RELEASE_JOURNAL_BYTES))

    real_read = update_cmd.os.read
    total_read = 0

    def tracked_read(fd, count):
        nonlocal total_read
        chunk = real_read(fd, count)
        total_read += len(chunk)
        return chunk

    monkeypatch.setattr(update_cmd.os, "read", tracked_read)
    value = _read_journal(transactions, transaction, journal)
    assert value["padding"]
    assert total_read <= update_cmd._MAX_RELEASE_JOURNAL_BYTES + 1

    journal.unlink()
    journal.touch()
    os.truncate(journal, update_cmd._MAX_RELEASE_JOURNAL_BYTES + 1)
    calls = 0

    def forbidden_read(fd, count):
        nonlocal calls
        calls += 1
        return real_read(fd, count)

    monkeypatch.setattr(update_cmd.os, "read", forbidden_read)
    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "oversize"
    assert calls == 0


def test_sparse_huge_journal_is_rejected_without_allocation_or_read(tmp_path, monkeypatch):
    transactions, transaction, journal = _journal_paths(tmp_path)
    journal.touch()
    os.truncate(journal, update_cmd._MAX_RELEASE_JOURNAL_BYTES * 128)
    calls = 0
    real_read = update_cmd.os.read

    def tracked_read(fd, count):
        nonlocal calls
        calls += 1
        return real_read(fd, count)

    monkeypatch.setattr(update_cmd.os, "read", tracked_read)
    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "oversize"
    assert calls == 0


@pytest.mark.parametrize("mutation", ["growth", "shrink"])
def test_size_change_during_read_fails_closed(tmp_path, monkeypatch, mutation):
    transactions, transaction, journal = _journal_paths(tmp_path)
    original = b'{"state":"original"}'
    journal.write_bytes(original)
    real_read = update_cmd.os.read
    changed = False

    def mutate_then_read(fd, count):
        nonlocal changed
        if not changed:
            changed = True
            if mutation == "growth":
                journal.write_bytes(original + b"x")
            else:
                journal.write_bytes(original[:5])
        return real_read(fd, count)

    monkeypatch.setattr(update_cmd.os, "read", mutate_then_read)
    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "changed"


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff",
        b"{not-json}",
        b"[1, 2, 3]",
        b'{"a":',
        b'{"a": NaN}',
    ],
)
def test_invalid_encoding_json_and_non_object_fail_closed(tmp_path, payload):
    transactions, transaction, journal = _journal_paths(tmp_path)
    journal.write_bytes(payload)

    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "invalid"
    assert "not-json" not in str(exc_info.value)


def test_deep_json_fails_closed_before_unbounded_parser_recursion(tmp_path):
    transactions, transaction, journal = _journal_paths(tmp_path)
    depth = update_cmd._MAX_RELEASE_JOURNAL_JSON_DEPTH + 10
    journal.write_bytes(
        b'{"nested":' + (b"{" * depth) + b"0" + (b"}" * depth) + b"}"
    )

    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "invalid"


@pytest.mark.parametrize("entry_kind", ["journal-symlink", "transaction-symlink", "fifo", "directory"])
def test_symlink_fifo_and_directory_are_rejected_without_outside_read(
    tmp_path, entry_kind
):
    transactions, transaction, journal = _journal_paths(tmp_path)
    outside = tmp_path / "outside-secret.txt"
    outside.write_bytes(b"DO NOT READ OR MODIFY")

    if entry_kind == "journal-symlink":
        journal.unlink()
        journal.symlink_to(outside)
    elif entry_kind == "transaction-symlink":
        journal.unlink()
        transaction.rmdir()
        outside_transaction = tmp_path / "outside-transaction"
        outside_transaction.mkdir()
        (outside_transaction / "journal.json").write_bytes(outside.read_bytes())
        transaction.symlink_to(outside_transaction, target_is_directory=True)
    elif entry_kind == "fifo":
        if not hasattr(os, "mkfifo"):
            pytest.skip("FIFO creation is unavailable")
        journal.unlink()
        os.mkfifo(journal)
    else:
        journal.unlink()
        journal.mkdir()

    before = outside.read_bytes()
    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "unsafe"
    assert outside.read_bytes() == before


@pytest.mark.skipif(
    not hasattr(os, "mknod") or os.name == "nt",
    reason="device-node creation is unavailable",
)
def test_device_node_is_rejected_without_opening_or_blocking(tmp_path):
    transactions, transaction, journal = _journal_paths(tmp_path)
    journal.unlink()
    try:
        os.mknod(journal, stat.S_IFCHR | 0o600, os.makedev(1, 3))
    except PermissionError:
        pytest.skip("device-node creation requires elevated privileges")

    with pytest.raises(update_cmd._ReleaseJournalReadError) as exc_info:
        _read_journal(transactions, transaction, journal)
    assert exc_info.value.kind == "unsafe"


def test_replacement_after_open_returns_anchored_original_not_replacement(tmp_path, monkeypatch):
    if not (
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and update_cmd.os.open in getattr(os, "supports_dir_fd", ())
    ):
        pytest.skip("descriptor-relative no-follow opening is unavailable")
    transactions, transaction, journal = _journal_paths(tmp_path)
    original = b'{"marker":"original"}'
    replacement = b'{"marker":"replacement"}'
    journal.write_bytes(original)
    saved = tmp_path / "opened-original.json"
    real_open = update_cmd.os.open
    replaced = False

    def replace_after_open(path, flags, *args, **kwargs):
        nonlocal replaced
        fd = real_open(path, flags, *args, **kwargs)
        if path == "journal.json" and not replaced:
            replaced = True
            journal.replace(saved)
            journal.write_bytes(replacement)
        return fd

    monkeypatch.setattr(update_cmd.os, "open", replace_after_open)
    value = _read_journal(transactions, transaction, journal)
    assert replaced is True
    assert value == {"marker": "original"}
    assert saved.read_bytes() == original
    assert journal.read_bytes() == replacement


@pytest.mark.parametrize("raised", [BaseException, KeyboardInterrupt])
def test_baseexception_during_read_closes_parent_and_journal_fds(
    tmp_path, monkeypatch, raised
):
    transactions, transaction, journal = _journal_paths(tmp_path)
    journal.write_bytes(b'{"ok":true}')
    real_open = update_cmd.os.open
    real_close = update_cmd.os.close
    opened: set[int] = set()
    closed: set[int] = set()

    def tracked_open(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        opened.add(fd)
        return fd

    def tracked_close(fd):
        closed.add(fd)
        return real_close(fd)

    def fail_read(_fd, _count):
        raise raised("injected read control exception")

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    monkeypatch.setattr(update_cmd.os, "close", tracked_close)
    monkeypatch.setattr(update_cmd.os, "read", fail_read)
    with pytest.raises(raised):
        _read_journal(transactions, transaction, journal)
    assert opened == closed


def test_posix_open_flags_anchor_parent_and_journal_without_following(tmp_path, monkeypatch):
    if not (
        os.name == "posix"
        and hasattr(os, "O_NOFOLLOW")
        and update_cmd.os.open in getattr(os, "supports_dir_fd", ())
    ):
        pytest.skip("descriptor-relative no-follow opening is unavailable")
    transactions, transaction, journal = _journal_paths(tmp_path)
    journal.write_bytes(b'{"ok":true}')
    real_open = update_cmd.os.open
    calls: list[tuple[object, int, dict]] = []

    def tracked_open(path, flags, *args, **kwargs):
        calls.append((path, flags, dict(kwargs)))
        return real_open(path, flags, *args, **kwargs)

    monkeypatch.setattr(update_cmd.os, "open", tracked_open)
    assert _read_journal(transactions, transaction, journal) == {"ok": True}

    journal_calls = [call for call in calls if call[0] == "journal.json"]
    assert len(journal_calls) == 1
    _path, journal_flags, journal_kwargs = journal_calls[0]
    assert journal_kwargs.get("dir_fd") is not None
    assert journal_flags & os.O_NOFOLLOW
    if hasattr(os, "O_NONBLOCK"):
        assert journal_flags & os.O_NONBLOCK
    if hasattr(os, "O_CLOEXEC"):
        assert journal_flags & os.O_CLOEXEC

    parent_calls = [call for call in calls if call[0] == str(transactions)]
    assert parent_calls
    _path, parent_flags, _kwargs = parent_calls[0]
    if hasattr(os, "O_DIRECTORY"):
        assert parent_flags & os.O_DIRECTORY
    assert parent_flags & os.O_NOFOLLOW
    if hasattr(os, "O_CLOEXEC"):
        assert parent_flags & os.O_CLOEXEC


def test_discovery_refuses_invalid_or_oversize_journal_without_exposing_contents(
    tmp_path, monkeypatch
):
    common = tmp_path / "common"
    transactions = common / "hermes-upgrade-transactions"
    transaction = transactions / _TX_ID
    transaction.mkdir(parents=True)
    monkeypatch.setattr(update_cmd, "_git_common_dir", lambda *_args: common)
    secret = "TOP_SECRET_JOURNAL_CONTENT"

    journal = transaction / "journal.json"
    journal.write_bytes(("{" + secret).encode())
    with pytest.raises(RuntimeError, match="journal") as invalid_info:
        update_cmd._find_unfinished_release_transaction(["git"], tmp_path)
    assert secret not in str(invalid_info.value)

    journal.unlink()
    journal.touch()
    os.truncate(journal, update_cmd._MAX_RELEASE_JOURNAL_BYTES + 1)
    with pytest.raises(RuntimeError, match="journal") as oversize_info:
        update_cmd._find_unfinished_release_transaction(["git"], tmp_path)
    assert secret not in str(oversize_info.value)


def _uncertain_context(tmp_path: Path) -> update_cmd.ReleaseUpgradeContext:
    common = tmp_path / "common"
    transactions = common / "hermes-upgrade-transactions"
    transaction = transactions / _TX_ID
    transaction.mkdir(parents=True)
    journal_path = transaction / "journal.json"
    candidate_path = tmp_path / f"hermes-upgrade-candidate-{_TX_ID}"
    journal = {
        "version": 2,
        "transaction_id": _TX_ID,
        "phase": "finalizing",
        "state": "finalizing",
        "original_branch": "main",
        "original_ref": "refs/heads/main",
        "original_head_sha": "1" * 40,
        "maintenance_branch": "hermes-release",
        "maintenance_old_sha": "1" * 40,
        "old_sha": "1" * 40,
        "release_tag": "v2.0.0",
        "base_sha": "1" * 40,
        "target_sha": "2" * 40,
        "backup_ref": f"refs/hermes-upgrade/backups/{_TX_ID}",
        "backup_created": True,
        "candidate_branch": f"hermes-upgrade-candidate/{_TX_ID}",
        "candidate_path": str(candidate_path),
        "candidate_sha": "2" * 40,
        "candidate_cleanup": False,
        "candidate_created": True,
        "payload_path": str(transaction / "runtime-local-maintenance.patch"),
        "payload_sha256": "3" * 64,
        "payload_bytes": 0,
        "stash_marker": "hermes-test-marker",
        "stash_sha": None,
        "local_state_present": False,
        "stash_capture_required": False,
        "stash_capture_confirmed": False,
        "stash_capture_uncertain": False,
        "stash_pending": False,
        "stash_apply_attempted": False,
        "stash_applied": False,
        "checkout_restored": False,
        "final_state_verified": False,
        "finalized": False,
    }
    candidate = dict(journal)
    candidate.update(
        {
            "phase": "finalized",
            "state": "finalized",
            "final_state_verified": True,
            "finalized": True,
        }
    )
    journal_path.write_text(json.dumps(journal), encoding="utf-8")
    return update_cmd.ReleaseUpgradeContext(
        root=tmp_path / "repo",
        common_dir=common,
        transaction_dir=transaction,
        journal_path=journal_path,
        journal=journal,
        final_marker_write_uncertain=True,
        final_marker_candidate=candidate,
        final_marker_prior_digest=update_cmd._terminal_journal_digest(journal),
    )


@pytest.mark.parametrize("tamper", ["oversize", "symlink"])
def test_uncertain_reconciliation_read_failure_is_git_free_and_retains_latch(
    tmp_path, monkeypatch, tamper
):
    context = _uncertain_context(tmp_path)
    before_journal = dict(context.journal)
    before_candidate = dict(context.final_marker_candidate)
    outside = tmp_path / "outside-journal.json"
    outside.write_bytes(b"outside evidence")
    if tamper == "oversize":
        context.journal_path.unlink()
        context.journal_path.touch()
        os.truncate(
            context.journal_path, update_cmd._MAX_RELEASE_JOURNAL_BYTES + 1
        )
    else:
        context.journal_path.unlink()
        context.journal_path.symlink_to(outside)

    def forbidden_git(*_args, **_kwargs):
        raise AssertionError("journal read failure must not invoke Git/subprocess")

    monkeypatch.setattr(update_cmd.subprocess, "run", forbidden_git)
    assert update_cmd._finalize_release_upgrade(["git"], context.root, context) is False
    assert context.final_marker_write_uncertain is True
    assert context.journal == before_journal
    assert context.final_marker_candidate == before_candidate
    assert outside.read_bytes() == b"outside evidence"
