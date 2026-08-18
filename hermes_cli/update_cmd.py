"""Hermes update pipeline — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): ``_cmd_update_impl``, ``_cmd_update_check``
and every module-level helper used only by the update path, plus the update-only
constants they read. Function bodies are lifted verbatim; the only mechanical
change is that references to helpers/constants that STAY in ``hermes_cli.main``
(and to moved-but-test-patched siblings) are routed through ``_m()`` — a lazy
``hermes_cli.main`` reference — so existing call sites and test monkeypatches
that target ``hermes_cli.main.<name>`` (``PROJECT_ROOT``, ``_is_windows``,
``_run_pre_update_backup``, ...) keep working unchanged. ``main.py`` re-imports
every public-ish name from here (``# noqa: F401``) so the argparse wiring and
the test-patch surface still resolve on ``hermes_cli.main``.

Three self-contained closures nested inside ``_cmd_update_impl``
(``_print_items``, ``_wait_for_service_active``, ``_service_restart_sec``) were
hoisted to module level; they capture no enclosing state (verified via
``symtable``). ``_restart_one_systemd_gateway_unit``, ``_resolve_manage_cmd``
and ``_on_unit_timeout`` DO capture enclosing locals and stay nested,
byte-identical.

Imports are one-way: ``hermes_cli.main`` imports this module, never the reverse
at import time (``_m()`` resolves lazily at call time, when main.py is fully
loaded, so there is no import cycle).
"""

import hashlib
import json
import logging
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import time as _time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import NoReturn, Optional
from urllib.parse import urlsplit

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

try:
    import msvcrt
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None

from hermes_cli.config import get_hermes_home
from hermes_constants import (
    FIRST_PARTY_MODULE_ROOTS,
    is_first_party_module,
    venv_python_path,
)

logger = logging.getLogger(__name__)


# Release journals are local recovery evidence, not arbitrary payload files.
# The current strict journal is only a few KiB; keeping a 1 MiB ceiling leaves
# ample room for schema-compatible additions while bounding parser work.
_MAX_RELEASE_JOURNAL_BYTES = 1024 * 1024
_MAX_RELEASE_JOURNAL_JSON_DEPTH = 64
_RELEASE_JOURNAL_FILENAME = "journal.json"
_RELEASE_JOURNAL_READ_CHUNK = 64 * 1024

# Keep capability detection stable while tests or callers instrument os.open.
# ``os.supports_dir_fd`` contains the original built-in function object, not a
# wrapper installed later by a fault-injection test.
_RELEASE_OS_OPEN = os.open
_RELEASE_OS_OPEN_SUPPORTS_DIR_FD = _RELEASE_OS_OPEN in getattr(
    os, "supports_dir_fd", ()
)


class _ReleaseJournalReadError(RuntimeError):
    """Stable, content-free failure categories for untrusted journal reads."""

    _MESSAGES = {
        "missing": "Release transaction journal is missing.",
        "unsafe": "Release transaction journal path or file is unsafe.",
        "oversize": "Release transaction journal exceeds the maximum allowed size.",
        "changed": "Release transaction journal changed while being read.",
        "invalid": "Release transaction journal is invalid.",
        "io": "Release transaction journal could not be read.",
    }

    def __init__(self, kind: str):
        self.kind = kind
        super().__init__(self._MESSAGES.get(kind, self._MESSAGES["io"]))


def _raise_release_journal_error(
    kind: str, cause: BaseException | None = None
) -> NoReturn:
    error = _ReleaseJournalReadError(kind)
    if cause is None:
        raise error
    raise error from cause


def _release_journal_lexical_path(path: Path | str) -> Path:
    """Normalize an absolute journal path without resolving symlinks."""

    try:
        value = Path(path)
    except (TypeError, ValueError) as exc:
        _raise_release_journal_error("unsafe", exc)
    if not value.is_absolute():
        _raise_release_journal_error("unsafe")
    # abspath/normpath remove lexical ``.``/``..`` components but do not
    # follow symlinks.  That distinction is required before no-follow opens.
    return Path(os.path.abspath(os.fspath(value)))


def _validate_release_journal_topology(
    transaction_dir: Path | str,
    journal_path: Path | str,
    expected_transactions_dir: Path | str,
) -> tuple[Path, Path, Path]:
    """Require ``<transactions>/<child>/journal.json`` exactly."""

    transactions = _release_journal_lexical_path(expected_transactions_dir)
    transaction = _release_journal_lexical_path(transaction_dir)
    journal = _release_journal_lexical_path(journal_path)
    if (
        transactions.name != "hermes-upgrade-transactions"
        or transaction.parent != transactions
        or journal != transaction / _RELEASE_JOURNAL_FILENAME
        or transaction.name in {"", ".", ".."}
    ):
        _raise_release_journal_error("unsafe")
    return transactions, transaction, journal


def _release_lstat(path: Path, *, missing_kind: str = "missing") -> os.stat_result:
    try:
        return os.lstat(path)
    except FileNotFoundError as exc:
        _raise_release_journal_error(missing_kind, exc)
    except OSError as exc:
        _raise_release_journal_error("unsafe", exc)


def _release_require_directory(info: os.stat_result) -> None:
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        _raise_release_journal_error("unsafe")


def _release_require_regular_file(info: os.stat_result) -> None:
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        _raise_release_journal_error("unsafe")


def _release_stat_identity(
    info: os.stat_result, *, include_size: bool
) -> tuple[int, int, int, int | None]:
    return (
        int(getattr(info, "st_dev", 0)),
        int(getattr(info, "st_ino", 0)),
        stat.S_IFMT(info.st_mode),
        int(getattr(info, "st_size", 0)) if include_size else None,
    )


def _release_same_stat(
    first: os.stat_result, second: os.stat_result, *, include_size: bool
) -> bool:
    return _release_stat_identity(first, include_size=include_size) == _release_stat_identity(
        second, include_size=include_size
    )


def _release_parent_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_DIRECTORY", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    return flags


def _release_journal_open_flags() -> int:
    flags = os.O_RDONLY
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    return flags


def _release_open_relative(path: str, flags: int, dir_fd: int) -> int:
    """Call relative ``os.open`` while tolerating legacy test wrappers."""

    current_open = os.open
    try:
        return current_open(path, flags, dir_fd=dir_fd)
    except TypeError:
        # A few pre-existing lifecycle tests wrap os.open with the historical
        # ``(path, flags, *args)`` shape.  Do not weaken the real POSIX path or
        # silently fall back on an actual platform error; bypass only a
        # replaced wrapper that cannot express the required keyword.
        if current_open is not _RELEASE_OS_OPEN:
            return _RELEASE_OS_OPEN(path, flags, dir_fd=dir_fd)
        raise


def _release_close_fd(fd: int) -> None:
    try:
        os.close(fd)
    except OSError:
        # The read error is authoritative; do not replace it with close noise.
        logger.debug("Could not close release journal descriptor", exc_info=True)


def _release_check_json_depth(text: str) -> None:
    depth = 0
    in_string = False
    escaped = False
    for char in text:
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char in "[{":
            depth += 1
            if depth > _MAX_RELEASE_JOURNAL_JSON_DEPTH:
                _raise_release_journal_error("invalid")
        elif char in "]}":
            depth -= 1


def _parse_release_transaction_journal(raw: bytes) -> dict:
    try:
        text = raw.decode("utf-8")
        _release_check_json_depth(text)

        def reject_constant(_value: str):
            raise ValueError("non-standard JSON constant")

        value = json.loads(text, parse_constant=reject_constant)
    except _ReleaseJournalReadError:
        raise
    except (UnicodeDecodeError, ValueError, RecursionError, MemoryError, OverflowError):
        _raise_release_journal_error("invalid")
    if not isinstance(value, dict):
        _raise_release_journal_error("invalid")
    return value


def _read_release_journal_fd(
    journal_fd: int,
    *,
    expected_info: os.stat_result,
) -> dict:
    """Read one already-open journal descriptor with a hard byte ceiling."""

    try:
        initial_info = os.fstat(journal_fd)
    except OSError as exc:
        _raise_release_journal_error("io", exc)
    _release_require_regular_file(initial_info)
    if not _release_same_stat(initial_info, expected_info, include_size=True):
        _raise_release_journal_error("changed")
    initial_size = int(getattr(initial_info, "st_size", -1))
    if initial_size < 0 or initial_size > _MAX_RELEASE_JOURNAL_BYTES:
        _raise_release_journal_error("oversize")

    chunks: list[bytes] = []
    total = 0
    while total <= _MAX_RELEASE_JOURNAL_BYTES:
        remaining = _MAX_RELEASE_JOURNAL_BYTES + 1 - total
        if remaining <= 0:
            break
        try:
            chunk = os.read(journal_fd, min(_RELEASE_JOURNAL_READ_CHUNK, remaining))
        except OSError as exc:
            _raise_release_journal_error("io", exc)
        if not isinstance(chunk, bytes):
            _raise_release_journal_error("io")
        if not chunk:
            break
        # os.read honors the requested count.  Keep the guard anyway so a
        # fault-injected or non-conforming implementation cannot make the
        # in-memory buffer exceed the max+1 probe bound.
        if len(chunk) > remaining:
            chunks.append(chunk[:remaining])
            total += remaining
            break
        chunks.append(chunk)
        total += len(chunk)
        if total >= _MAX_RELEASE_JOURNAL_BYTES + 1:
            break

    raw = b"".join(chunks)
    try:
        final_info = os.fstat(journal_fd)
    except OSError as exc:
        _raise_release_journal_error("io", exc)
    _release_require_regular_file(final_info)
    final_size = int(getattr(final_info, "st_size", -1))
    if final_size < 0 or final_size > _MAX_RELEASE_JOURNAL_BYTES:
        _raise_release_journal_error("oversize")
    if total > _MAX_RELEASE_JOURNAL_BYTES:
        _raise_release_journal_error("oversize")
    if final_size != initial_size or total != final_size:
        _raise_release_journal_error("changed")
    return _parse_release_transaction_journal(raw)


def _read_release_transaction_journal_posix(
    transactions: Path,
    transaction: Path,
    journal: Path,
    transactions_info: os.stat_result,
    transaction_info: os.stat_result,
    journal_info: os.stat_result,
) -> dict:
    """Use descriptor-relative no-follow opens for the POSIX path."""

    transactions_fd: int | None = None
    transaction_fd: int | None = None
    journal_fd: int | None = None
    try:
        try:
            transactions_fd = os.open(str(transactions), _release_parent_open_flags())
        except FileNotFoundError as exc:
            _raise_release_journal_error("missing", exc)
        except OSError as exc:
            _raise_release_journal_error("unsafe", exc)
        assert transactions_fd is not None
        try:
            opened_transactions_info = os.fstat(transactions_fd)
        except OSError as exc:
            _raise_release_journal_error("io", exc)
        _release_require_directory(opened_transactions_info)
        if not _release_same_stat(
            opened_transactions_info, transactions_info, include_size=False
        ):
            _raise_release_journal_error("changed")

        try:
            transaction_fd = _release_open_relative(
                transaction.name,
                _release_parent_open_flags(),
                transactions_fd,
            )
        except FileNotFoundError as exc:
            _raise_release_journal_error("missing", exc)
        except OSError as exc:
            _raise_release_journal_error("unsafe", exc)
        assert transaction_fd is not None
        try:
            opened_transaction_info = os.fstat(transaction_fd)
        except OSError as exc:
            _raise_release_journal_error("io", exc)
        _release_require_directory(opened_transaction_info)
        if not _release_same_stat(
            opened_transaction_info, transaction_info, include_size=False
        ):
            _raise_release_journal_error("changed")

        try:
            journal_fd = _release_open_relative(
                _RELEASE_JOURNAL_FILENAME,
                _release_journal_open_flags(),
                transaction_fd,
            )
        except FileNotFoundError as exc:
            _raise_release_journal_error("missing", exc)
        except OSError as exc:
            _raise_release_journal_error("unsafe", exc)
        assert journal_fd is not None
        return _read_release_journal_fd(journal_fd, expected_info=journal_info)
    finally:
        if journal_fd is not None:
            _release_close_fd(journal_fd)
        if transaction_fd is not None:
            _release_close_fd(transaction_fd)
        if transactions_fd is not None:
            _release_close_fd(transactions_fd)


def _read_release_transaction_journal_portable(
    transactions: Path,
    transaction: Path,
    journal: Path,
    transactions_info: os.stat_result,
    transaction_info: os.stat_result,
    journal_info: os.stat_result,
) -> dict:
    """Best-effort portable fallback where dir_fd/no-follow is unavailable.

    The fallback lstat-checks the parent and journal before and after opening,
    opens with the platform's binary/non-inheritable flags when available, and
    compares descriptor identity, type, and size.  The stdlib has no portable
    race-free no-follow equivalent on Windows/reparse-point filesystems; a
    mismatch fails closed rather than claiming POSIX descriptor anchoring.
    """

    journal_fd: int | None = None
    try:
        try:
            journal_fd = os.open(
                str(journal),
                _release_journal_open_flags()
                | getattr(os, "O_BINARY", 0)
                | getattr(os, "O_NOINHERIT", 0),
            )
        except FileNotFoundError as exc:
            _raise_release_journal_error("missing", exc)
        except OSError as exc:
            _raise_release_journal_error("unsafe", exc)
        assert journal_fd is not None

        try:
            current_transactions_info = os.lstat(transactions)
            current_transaction_info = os.lstat(transaction)
            current_journal_info = os.lstat(journal)
        except FileNotFoundError as exc:
            _raise_release_journal_error("changed", exc)
        except OSError as exc:
            _raise_release_journal_error("unsafe", exc)
        _release_require_directory(current_transactions_info)
        _release_require_directory(current_transaction_info)
        _release_require_regular_file(current_journal_info)
        if not _release_same_stat(
            current_transactions_info, transactions_info, include_size=False
        ) or not _release_same_stat(
            current_transaction_info, transaction_info, include_size=False
        ):
            _raise_release_journal_error("changed")
        if not _release_same_stat(current_journal_info, journal_info, include_size=True):
            _raise_release_journal_error("changed")
        return _read_release_journal_fd(journal_fd, expected_info=journal_info)
    finally:
        if journal_fd is not None:
            _release_close_fd(journal_fd)


def _read_release_transaction_journal(
    transaction_dir: Path | str,
    *,
    journal_path: Path | str | None = None,
    expected_transactions_dir: Path | str | None = None,
) -> dict:
    """Read a release journal through one bounded, no-follow descriptor path.

    On POSIX this anchors the transactions root, transaction child, and
    ``journal.json`` through directory descriptors.  On platforms without
    ``dir_fd``/``O_NOFOLLOW`` it uses the documented best-effort lstat/fstat
    fallback above and rejects identity changes.
    """

    transaction_value = Path(transaction_dir)
    expected_value = (
        Path(expected_transactions_dir)
        if expected_transactions_dir is not None
        else transaction_value.parent
    )
    journal_value = (
        Path(journal_path)
        if journal_path is not None
        else transaction_value / _RELEASE_JOURNAL_FILENAME
    )
    transactions, transaction, journal = _validate_release_journal_topology(
        transaction_value, journal_value, expected_value
    )
    transactions_info = _release_lstat(transactions)
    transaction_info = _release_lstat(transaction)
    journal_info = _release_lstat(journal)
    _release_require_directory(transactions_info)
    _release_require_directory(transaction_info)
    _release_require_regular_file(journal_info)
    journal_size = int(getattr(journal_info, "st_size", -1))
    if journal_size < 0 or journal_size > _MAX_RELEASE_JOURNAL_BYTES:
        _raise_release_journal_error("oversize")

    if (
        os.name == "posix"
        and _RELEASE_OS_OPEN_SUPPORTS_DIR_FD
        and bool(getattr(os, "O_NOFOLLOW", 0))
    ):
        return _read_release_transaction_journal_posix(
            transactions,
            transaction,
            journal,
            transactions_info,
            transaction_info,
            journal_info,
        )
    return _read_release_transaction_journal_portable(
        transactions,
        transaction,
        journal,
        transactions_info,
        transaction_info,
        journal_info,
    )


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Lets callers keep patching ``hermes_cli.main.<helper>`` (the historical
    test surface) and have those patches reach this code path, and defers the
    import so ``hermes_cli.main`` -> ``hermes_cli.update_cmd`` stays one-way
    at import time.
    """
    from hermes_cli import main

    return main


@dataclass(frozen=True)
class ReleaseTarget:
    """An exact, immutable release target resolved from one tag ref."""

    tag: str
    target_sha: str
    ref: str


@dataclass(frozen=True)
class ReleaseBaseMetadata:
    """Validated metadata describing the base of the maintenance delta."""

    tag: str
    base_sha: str
    target_sha: str | None = None
    patch_sha256: str | None = None
    patch_bytes: int | None = None
    format_version: int = 1
    integration_mode: str = "snapshot-patch"


@dataclass(frozen=True)
class ReleaseGitSnapshot:
    """Exact live-checkout identity used to guard release mutations."""

    root: Path
    common_dir: Path
    symbolic_head: str | None
    head_sha: str
    head_tree_sha: str
    maintenance_ref: str
    maintenance_ref_sha: str
    index_tree_sha: str
    tracked_diff_sha256: str
    status_v2_sha256: str
    # The status digest already commits to the exact untracked names and
    # bytes.  Keep only a bounded count here so transaction journals do not
    # persist a potentially sensitive path list.
    untracked_count: int

    def to_journal(self) -> dict:
        value = {
            "root": str(self.root),
            "common_dir": str(self.common_dir),
            "symbolic_head": self.symbolic_head,
            "head_sha": self.head_sha,
            "head_tree_sha": self.head_tree_sha,
            "maintenance_ref": self.maintenance_ref,
            "maintenance_ref_sha": self.maintenance_ref_sha,
            "index_tree_sha": self.index_tree_sha,
            "tracked_diff_sha256": self.tracked_diff_sha256,
            "status_v2_sha256": self.status_v2_sha256,
            "untracked_count": self.untracked_count,
        }
        _validate_release_snapshot_value(value)
        return dict(value)

    @classmethod
    def from_journal(cls, value: object) -> "ReleaseGitSnapshot":
        _validate_release_snapshot_value(value)
        assert isinstance(value, dict)
        root = value["root"]
        common_dir = value["common_dir"]
        symbolic_head = value["symbolic_head"]
        assert isinstance(root, str)
        assert isinstance(common_dir, str)
        assert symbolic_head is None or isinstance(symbolic_head, str)
        return cls(
            root=Path(root),
            common_dir=Path(common_dir),
            symbolic_head=symbolic_head,
            head_sha=value["head_sha"].lower(),
            head_tree_sha=value["head_tree_sha"].lower(),
            maintenance_ref=value["maintenance_ref"],
            maintenance_ref_sha=value["maintenance_ref_sha"].lower(),
            index_tree_sha=value["index_tree_sha"].lower(),
            tracked_diff_sha256=value["tracked_diff_sha256"].lower(),
            status_v2_sha256=value["status_v2_sha256"].lower(),
            untracked_count=value["untracked_count"],
        )


@dataclass(frozen=True)
class ReleaseUpgradeResult:
    """The durable result of a promoted candidate."""

    old_sha: str
    target_sha: str
    candidate_sha: str
    backup_ref: str
    candidate_path: Path | None = None
    context: "ReleaseUpgradeContext | None" = None


@dataclass
class ReleaseUpgradeContext:
    """Durable user-state state carried from promotion to outer finalization."""

    root: Path
    common_dir: Path
    transaction_dir: Path
    journal_path: Path
    journal: dict
    final_marker_write_uncertain: bool = False
    final_marker_candidate: Optional[dict] = None
    final_marker_prior_digest: str | None = None
    snapshots: dict[str, ReleaseGitSnapshot] = field(default_factory=dict)


def _git_common_dir(git_cmd: list[str], cwd: Path | str) -> Path:
    """Resolve the repository's common Git directory, including worktrees."""

    result = subprocess.run(
        git_cmd + ["rev-parse", "--git-common-dir"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0 or not result.stdout.strip():
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(
            "Could not resolve Git common directory for update lock"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    common = Path(result.stdout.strip())
    if not common.is_absolute():
        common = Path(cwd) / common
    return common.resolve()


def _directory_fsync_is_windows() -> bool:
    """Return whether directory fsync must use the Windows best-effort path."""

    return os.name == "nt" or sys.platform == "win32"


def _fsync_directory(path: Path | str, *, required: bool) -> None:
    """Fsync a directory entry, failing closed where POSIX requires it.

    POSIX opens use ``O_DIRECTORY`` when the platform provides it, and a
    required open/fsync failure is propagated.  Windows does not provide a
    portable directory-fsync contract; it gets an explicit best-effort
    attempt and a warning rather than a false claim of POSIX-equivalent
    durability.
    """

    directory = Path(path)
    windows = _directory_fsync_is_windows()
    flags = os.O_RDONLY
    if not windows:
        flags |= getattr(os, "O_DIRECTORY", 0)
    fd: int | None = None
    try:
        fd = os.open(str(directory), flags)
        os.fsync(fd)
    except (OSError, NotImplementedError) as exc:
        if required and not windows:
            raise
        logger.warning(
            "Directory fsync is best effort on this platform for %s: %s",
            directory,
            exc,
        )
    finally:
        if fd is not None:
            try:
                os.close(fd)
            except OSError as exc:
                if required and not windows:
                    raise
                logger.warning(
                    "Could not close best-effort directory fsync handle for %s: %s",
                    directory,
                    exc,
                )


def _atomic_write_bytes(
    path: Path, payload: bytes, *, required_parent_fsync: bool = False
) -> None:
    """Write a regular file with fsync + replace, never following symlinks."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        existing = path.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None:
        if stat.S_ISLNK(existing.st_mode):
            raise RuntimeError(f"Refusing to write through symlink: {path}")
        if not stat.S_ISREG(existing.st_mode):
            raise RuntimeError(f"Refusing to replace non-regular file: {path}")

    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
        if required_parent_fsync:
            _fsync_directory(path.parent, required=True)
        else:
            try:
                dir_fd = os.open(str(path.parent), os.O_RDONLY)
            except OSError:
                dir_fd = None
            if dir_fd is not None:
                try:
                    os.fsync(dir_fd)
                finally:
                    os.close(dir_fd)
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


class RepositoryUpdateLock:
    """Cross-process advisory lock scoped to a Git repository common dir.

    On POSIX the lock is held on the Git common-directory inode itself, not on
    the metadata pathname.  Removing or replacing ``hermes-update.lock`` can
    therefore never create a new inode that another updater can acquire.  The
    metadata file is only an atomically-written diagnostic for operators.
    """

    def __init__(self, repo_root: Path | str, git_cmd: list[str] | None = None):
        self.repo_root = Path(repo_root)
        self.git_cmd = list(git_cmd or ["git"])
        self.lock_path: Path | None = None
        self._handle = None
        self._handle_is_fd = False
        self.acquired = False

    @staticmethod
    def _holder_text(path: Path) -> str:
        try:
            return path.read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            return ""

    @staticmethod
    def _close_handle(handle, *, is_fd: bool) -> None:
        try:
            if is_fd:
                os.close(handle)
            else:
                handle.close()
        except OSError:
            logger.debug("Could not close repository update lock", exc_info=True)

    def acquire(self):
        if self.acquired:
            return self
        common = _git_common_dir(self.git_cmd, self.repo_root)
        self.lock_path = common / "hermes-update.lock"

        # POSIX directory descriptors are stable across metadata unlink/replace
        # races.  Windows keeps the open regular file because NTFS denies
        # unlinking an open file and byte-range locking is its compatible
        # cross-process primitive.
        handle_is_fd = fcntl is not None
        try:
            if handle_is_fd:
                handle = os.open(str(common), os.O_RDONLY)
            else:  # pragma: no cover - exercised on Windows
                handle = self.lock_path.open("a+b")
        except OSError as exc:
            raise RuntimeError(
                f"Cannot open repository update lock {self.lock_path}: {exc}"
            ) from exc

        try:
            if handle_is_fd:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            elif msvcrt is not None:  # pragma: no cover - exercised on Windows
                handle.write(b"0")
                handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:  # pragma: no cover - unsupported platform
                raise OSError("no cross-process locking primitive is available")
        except OSError as exc:
            holder = self._holder_text(self.lock_path)
            self._close_handle(handle, is_fd=handle_is_fd)
            detail = f" ({holder})" if holder else ""
            raise RuntimeError(
                "Cannot acquire repository update lock "
                f"{self.lock_path}: another hermes update/upgrade is already "
                f"running{detail}. Wait for it to finish; do not delete the lock file."
            ) from exc

        # Mark ownership before writing diagnostics.  If metadata I/O fails,
        # release() can see and unlock the acquired kernel handle in this same
        # exception path instead of leaving it unreachable.
        self._handle = handle
        self._handle_is_fd = handle_is_fd
        self.acquired = True
        metadata = (
            f"pid={os.getpid()}\n"
            f"started={_time.time():.6f}\n"
            f"cwd={self.repo_root}\n"
        ).encode("utf-8")
        try:
            if handle_is_fd:
                _atomic_write_bytes(self.lock_path, metadata)
            else:  # pragma: no cover - exercised on Windows
                handle.seek(0)
                handle.truncate()
                handle.write(metadata)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException as exc:
            self.release()
            raise RuntimeError(
                f"Could not write repository update lock metadata {self.lock_path}; "
                "the lock was released."
            ) from exc
        return self

    def release(self) -> None:
        handle = self._handle
        handle_is_fd = self._handle_is_fd
        self._handle = None
        self._handle_is_fd = False
        self.acquired = False
        if handle is None:
            return
        try:
            if handle_is_fd:
                fcntl.flock(handle, fcntl.LOCK_UN)
            elif msvcrt is not None:  # pragma: no cover - exercised on Windows
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        except OSError:
            logger.debug("Could not release repository update lock", exc_info=True)
        finally:
            self._close_handle(handle, is_fd=handle_is_fd)

    def __enter__(self):
        return self.acquire()

    def __exit__(self, _exc_type, _exc, _tb):
        self.release()
        return False


_SHA_RE = re.compile(r"^[0-9a-fA-F]{40}$")
_RELEASE_SNAPSHOT_FIELDS = frozenset(
    {
        "root",
        "common_dir",
        "symbolic_head",
        "head_sha",
        "head_tree_sha",
        "maintenance_ref",
        "maintenance_ref_sha",
        "index_tree_sha",
        "tracked_diff_sha256",
        "status_v2_sha256",
        "untracked_count",
    }
)
_RELEASE_SNAPSHOT_MAX_PATH = 4096
_RELEASE_SNAPSHOT_MAX_REF = 1024
_RELEASE_SNAPSHOT_MAX_UNTRACKED = 100_000
_RELEASE_EMPTY_DIGEST = hashlib.sha256(b"").hexdigest()
_RELEASE_MAINTENANCE_REF = "refs/heads/hermes-release"


class ReleaseGitStateError(RuntimeError):
    """The live checkout no longer matches a durable release snapshot."""


class ReleaseFinalizationIncompleteError(RuntimeError):
    """Release user-state finalization did not reach a verified terminal state."""

    def __init__(self, journal_path: Path | str):
        self.journal_path = Path(journal_path)
        super().__init__(
            "Release finalization incomplete; recovery journal retained at "
            f"{self.journal_path}."
        )


def _release_snapshot_text(value: object, *, label: str, limit: int) -> str:
    if not isinstance(value, str) or not value or len(value) > limit:
        raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {label}.")
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {label}.")
    return value


def _release_snapshot_path(value: object, *, label: str) -> str:
    path = _release_snapshot_text(value, label=label, limit=_RELEASE_SNAPSHOT_MAX_PATH)
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {label}.")
    return path


def _release_snapshot_ref(value: object, *, label: str) -> str:
    ref = _release_snapshot_text(value, label=label, limit=_RELEASE_SNAPSHOT_MAX_REF)
    if (
        not ref.startswith("refs/")
        or ref.endswith("/")
        or "//" in ref
        or ".." in ref
        or "@{" in ref
        or any(char in ref for char in " ~^:?*[\\")
    ):
        raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {label}.")
    components = ref.split("/")
    if len(components) < 2 or any(
        not component
        or component in {".", ".."}
        or component.startswith(".")
        or component.endswith(".")
        or component.endswith(".lock")
        for component in components
    ):
        raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {label}.")
    return ref


def _validate_release_snapshot_value(value: object) -> None:
    """Validate the bounded, exact journal representation of a Git snapshot."""
    if type(value) is not dict:
        raise ReleaseGitStateError("Release transaction snapshot is not an object.")
    if set(value) != _RELEASE_SNAPSHOT_FIELDS:
        missing = _RELEASE_SNAPSHOT_FIELDS.difference(value)
        extra = set(value).difference(_RELEASE_SNAPSHOT_FIELDS)
        details = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if extra:
            details.append("unexpected " + ", ".join(sorted(extra)))
        raise ReleaseGitStateError(
            "Release transaction snapshot schema is invalid" +
            (": " + "; ".join(details) if details else ".")
        )
    _release_snapshot_path(value["root"], label="root path")
    _release_snapshot_path(value["common_dir"], label="common directory")
    symbolic_head = value["symbolic_head"]
    if symbolic_head is not None:
        symbolic_head = _release_snapshot_ref(symbolic_head, label="symbolic HEAD")
        if not symbolic_head.startswith("refs/heads/"):
            raise ReleaseGitStateError(
                "Release transaction snapshot symbolic HEAD is not a local branch."
            )
    _release_snapshot_ref(value["maintenance_ref"], label="maintenance ref")
    for name in (
        "head_sha",
        "head_tree_sha",
        "maintenance_ref_sha",
        "index_tree_sha",
    ):
        if not isinstance(value[name], str) or _SHA_RE.fullmatch(value[name]) is None:
            raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {name}.")
    for name in ("tracked_diff_sha256", "status_v2_sha256"):
        if (
            not isinstance(value[name], str)
            or re.fullmatch(r"[0-9a-fA-F]{64}", value[name]) is None
        ):
            raise ReleaseGitStateError(f"Release transaction snapshot has an invalid {name}.")
    count = value["untracked_count"]
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or count < 0
        or count > _RELEASE_SNAPSHOT_MAX_UNTRACKED
    ):
        raise ReleaseGitStateError(
            "Release transaction snapshot has an invalid untracked count."
        )


def _release_git_detail(result: subprocess.CompletedProcess) -> str:
    value = result.stderr or result.stdout or b""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace").strip()
    return str(value).strip()


def _release_git_bytes(
    git_cmd: list[str], root: Path, args: list[str], *, label: str
) -> bytes:
    result = subprocess.run(git_cmd + args, cwd=root, capture_output=True)
    if result.returncode != 0:
        detail = _release_git_detail(result)
        raise ReleaseGitStateError(
            f"Could not inspect release checkout {label}"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    output = result.stdout
    if isinstance(output, str):
        output = output.encode("utf-8", errors="surrogateescape")
    if not isinstance(output, bytes):
        raise ReleaseGitStateError(f"Git returned malformed release checkout output for {label}.")
    return output


def _release_git_text(
    git_cmd: list[str], root: Path, args: list[str], *, label: str
) -> str:
    output = _release_git_bytes(git_cmd, root, args, label=label)
    try:
        text = output.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ReleaseGitStateError(f"Git returned malformed text for release checkout {label}.") from exc
    text = text.rstrip("\n")
    if not text or "\n" in text or "\r" in text:
        raise ReleaseGitStateError(f"Git returned malformed release checkout {label}.")
    return text


def _release_git_sha(
    git_cmd: list[str], root: Path, args: list[str], *, label: str
) -> str:
    value = _release_git_text(git_cmd, root, args, label=label).strip()
    if _SHA_RE.fullmatch(value) is None:
        raise ReleaseGitStateError(f"Git returned a malformed SHA for release checkout {label}.")
    return value.lower()


def _release_symbolic_head(git_cmd: list[str], root: Path) -> str | None:
    result = subprocess.run(
        git_cmd + ["symbolic-ref", "-q", "HEAD"],
        cwd=root,
        capture_output=True,
    )
    output = result.stdout or b""
    error = result.stderr or b""
    if isinstance(output, str):
        output = output.encode("utf-8", errors="surrogateescape")
    if isinstance(error, str):
        error = error.encode("utf-8", errors="surrogateescape")
    if result.returncode == 1:
        if output or error:
            raise ReleaseGitStateError("Git returned malformed detached-HEAD identity.")
        return None
    if result.returncode != 0:
        detail = _release_git_detail(result)
        raise ReleaseGitStateError(
            "Could not inspect symbolic HEAD"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    try:
        symbolic = output.decode("utf-8").rstrip("\n")
    except UnicodeDecodeError as exc:
        raise ReleaseGitStateError("Git returned malformed symbolic HEAD output.") from exc
    if (
        not symbolic
        or "\n" in symbolic
        or "\r" in symbolic
        or not symbolic.startswith("refs/heads/")
    ):
        raise ReleaseGitStateError("Git returned an unsupported symbolic HEAD identity.")
    return symbolic


def _capture_release_git_snapshot(
    git_cmd: list[str],
    cwd: Path | str,
    *,
    maintenance_ref: str = _RELEASE_MAINTENANCE_REF,
    expected_root: Path | None = None,
    expected_common_dir: Path | None = None,
) -> ReleaseGitSnapshot:
    """Capture every Git identity relevant to a live release mutation."""
    root = Path(cwd).resolve()
    expected_root = expected_root.resolve() if expected_root is not None else root
    if root != expected_root:
        raise ReleaseGitStateError("Release checkout root changed from its pinned absolute path.")
    shown_root = Path(
        _release_git_text(git_cmd, root, ["rev-parse", "--show-toplevel"], label="repository root")
    ).resolve()
    if shown_root != expected_root:
        raise ReleaseGitStateError(
            f"Release checkout root mismatch: expected {expected_root}, got {shown_root}."
        )
    common_dir = _git_common_dir(git_cmd, root)
    if expected_common_dir is not None and common_dir != expected_common_dir.resolve():
        raise ReleaseGitStateError(
            f"Release Git common directory changed: expected {expected_common_dir}, got {common_dir}."
        )
    symbolic_head = _release_symbolic_head(git_cmd, root)
    head_sha = _release_git_sha(
        git_cmd, root, ["rev-parse", "--verify", "HEAD^{commit}"], label="HEAD"
    )
    head_tree_sha = _release_git_sha(
        git_cmd,
        root,
        ["rev-parse", "--verify", f"{head_sha}^{{tree}}"],
        label="HEAD tree",
    )
    maintenance_ref_sha = _release_git_sha(
        git_cmd,
        root,
        ["rev-parse", "--verify", f"{maintenance_ref}^{{commit}}"],
        label="maintenance ref",
    )
    index_tree_sha = _release_git_sha(
        git_cmd, root, ["write-tree"], label="index tree"
    )
    tracked_diff = _release_git_bytes(
        git_cmd,
        root,
        ["diff", "--no-ext-diff", "--binary", "--"],
        label="tracked worktree diff",
    )
    status_v2 = _release_git_bytes(
        git_cmd,
        root,
        ["status", "--porcelain=v2", "--untracked-files=all", "--no-renames", "-z"],
        label="porcelain-v2 status",
    )
    untracked_count = sum(
        1 for record in status_v2.split(b"\0") if record.startswith(b"? ")
    )
    if untracked_count > _RELEASE_SNAPSHOT_MAX_UNTRACKED:
        raise ReleaseGitStateError(
            "Release checkout has too many untracked paths to snapshot safely."
        )
    return ReleaseGitSnapshot(
        root=expected_root,
        common_dir=common_dir,
        symbolic_head=symbolic_head,
        head_sha=head_sha,
        head_tree_sha=head_tree_sha,
        maintenance_ref=maintenance_ref,
        maintenance_ref_sha=maintenance_ref_sha,
        index_tree_sha=index_tree_sha,
        tracked_diff_sha256=hashlib.sha256(tracked_diff).hexdigest(),
        status_v2_sha256=hashlib.sha256(status_v2).hexdigest(),
        untracked_count=untracked_count,
    )


def _release_snapshot_is_clean(snapshot: ReleaseGitSnapshot) -> bool:
    return (
        snapshot.index_tree_sha == snapshot.head_tree_sha
        and snapshot.tracked_diff_sha256 == _RELEASE_EMPTY_DIGEST
        and snapshot.status_v2_sha256 == _RELEASE_EMPTY_DIGEST
        and snapshot.untracked_count == 0
    )


def _release_snapshot_branch(snapshot: ReleaseGitSnapshot) -> str | None:
    if snapshot.symbolic_head is None:
        return None
    if not snapshot.symbolic_head.startswith("refs/heads/"):
        raise ReleaseGitStateError("Release checkout symbolic HEAD is not a local branch.")
    branch = snapshot.symbolic_head.removeprefix("refs/heads/")
    if not branch or ".." in branch or "\x00" in branch:
        raise ReleaseGitStateError("Release checkout branch identity is malformed.")
    return branch


def _validate_release_git_snapshot(
    git_cmd: list[str], root: Path, expected: ReleaseGitSnapshot, *, label: str
) -> ReleaseGitSnapshot:
    """Fail closed unless the live checkout exactly equals ``expected``."""
    actual = _capture_release_git_snapshot(
        git_cmd,
        root,
        maintenance_ref=expected.maintenance_ref,
        expected_root=expected.root,
        expected_common_dir=expected.common_dir,
    )
    if actual != expected:
        differences = [
            name
            for name in (
                "root",
                "common_dir",
                "symbolic_head",
                "head_sha",
                "head_tree_sha",
                "maintenance_ref_sha",
                "index_tree_sha",
                "tracked_diff_sha256",
                "status_v2_sha256",
                "untracked_count",
            )
            if getattr(actual, name) != getattr(expected, name)
        ]
        raise ReleaseGitStateError(
            f"Release checkout manual interference at {label}: "
            + ", ".join(differences or ["snapshot mismatch"])
        )
    return actual


def _release_validate_no_unmerged_index(git_cmd: list[str], root: Path) -> None:
    """Require a well-formed, fully merged index before user-state validation."""
    result = subprocess.run(
        git_cmd + ["ls-files", "-u", "-z"],
        cwd=root,
        capture_output=True,
    )
    if result.returncode != 0:
        detail = _release_git_detail(result)
        raise ReleaseGitStateError(
            "Could not inspect release checkout unmerged index entries"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    output = result.stdout
    if not isinstance(output, bytes):
        raise ReleaseGitStateError(
            "Git returned malformed release checkout unmerged index output."
        )
    if output:
        raise ReleaseGitStateError("Release checkout has unmerged index entries.")


def _validate_release_restored_snapshot_identity(
    git_cmd: list[str],
    root: Path,
    expected: ReleaseGitSnapshot,
    *,
    label: str,
) -> ReleaseGitSnapshot:
    """Capture live user state and compare only immutable checkout identity."""
    _release_validate_no_unmerged_index(git_cmd, root)
    actual = _capture_release_git_snapshot(
        git_cmd,
        root,
        maintenance_ref=expected.maintenance_ref,
        expected_root=expected.root,
        expected_common_dir=expected.common_dir,
    )
    differences = [
        name
        for name in (
            "root",
            "common_dir",
            "symbolic_head",
            "head_sha",
            "head_tree_sha",
            "maintenance_ref",
            "maintenance_ref_sha",
        )
        if getattr(actual, name) != getattr(expected, name)
    ]
    if differences:
        raise ReleaseGitStateError(
            f"Release checkout identity mismatch at {label}: "
            + ", ".join(differences)
        )
    return actual


def _release_clean_snapshot(
    snapshot: ReleaseGitSnapshot,
    *,
    symbolic_head: str | None,
    head_sha: str,
    maintenance_ref_sha: str,
    head_tree_sha: str | None = None,
) -> ReleaseGitSnapshot:
    tree = head_tree_sha or snapshot.head_tree_sha
    return replace(
        snapshot,
        symbolic_head=symbolic_head,
        head_sha=head_sha,
        head_tree_sha=tree,
        maintenance_ref_sha=maintenance_ref_sha,
        index_tree_sha=tree,
        tracked_diff_sha256=_RELEASE_EMPTY_DIGEST,
        status_v2_sha256=_RELEASE_EMPTY_DIGEST,
        untracked_count=0,
    )


def _release_snapshot_with_journal(
    context: ReleaseUpgradeContext,
    key: str,
    snapshot: ReleaseGitSnapshot,
    *,
    phase: str | None = None,
    **updates,
) -> None:
    context.snapshots[key] = snapshot
    updates[key] = snapshot.to_journal()
    _journal_update(context, phase, **updates)


def _release_context_snapshot(
    context: ReleaseUpgradeContext, key: str
) -> ReleaseGitSnapshot:
    # The durable journal is the binding record.  Do not let a mutable in-memory
    # cache silently replace a malformed or tampered on-disk snapshot during
    # recovery/finalization.
    snapshot = ReleaseGitSnapshot.from_journal(context.journal.get(key))
    expected_root = context.root.resolve()
    expected_common = context.common_dir.resolve()
    if snapshot.root != expected_root or snapshot.common_dir != expected_common:
        raise ReleaseGitStateError(
            f"Release transaction snapshot {key} is bound to a different Git checkout."
        )
    return snapshot


def _release_mark_manual_interference(
    context: ReleaseUpgradeContext, reason: object
) -> None:
    try:
        message = str(reason)
    except BaseException:
        message = "unexpected live Git state"
    message = " ".join(
        " " if (ord(char) < 0x20 or ord(char) == 0x7F) else char
        for char in message
    ).strip()[:512] or "unexpected live Git state"
    context.journal["manual_interference"] = True
    context.journal["interference_reason"] = message
    try:
        _journal_update(
            context,
            "manual-interference",
            manual_interference=True,
            interference_reason=message,
        )
    except BaseException:
        # The in-memory latch still prevents a finalizer in this process from
        # issuing a destructive Git command; the original evidence remains on disk.
        logger.warning("Could not durably mark manual release interference", exc_info=True)


def _release_has_manual_interference(context: ReleaseUpgradeContext) -> bool:
    """Return whether this transaction has latched a live-checkout race."""
    return bool(
        context.journal.get("manual_interference")
        or context.journal.get("phase") == "manual-interference"
    )


def _release_assert_identity(
    snapshot: ReleaseGitSnapshot,
    *,
    symbolic_head: str | None,
    head_sha: str,
    maintenance_ref_sha: str,
    label: str,
    clean: bool = False,
) -> None:
    if (
        snapshot.symbolic_head != symbolic_head
        or snapshot.head_sha != head_sha
        or snapshot.maintenance_ref_sha != maintenance_ref_sha
    ):
        raise ReleaseGitStateError(f"Release checkout identity failed at {label}.")
    if clean and not _release_snapshot_is_clean(snapshot):
        raise ReleaseGitStateError(f"Release checkout is not clean at {label}.")


def _is_official_origin_url(url: str | None) -> bool:
    """Return whether ``url`` is a strict official repository origin."""

    if not isinstance(url, str) or not url.strip():
        return False
    value = url.strip()
    official_paths = {
        "NousResearch/hermes-agent",
        "NousResearch/hermes-agent.git",
    }
    if value.startswith("git@"):
        host, separator, path = value.partition(":")
        if not separator or host.lower() != "git@github.com":
            return False
        if "?" in path or "#" in path:
            return False
        return path in official_paths

    try:
        parsed = urlsplit(value)
        hostname = parsed.hostname
        port = parsed.port
    except ValueError:
        return False
    if hostname is None or hostname.lower() != "github.com":
        return False
    if parsed.username not in (None, "git") or parsed.password is not None:
        return False
    if parsed.query or parsed.fragment:
        return False
    scheme = parsed.scheme.lower()
    if scheme not in {"https", "ssh"}:
        return False
    standard_port = 443 if scheme == "https" else 22
    if port is not None and port != standard_port:
        return False
    if not parsed.path.startswith("/") or parsed.path.startswith("//"):
        return False
    return parsed.path[1:] in official_paths


def _validate_release_tag_name(
    git_cmd: list[str], cwd: Path | str, release_tag: str
) -> None:
    if (
        not isinstance(release_tag, str)
        or not release_tag
        or release_tag.startswith("refs/")
        or "\x00" in release_tag
    ):
        raise RuntimeError(f"Invalid release tag name: {release_tag!r}")
    result = subprocess.run(
        git_cmd + ["check-ref-format", f"refs/tags/{release_tag}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"Invalid release tag name: {release_tag!r}")


def _origin_url_for_release(git_cmd: list[str], cwd: Path | str) -> str | None:
    """Resolve the configured origin through this module's local helper."""

    return _get_origin_url(git_cmd, cwd)


def _resolve_release_target(
    git_cmd: list[str], cwd: Path | str, release_tag: str
) -> ReleaseTarget:
    """Fetch and resolve one exact official tag into a private ref namespace."""

    _validate_release_tag_name(git_cmd, cwd, release_tag)
    origin_url = _origin_url_for_release(git_cmd, cwd)
    if not _is_official_origin_url(origin_url):
        raise RuntimeError(
            "Release upgrade requires the official origin "
            "https://github.com/NousResearch/hermes-agent.git; "
            f"configured origin is {origin_url or '<missing>'}."
        )

    private_ref = f"refs/hermes-upgrade/tags/{release_tag}"
    source_ref = f"refs/tags/{release_tag}"
    fetch = subprocess.run(
        git_cmd
        + ["fetch", "--no-tags", "origin", f"+{source_ref}:{private_ref}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if fetch.returncode != 0:
        detail = (fetch.stderr or fetch.stdout or "").strip()
        raise RuntimeError(
            f"Could not fetch exact release tag {release_tag}: "
            f"{detail.splitlines()[0] if detail else 'git fetch failed'}"
        )

    resolved = subprocess.run(
        git_cmd + ["rev-parse", "--verify", f"{private_ref}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    target_sha = resolved.stdout.strip()
    if resolved.returncode != 0 or not _SHA_RE.fullmatch(target_sha):
        raise RuntimeError(
            f"Fetched release tag {release_tag} did not resolve to an immutable commit."
        )
    return ReleaseTarget(release_tag, target_sha.lower(), private_ref)


def _read_release_base_metadata(cwd: Path | str) -> ReleaseBaseMetadata:
    """Parse the JSON ``local-patches/.release_base`` contract."""

    path = Path(cwd) / _LOCAL_PATCHES_DIRNAME / ".release_base"
    try:
        info = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Missing {path}; cannot establish the base of local maintenance changes."
        ) from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"Refusing non-regular release metadata: {path}")
    try:
        raw = path.read_bytes()
    except OSError as exc:
        raise RuntimeError(f"Invalid release metadata {path}: {exc}") from exc
    return _parse_release_base_metadata(raw, str(path))


def _parse_release_base_metadata(raw: bytes, source: str) -> ReleaseBaseMetadata:
    try:
        payload = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Invalid release metadata {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Invalid release metadata {source}: expected an object")
    tag = payload.get("tag")
    base_sha = payload.get("base_sha")
    target_sha = payload.get("target_sha")
    patch_sha256 = payload.get("patch_sha256")
    patch_bytes = payload.get("patch_bytes")
    format_version = payload.get("format_version", 1)
    integration_mode = payload.get("integration_mode")
    if not isinstance(tag, str) or not tag:
        raise RuntimeError(f"Invalid release metadata {source}: missing tag")
    if not isinstance(base_sha, str) or not _SHA_RE.fullmatch(base_sha):
        raise RuntimeError(f"Invalid release metadata {source}: missing base_sha")
    if target_sha is not None and (
        not isinstance(target_sha, str) or not _SHA_RE.fullmatch(target_sha)
    ):
        raise RuntimeError(f"Invalid release metadata {source}: invalid target_sha")
    if patch_sha256 is not None and (
        not isinstance(patch_sha256, str)
        or not re.fullmatch(r"[0-9a-fA-F]{64}", patch_sha256)
    ):
        raise RuntimeError(f"Invalid release metadata {source}: invalid patch_sha256")
    if patch_bytes is not None and (
        not isinstance(patch_bytes, int) or patch_bytes < 0
    ):
        raise RuntimeError(f"Invalid release metadata {source}: invalid patch_bytes")
    if not isinstance(format_version, int) or isinstance(format_version, bool):
        raise RuntimeError(f"Invalid release metadata {source}: invalid format_version")
    if format_version not in {1, 2}:
        raise RuntimeError(
            f"Invalid release metadata {source}: unsupported format_version {format_version}"
        )
    if integration_mode is None:
        integration_mode = "snapshot-patch" if format_version == 1 else "incremental-merge"
    if integration_mode not in {"snapshot-patch", "incremental-merge"}:
        raise RuntimeError(f"Invalid release metadata {source}: invalid integration_mode")
    if format_version == 1 and integration_mode != "snapshot-patch":
        raise RuntimeError(
            f"Invalid release metadata {source}: format_version 1 requires snapshot-patch"
        )
    if format_version == 2 and integration_mode != "incremental-merge":
        raise RuntimeError(
            f"Invalid release metadata {source}: format_version 2 requires incremental-merge"
        )
    return ReleaseBaseMetadata(
        tag=tag,
        base_sha=base_sha.lower(),
        target_sha=target_sha.lower() if isinstance(target_sha, str) else None,
        patch_sha256=patch_sha256.lower() if isinstance(patch_sha256, str) else None,
        patch_bytes=patch_bytes,
        format_version=format_version,
        integration_mode=integration_mode,
    )


def _git_resolve_commit(git_cmd: list[str], cwd: Path | str, ref: str) -> str | None:
    result = subprocess.run(
        git_cmd + ["rev-parse", "--verify", f"{ref}^{{commit}}"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    value = result.stdout.strip()
    return value.lower() if result.returncode == 0 and _SHA_RE.fullmatch(value) else None


def _validate_release_base_metadata(
    git_cmd: list[str], cwd: Path | str, metadata: ReleaseBaseMetadata, head_sha: str
) -> None:
    if not _SHA_RE.fullmatch(head_sha):
        raise RuntimeError("Current maintenance HEAD is not an immutable commit SHA.")
    if _git_resolve_commit(git_cmd, cwd, metadata.base_sha) != metadata.base_sha:
        raise RuntimeError(
            f"Release metadata base {metadata.base_sha} is missing from the repository."
        )
    ancestry = subprocess.run(
        git_cmd + ["merge-base", "--is-ancestor", metadata.base_sha, head_sha],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if ancestry.returncode != 0:
        raise RuntimeError(
            f"Release metadata base {metadata.tag} ({metadata.base_sha}) is not an "
            "ancestor of the maintenance branch; refusing to guess a payload base."
        )
    known_base = _git_resolve_commit(
        git_cmd, cwd, f"refs/tags/{metadata.tag}"
    ) or _git_resolve_commit(
        git_cmd, cwd, f"refs/hermes-upgrade/tags/{metadata.tag}"
    )
    if known_base is None or known_base != metadata.base_sha:
        raise RuntimeError(
            f"Release metadata tag {metadata.tag} does not resolve consistently to "
            f"{metadata.base_sha}; refusing to generate local payload."
        )
    if metadata.target_sha and _git_resolve_commit(git_cmd, cwd, metadata.target_sha) != metadata.target_sha:
        raise RuntimeError(
            f"Release metadata target {metadata.target_sha} is missing from the repository."
        )


def _validate_incremental_release_target(
    git_cmd: list[str],
    cwd: Path | str,
    *,
    previous_base_sha: str,
    target_sha: str,
) -> None:
    """Require release upgrades to advance the recorded upstream lineage."""

    ancestry = subprocess.run(
        git_cmd + ["merge-base", "--is-ancestor", previous_base_sha, target_sha],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if ancestry.returncode == 0:
        return
    if ancestry.returncode == 1:
        raise RuntimeError(
            f"Release target {target_sha} does not descend from the recorded upstream "
            f"base {previous_base_sha}; refusing a non-linear maintenance merge."
        )
    detail = (ancestry.stderr or ancestry.stdout or "").strip()
    raise RuntimeError(
        "Could not verify release ancestry"
        + (f": {detail.splitlines()[0]}" if detail else ".")
    )


def _maintenance_pathspec() -> str:
    return f":(exclude){_LOCAL_PATCHES_DIRNAME}"


def _reject_gitlink_changes(
    git_cmd: list[str], cwd: Path | str, base_sha: str, head_sha: str
) -> None:
    raw = subprocess.run(
        git_cmd
        + ["diff", "--raw", "-z", base_sha, head_sha, "--", ".", _maintenance_pathspec()],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if raw.returncode != 0:
        detail = raw.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"Could not inspect local maintenance delta{': ' + detail if detail else ''}."
        )
    for record in raw.stdout.split(b"\0"):
        if record and (b" 160000 " in record or record.startswith(b":160000 ")):
            raise RuntimeError(
                "Local maintenance delta contains a Git submodule/gitlink; "
                "recursive submodule replay is not supported."
            )


def _validate_patch_artifact(
    cwd: Path | str, metadata: ReleaseBaseMetadata
) -> None:
    """Validate artifact shape, but never use it as the payload source."""

    path = Path(cwd) / _LOCAL_PATCHES_DIRNAME / "0001-local-maintenance.patch"
    try:
        info = path.lstat()
    except FileNotFoundError:
        logger.warning("Local patch artifact is missing; using Git history instead: %s", path)
        return
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"Refusing non-regular local patch artifact: {path}")
    content = path.read_bytes()
    if metadata.patch_sha256 and hashlib.sha256(content).hexdigest() != metadata.patch_sha256:
        logger.warning("Local patch artifact is stale; using Git history instead: %s", path)
    if metadata.patch_bytes is not None and len(content) != metadata.patch_bytes:
        logger.warning("Local patch artifact size is stale; using Git history instead: %s", path)


def _git_diff_bytes(
    git_cmd: list[str], cwd: Path | str, base_sha: str, head_sha: str
) -> bytes:
    _reject_gitlink_changes(git_cmd, cwd, base_sha, head_sha)
    result = subprocess.run(
        git_cmd
        + [
            "diff",
            "--binary",
            "--full-index",
            "--no-ext-diff",
            "--no-textconv",
            "--find-renames",
            base_sha,
            head_sha,
            "--",
            ".",
            _maintenance_pathspec(),
        ],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            f"Could not generate Git maintenance payload{': ' + detail if detail else ''}."
        )
    return result.stdout


def _generate_runtime_local_payload(
    git_cmd: list[str],
    cwd: Path | str,
    metadata: ReleaseBaseMetadata,
    *,
    head_sha: str | None = None,
) -> bytes:
    """Generate the local payload from committed Git truth, not patch files."""

    root = Path(cwd)
    if head_sha is None:
        head_sha = _git_resolve_commit(git_cmd, root, "HEAD")
    if head_sha is None:
        raise RuntimeError("Could not resolve the maintenance branch HEAD.")
    _validate_release_base_metadata(git_cmd, root, metadata, head_sha)
    _validate_patch_artifact(root, metadata)
    return _git_diff_bytes(git_cmd, root, metadata.base_sha, head_sha)


def _read_release_base_metadata_at_commit(
    git_cmd: list[str], cwd: Path | str, head_sha: str
) -> ReleaseBaseMetadata:
    """Read release metadata from ``head_sha`` without changing the worktree."""

    result = subprocess.run(
        git_cmd + ["show", f"{head_sha}:{_LOCAL_PATCHES_DIRNAME}/.release_base"],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode != 0:
        detail = result.stderr.decode(errors="replace").strip()
        raise RuntimeError(
            "Could not read committed release metadata from maintenance HEAD"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    return _parse_release_base_metadata(
        result.stdout,
        f"{head_sha}:{_LOCAL_PATCHES_DIRNAME}/.release_base",
    )


def _capture_release_checkout_identity(
    git_cmd: list[str], cwd: Path | str
) -> tuple[str | None, str]:
    """Capture symbolic branch (if any) and exact HEAD before mutation."""

    root = Path(cwd)
    head_sha = _git_resolve_commit(git_cmd, root, "HEAD")
    if head_sha is None:
        raise RuntimeError("Could not capture the exact original HEAD SHA.")
    symbolic = subprocess.run(
        git_cmd + ["symbolic-ref", "--quiet", "--short", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    original_branch = symbolic.stdout.strip() if symbolic.returncode == 0 else None
    return original_branch or None, head_sha


def _write_transaction_journal(
    common_dir: Path, payload: dict, *, path: Path | None = None
) -> Path:
    path = Path(path) if path is not None else common_dir / "hermes-upgrade-transaction.json"
    _atomic_write_bytes(
        path,
        (json.dumps(payload, sort_keys=True, indent=2) + "\n").encode(),
        required_parent_fsync=True,
    )
    return path


def _journal_update(
    context: ReleaseUpgradeContext, phase: str | None = None, **updates
) -> None:
    if phase is not None:
        context.journal["phase"] = phase
        # Keep the old key for operators/tools that consumed the first journal.
        context.journal["state"] = phase
    context.journal.update(updates)
    _write_transaction_journal(
        context.common_dir, context.journal, path=context.journal_path
    )


def _find_unfinished_release_transaction(
    git_cmd: list[str], cwd: Path | str
) -> tuple[Path, dict] | None:
    """Return the first durable transaction journal that still requires recovery."""

    common = _git_common_dir(git_cmd, cwd)
    transactions = common / "hermes-upgrade-transactions"
    try:
        transactions_info = transactions.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise RuntimeError(
            "Could not inspect release transaction recovery state."
        ) from None
    if stat.S_ISLNK(transactions_info.st_mode) or not stat.S_ISDIR(
        transactions_info.st_mode
    ):
        raise RuntimeError(
            "Refusing unsafe release transaction recovery state."
        )
    try:
        entries = sorted(transactions.iterdir())
    except FileNotFoundError:
        return None
    except OSError:
        raise RuntimeError(
            "Could not inspect release transaction recovery state."
        ) from None
    try:
        current_transactions_info = transactions.lstat()
    except FileNotFoundError:
        return None
    except OSError:
        raise RuntimeError(
            "Could not inspect release transaction recovery state."
        ) from None
    if (
        stat.S_ISLNK(current_transactions_info.st_mode)
        or not stat.S_ISDIR(current_transactions_info.st_mode)
        or not _release_same_stat(
            current_transactions_info, transactions_info, include_size=False
        )
    ):
        raise RuntimeError(
            "Refusing unsafe release transaction recovery state."
        )

    for entry in entries:
        try:
            info = entry.lstat()
        except FileNotFoundError:
            continue
        except OSError:
            raise RuntimeError(
                "Could not inspect release transaction recovery state."
            ) from None
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(
                f"Unexpected release transaction entry; refusing to continue: {entry}"
            )
        journal_path = entry / _RELEASE_JOURNAL_FILENAME
        try:
            journal = _read_release_transaction_journal(
                entry,
                journal_path=journal_path,
                expected_transactions_dir=transactions,
            )
        except _ReleaseJournalReadError as exc:
            if exc.kind == "missing":
                # Preserve the pre-existing state-machine rule: a transaction
                # child without a journal is not itself proof of unfinished
                # recovery.  Every present journal failure is unsafe evidence.
                continue
            category = {
                "oversize": "oversized",
                "changed": "changed while being read",
                "invalid": "invalid",
                "unsafe": "unsafe",
                "io": "unreadable",
            }.get(exc.kind, "unreadable")
            raise RuntimeError(
                f"Unfinished release transaction journal is {category}: {journal_path}"
            ) from None
        return journal_path, journal
    return None


def _recovery_safe_text(value: object, *, limit: int = 4096) -> str | None:
    """Return journal text that is safe to display, or ``None``.

    Journals survive hard kills and are local input, not trusted program state.
    In particular, do not let a newline, terminal control, or an overlong value
    turn recovery output into a second command or an opaque dump of the journal.
    """

    if not isinstance(value, str) or not value or len(value) > limit:
        return None
    if any(ord(char) < 0x20 or ord(char) == 0x7F for char in value):
        return None
    return value


def _recovery_sha(value: object) -> str | None:
    if not isinstance(value, str) or not _SHA_RE.fullmatch(value):
        return None
    return value.lower()


def _recovery_ref(value: object) -> str | None:
    """Validate a full Git ref without executing Git from the formatter."""

    value = _recovery_safe_text(value)
    if value is None or not value.startswith("refs/"):
        return None
    if value.endswith("/") or "//" in value or ".." in value or "@{" in value:
        return None
    if any(char in value for char in " ~^:?*[\\"):
        return None
    components = value.split("/")
    if len(components) < 2:
        return None
    for component in components:
        if (
            not component
            or component in {".", ".."}
            or component.startswith(".")
            or component.endswith(".")
            or component.endswith(".lock")
        ):
            return None
    return value


def _recovery_branch(value: object) -> str | None:
    """Validate a short branch name before putting it in a Git command."""

    value = _recovery_safe_text(value)
    if value is None or value.startswith("-") or value.startswith("refs/"):
        return None
    return value if _recovery_ref(f"refs/heads/{value}") else None


def _recovery_absolute_path(value: object) -> Path | None:
    value = _recovery_safe_text(value)
    if value is None or not os.path.isabs(value):
        return None
    return Path(os.path.normpath(value))


def _recovery_candidate_root(journal_path: Path) -> Path | None:
    """Derive the sibling candidate root from the journal's Git common dir."""

    path = _recovery_absolute_path(str(journal_path))
    if path is None:
        return None
    # <common>/hermes-upgrade-transactions/<transaction>/journal.json
    common = path.parent.parent.parent
    return common.parent.parent


def _recovery_candidate_path(journal_path: Path, value: object) -> Path | None:
    """Accept only a direct child of the transaction's expected candidate root."""

    candidate = _recovery_absolute_path(value)
    expected_root = _recovery_candidate_root(journal_path)
    if candidate is None or expected_root is None:
        return None
    if candidate.parent != expected_root or candidate == expected_root:
        return None
    return candidate


_RECOVERY_PHASES = frozenset(
    {
        "prepared",
        "journal-created",
        "payload-persisted",
        "payload-integrity-failed",
        "stash-capture-started",
        "stash-capture-verifying",
        "stash-capture-uncertain",
        "stash-captured",
        "stashed",
        "no-stash",
        "stash-pending",
        "backup-failed",
        "backup-created",
        "candidate-create-failed",
        "candidate-created",
        "candidate-validated",
        "promoting",
        "promotion-needs-recovery",
        "promotion-cas-failed",
        "promotion-uncertain",
        "manual-interference",
        "promoted",
        "candidate-cleanup",
        "candidate-cleanup-failed",
        "finalizing",
        "stash-restore-conflict",
        "stash-restore-failed",
        "stash-drop-failed",
        "stash-drop-unverified",
        "finalized",
    }
)


def _recovery_command(*parts: object) -> str:
    return "$ " + " ".join(shlex.quote(str(part)) for part in parts)


def _recovery_stash_list_command(marker: str) -> str:
    # Keep the format literal and quote only the journal-derived marker.
    return (
        "$ git stash list --format='%gd %H %s' | grep -F -- "
        + shlex.quote(marker)
    )


def _format_release_transaction_recovery_guidance(
    journal_path: Path | str,
    journal: dict,
    *,
    repo_root: Path | str,
) -> str:
    """Format immutable, phase-aware recovery guidance without side effects.

    This function deliberately performs no Git or filesystem operations.  Every
    value interpolated into a command is validated locally and shell-quoted;
    malformed journal fields become ``unavailable`` instead of becoming a
    recovery command.  The uncertain-capture branch is intentionally terminal:
    it emits inspection commands only.
    """

    if not isinstance(journal, dict):
        return (
            "Recovery guidance unavailable: the unfinished release journal is "
            "not a JSON object. Preserve the journal and do not run cleanup."
        )

    journal_path = Path(journal_path)
    journal_display = _recovery_safe_text(str(journal_path))
    journal_display = journal_display or "<unavailable>"
    journal_command_path = (
        shlex.quote(journal_display) if journal_display != "<unavailable>" else None
    )
    repo_display = _recovery_safe_text(str(repo_root)) or "<unavailable>"
    phase_raw = journal.get("phase") if "phase" in journal else journal.get("state")
    phase = _recovery_safe_text(phase_raw)
    if phase not in _RECOVERY_PHASES:
        phase = None
    phase_key = phase or ""

    original_branch = _recovery_branch(journal.get("original_branch"))
    original_branch_present = "original_branch" in journal
    original_head_sha = _recovery_sha(journal.get("original_head_sha"))
    original_ref = (
        _recovery_ref(journal.get("original_ref"))
        if "original_ref" in journal
        else None
    )
    if original_branch is not None and "original_ref" in journal:
        if original_ref != f"refs/heads/{original_branch}":
            original_branch = None
    elif original_branch_present and journal.get("original_branch") is None:
        if journal.get("original_ref") not in (None, ""):
            original_head_sha = None
    maintenance_branch = _recovery_branch(journal.get("maintenance_branch"))
    maintenance_old_raw = (
        journal.get("maintenance_old_sha")
        if "maintenance_old_sha" in journal
        else journal.get("old_sha")
    )
    maintenance_old_sha = _recovery_sha(maintenance_old_raw)
    candidate_sha = _recovery_sha(journal.get("candidate_sha"))
    current_sha = _recovery_sha(journal.get("current_sha"))
    current_or_candidate_sha = current_sha or candidate_sha
    target_sha = _recovery_sha(journal.get("target_sha"))

    if "maintenance_ref" in journal:
        maintenance_ref = _recovery_ref(journal.get("maintenance_ref"))
    elif maintenance_branch is not None:
        maintenance_ref = f"refs/heads/{maintenance_branch}"
    else:
        maintenance_ref = None
    if (
        maintenance_ref is not None
        and maintenance_branch is not None
        and maintenance_ref != f"refs/heads/{maintenance_branch}"
    ):
        maintenance_ref = None
    backup_ref = _recovery_ref(journal.get("backup_ref"))
    candidate_branch = _recovery_branch(journal.get("candidate_branch"))
    candidate_path = _recovery_candidate_path(journal_path, journal.get("candidate_path"))
    repo_path = _recovery_absolute_path(str(repo_root))
    if candidate_path is not None and repo_path is not None and candidate_path == repo_path:
        candidate_path = None

    marker = _recovery_safe_text(journal.get("stash_marker"))
    stash_sha = _recovery_sha(journal.get("stash_sha"))
    capture_required = journal.get("stash_capture_required")
    capture_confirmed_raw = journal.get("stash_capture_confirmed")
    capture_uncertain_raw = journal.get("stash_capture_uncertain")
    capture_confirmed = capture_confirmed_raw is True
    capture_uncertain_flag = capture_uncertain_raw is True
    local_state_present = journal.get("local_state_present")
    stash_pending_raw = journal.get("stash_pending")
    stash_pending = stash_pending_raw is True
    stash_apply_attempted = journal.get("stash_apply_attempted") is True
    stash_applied = journal.get("stash_applied") is True

    # Unknown local state is handled like present local state.  The only state
    # that authorizes destructive recovery without a confirmed stash is an
    # explicit, durably recorded ``False``.
    capture_uncertain = (
        capture_uncertain_flag
        or (
            local_state_present is not False
            and (not capture_confirmed or stash_sha is None)
        )
        or (capture_required is True and not capture_confirmed)
        or (stash_pending and (not capture_confirmed or stash_sha is None))
    )
    destructive_recovery_blocked = capture_uncertain

    post_cas_phases = {
        "promoting",
        "promotion-needs-recovery",
        "promotion-uncertain",
        "promoted",
        "candidate-cleanup",
        "candidate-cleanup-failed",
        "finalizing",
        "stash-restore-conflict",
        "stash-restore-failed",
        "stash-drop-failed",
        "stash-drop-unverified",
        "finalized",
    }
    post_cas = phase_key in post_cas_phases
    stash_conflict = stash_pending and stash_apply_attempted and not stash_applied
    cleanup_unresolved = phase_key in {
        "candidate-cleanup-failed",
        "candidate-cleanup",
    } or (
        journal.get("candidate_created") is True
        and journal.get("candidate_cleanup") is not True
        and post_cas
    )

    def field(label: str, value: object, *, quote: bool = False) -> None:
        if value is None:
            lines.append(f"{label}: unavailable (invalid journal value)")
        else:
            rendered = shlex.quote(str(value)) if quote else str(value)
            lines.append(f"{label}: {rendered}")

    lines = [
        "Unfinished release transaction recovery guidance (no commands were executed):",
        f"journal: {journal_display}",
        f"phase: {phase or 'unavailable (invalid journal value)'}",
        f"repo root: {repo_display}",
    ]
    if original_branch_present and journal.get("original_branch") is None:
        lines.append("original checkout: detached HEAD")
    elif original_branch is not None:
        lines.append(f"original checkout: branch {original_branch}")
    else:
        lines.append("original checkout: symbolic branch unavailable (invalid journal value)")
    field("original HEAD SHA", original_head_sha)
    field("maintenance ref", maintenance_ref)
    field("maintenance old SHA", maintenance_old_sha)
    field("target SHA", target_sha)
    field("current/candidate SHA", current_or_candidate_sha)
    field("candidate SHA", candidate_sha)
    field("backup ref", backup_ref)
    field("candidate path", candidate_path, quote=True)
    field("candidate branch", candidate_branch)
    field("stash marker", marker, quote=True)
    field(
        "local state present",
        local_state_present if isinstance(local_state_present, bool) else None,
    )
    field(
        "stash capture required",
        capture_required if isinstance(capture_required, bool) else None,
    )
    field(
        "stash capture confirmed",
        capture_confirmed_raw if isinstance(capture_confirmed_raw, bool) else None,
    )
    field(
        "stash capture uncertain",
        capture_uncertain_raw if isinstance(capture_uncertain_raw, bool) else None,
    )
    field("stash pending", stash_pending_raw if isinstance(stash_pending_raw, bool) else None)
    field("stash SHA", stash_sha)

    known_objects: list[str] = []
    for value in (original_head_sha, maintenance_old_sha, target_sha, candidate_sha, backup_ref):
        if value is not None and value not in known_objects:
            known_objects.append(value)

    def add_inspection_commands(*, include_stash: bool) -> None:
        lines.append("Nondestructive inspection commands:")
        lines.append(_recovery_command("git", "status", "--short"))
        if marker is not None:
            lines.append(_recovery_stash_list_command(marker))
        else:
            lines.append("Stash-list marker command unavailable (invalid journal value).")
        objects = list(known_objects)
        if include_stash and stash_sha is not None:
            objects.append(stash_sha)
        seen: set[str] = set()
        for value in objects:
            if value in seen:
                continue
            seen.add(value)
            lines.append(_recovery_command("git", "show", "--stat", value))

    def add_original_checkout() -> bool:
        if original_branch is not None:
            lines.append(_recovery_command("git", "checkout", "--force", original_branch))
            return True
        if original_branch_present and journal.get("original_branch") is None and original_head_sha:
            lines.append(_recovery_command("git", "checkout", "--detach", original_head_sha))
            return True
        lines.append("Original checkout command unavailable (invalid journal identity).")
        return False

    def add_stash_apply() -> bool:
        if not stash_pending:
            return False
        if not capture_confirmed or capture_uncertain_flag or stash_sha is None:
            lines.append("Immutable stash apply command unavailable; do not use a movable stash selector.")
            return False
        lines.append(_recovery_command("git", "stash", "apply", "--index", stash_sha))
        return True

    if capture_uncertain:
        lines.extend(
            [
                "SAFETY STOP: stash capture is uncertain or has no verified immutable SHA while local state may be present.",
                "DO NOT run destructive checkout/reset/clean/stash apply/drop cleanup commands.",
                "Preserve the exact journal above and do not archive, rename, or delete it yet.",
            ]
        )
        add_inspection_commands(include_stash=False)
        lines.append("Only the inspection commands above are authorized until capture is independently verified.")
        return "\n".join(lines)

    if phase is None:
        lines.extend(
            [
                "SAFETY STOP: transaction phase is unavailable or invalid; no recovery mutators are authorized.",
                "Preserve the exact journal above and use inspection output only until a trusted phase is established.",
            ]
        )
        add_inspection_commands(include_stash=capture_confirmed and stash_sha is not None)
        return "\n".join(lines)

    if post_cas:
        lines.append("Recovery class: post-CAS/promotion state; verify identities before every write.")
        add_inspection_commands(include_stash=True)
        if destructive_recovery_blocked:
            lines.append("Destructive rollback and checkout/reset commands are withheld until stash capture is confirmed.")
        elif maintenance_ref and maintenance_old_sha and candidate_sha:
            lines.append(
                "1. Only if the maintenance ref currently resolves to the candidate SHA, run this exact CAS rollback:"
            )
            lines.append(_recovery_command("git", "update-ref", maintenance_ref, maintenance_old_sha, candidate_sha))
            if maintenance_branch:
                lines.append(
                    "2. After the CAS succeeds, and only after confirmed stash capture "
                    "(or no local state was present):"
                )
                lines.append(_recovery_command("git", "checkout", "--force", maintenance_branch))
                lines.append(_recovery_command("git", "reset", "--hard", maintenance_old_sha))
            else:
                lines.append("Maintenance checkout/reset commands unavailable (invalid branch value).")
            lines.append("3. Restore the original checkout only after verifying the current ref identity:")
            original_restored = add_original_checkout()
            if original_restored and original_branch and original_head_sha:
                lines.append(_recovery_command("git", "reset", "--hard", original_head_sha))
            if stash_pending:
                lines.append("4. After the checkout identity is verified, apply the immutable stash:")
                add_stash_apply()
        else:
            lines.append("CAS rollback command unavailable: maintenance ref, old SHA, or candidate SHA is invalid.")
    else:
        lines.append("Recovery class: pre-promotion; verify the original checkout identity before applying user state.")
        add_inspection_commands(include_stash=True)
        if original_branch or (original_branch_present and journal.get("original_branch") is None and original_head_sha):
            lines.append("After verifying the original checkout identity, restore it:")
            add_original_checkout()
        if stash_pending:
            lines.append("Apply user state only after verifying the original checkout identity:")
            add_stash_apply()

    if stash_conflict and not capture_uncertain:
        lines.append("Stash restore remains unresolved; do not clean any stash reflog entry yet.")
        if capture_confirmed and stash_sha:
            lines.append("verify files and index before any stash reflog cleanup; never use stash@{N}:")
            if not any(
                line == _recovery_command("git", "stash", "apply", "--index", stash_sha)
                for line in lines
            ):
                lines.append(_recovery_command("git", "stash", "apply", "--index", stash_sha))
        else:
            lines.append("Immutable stash apply command unavailable until a verified SHA exists.")

    if cleanup_unresolved:
        lines.append("Candidate cleanup is unresolved; inspect it only after live checkout and stash recovery verification:")
        lines.append(_recovery_command("git", "worktree", "list", "--porcelain"))
        if candidate_path is not None:
            lines.append(_recovery_command("git", "-C", candidate_path, "status", "--short"))
        else:
            lines.append("Candidate status command unavailable (invalid candidate path).")
        if candidate_path is not None and candidate_branch is not None:
            lines.append("OPTIONAL final cleanup, only after live checkout and user-file/stash recovery are verified:")
            lines.append(_recovery_command("git", "worktree", "remove", "--force", "--", candidate_path))
            lines.append(_recovery_command("git", "branch", "-D", "--", candidate_branch))
        else:
            lines.append("Optional candidate removal commands withheld because a candidate path or branch is invalid.")

    if journal_command_path is not None:
        resolved_journal = shlex.quote(f"{journal_path}.resolved")
        lines.extend(
            [
                "Final journal acknowledgment: only after live checkout and user files are verified, archive/rename the journal; never auto-delete it:",
                f"$ mv -- {journal_command_path} {resolved_journal}",
                "On platforms without POSIX mv, use the platform's equivalent rename to the same .resolved path after verification.",
            ]
        )
    else:
        lines.append("Final journal acknowledgment command unavailable; preserve the journal and do not delete it.")
    return "\n".join(lines)


def _reject_unfinished_release_transaction(
    git_cmd: list[str], cwd: Path | str
) -> None:
    found = _find_unfinished_release_transaction(git_cmd, cwd)
    if found is None:
        return
    journal_path, journal = found
    guidance = _format_release_transaction_recovery_guidance(
        journal_path, journal, repo_root=Path(cwd).absolute()
    )
    raise RuntimeError(
        "An unfinished release transaction is already recorded at "
        f"{journal_path}; recover it before starting another release upgrade.\n\n"
        f"{guidance}"
    )


def _create_release_upgrade_context(
    git_cmd: list[str],
    cwd: Path | str,
    *,
    original_branch: str | None,
    original_head_sha: str,
    maintenance_old_sha: str,
    release_tag: str,
    base_sha: str,
    target_sha: str,
    payload: bytes,
    stash_marker: str | None = None,
) -> ReleaseUpgradeContext:
    """Create and durably seed a release transaction before live Git moves."""

    root = Path(cwd)
    common = _git_common_dir(git_cmd, root)
    transactions = common / "hermes-upgrade-transactions"
    try:
        transactions_info = transactions.lstat()
    except FileNotFoundError:
        transactions.mkdir(exist_ok=False)
        # Persist the newly-created transactions root before creating a child
        # that may contain the first discoverable recovery journal.
        _fsync_directory(common, required=True)
    else:
        if stat.S_ISLNK(transactions_info.st_mode) or not stat.S_ISDIR(
            transactions_info.st_mode
        ):
            raise RuntimeError(
                f"Refusing non-directory release transaction root: {transactions}"
            )

    transaction_id = uuid.uuid4().hex
    transaction_dir = transactions / transaction_id
    transaction_dir.mkdir(exist_ok=False)
    # The child directory entry must be durable before its journal/payload
    # can become discoverable or any live Git state can be changed.
    _fsync_directory(transactions, required=True)
    payload_path = transaction_dir / "runtime-local-maintenance.patch"
    backup_ref = f"refs/hermes-upgrade/backups/{transaction_id}"
    candidate_branch = f"hermes-upgrade-candidate/{transaction_id}"
    candidate_path = common.parent.parent / f"hermes-upgrade-candidate-{transaction_id}"
    journal_path = transaction_dir / "journal.json"
    marker = stash_marker or f"hermes-update-autostash-{os.getpid()}-{transaction_id}"
    journal = {
        "version": 2,
        "transaction_id": transaction_id,
        "phase": "prepared",
        "state": "prepared",
        "original_branch": original_branch,
        "original_ref": f"refs/heads/{original_branch}" if original_branch else None,
        "original_head_sha": original_head_sha,
        "maintenance_branch": "hermes-release",
        "maintenance_old_sha": maintenance_old_sha,
        "old_sha": maintenance_old_sha,
        "release_tag": release_tag,
        "base_sha": base_sha,
        "target_sha": target_sha,
        "backup_ref": backup_ref,
        "backup_created": False,
        "candidate_branch": candidate_branch,
        "candidate_path": str(candidate_path),
        "candidate_sha": None,
        "candidate_cleanup": False,
        "payload_path": str(payload_path),
        "payload_sha256": hashlib.sha256(payload).hexdigest(),
        "payload_bytes": len(payload),
        "stash_marker": marker,
        "stash_sha": None,
        # These fields are deliberately tri-state/explicit: a missing or
        # unconfirmed capture must never be treated as an empty worktree.
        "local_state_present": None,
        "stash_capture_required": None,
        "stash_capture_confirmed": False,
        "stash_capture_uncertain": False,
        "stash_pending": False,
        "stash_apply_attempted": False,
        "stash_applied": False,
        "checkout_restored": False,
        "final_state_verified": False,
    }
    context = ReleaseUpgradeContext(
        root=root,
        common_dir=common,
        transaction_dir=transaction_dir,
        journal_path=journal_path,
        journal=journal,
    )
    # The journal exists before payload capture; the payload itself is then
    # written as a regular fsynced file before any stash/checkout/reset.
    _write_transaction_journal(common, journal, path=journal_path)
    _atomic_write_bytes(
        payload_path,
        payload,
        required_parent_fsync=True,
    )
    info = payload_path.lstat()
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RuntimeError(f"Durable release payload is not a regular file: {payload_path}")
    _journal_update(context, "payload-persisted")
    return context


def _candidate_artifact_directory(candidate: Path) -> Path:
    directory = candidate / _LOCAL_PATCHES_DIRNAME
    try:
        info = directory.lstat()
    except FileNotFoundError:
        directory.mkdir(parents=True)
        return directory
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
        raise RuntimeError(f"Refusing non-directory local-patches path: {directory}")
    allowed = {"0001-local-maintenance.patch", "README.md", ".release_base"}
    for entry in directory.iterdir():
        try:
            entry_info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(entry_info.st_mode):
            raise RuntimeError(f"Refusing symlink in local-patches: {entry}")
        if entry.name in allowed:
            if not stat.S_ISREG(entry_info.st_mode):
                raise RuntimeError(f"Refusing non-regular local-patches artifact: {entry}")
        elif entry.suffix == ".patch" and stat.S_ISREG(entry_info.st_mode):
            entry.unlink()
        else:
            raise RuntimeError(f"Unexpected local-patches entry in candidate: {entry.name}")
    return directory


def _commit_candidate_changes(git_cmd: list[str], candidate: Path, message: str) -> None:
    cached = subprocess.run(
        git_cmd + ["diff", "--cached", "--quiet"],
        cwd=candidate,
        capture_output=True,
    )
    if cached.returncode == 0:
        return
    if cached.returncode != 1:
        raise RuntimeError("Could not inspect candidate index before commit.")
    commit = subprocess.run(
        git_cmd + ["commit", "--no-gpg-sign", "-m", message],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if commit.returncode != 0:
        detail = (commit.stderr or commit.stdout or "").strip()
        raise RuntimeError(
            f"Could not commit isolated upgrade candidate: "
            f"{detail.splitlines()[0] if detail else 'git commit failed'}"
        )


def _apply_payload_to_candidate(
    git_cmd: list[str], candidate: Path, payload: bytes, *, label: str
) -> None:
    if not payload:
        return
    fd, tmp_name = tempfile.mkstemp(prefix="hermes-upgrade-payload-", suffix=".patch")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        applied = subprocess.run(
            git_cmd
            + ["apply", "--3way", "--index", "--whitespace=nowarn", str(tmp_path)],
            cwd=candidate,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        unmerged = subprocess.run(
            git_cmd + ["ls-files", "--unmerged"],
            cwd=candidate,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        unmerged_output = (unmerged.stdout or "").strip()
        conflict_paths: list[str] = []
        for line in unmerged_output.splitlines():
            # ``git ls-files --unmerged`` emits one record per index stage,
            # with the path after a tab. Keep a bounded set of paths rather
            # than exposing the repeated stage records or an unbounded index.
            if "\t" in line:
                path = line.split("\t", 1)[1].strip()
            else:
                fields = line.split(maxsplit=3)
                path = fields[3].strip() if len(fields) == 4 else line.strip()
            if path and path not in conflict_paths:
                conflict_paths.append(path)

        if applied.returncode != 0 or unmerged_output or unmerged.returncode != 0:
            detail_parts: list[str] = []
            if conflict_paths:
                shown_paths = conflict_paths[:20]
                suffix = (
                    f" (+{len(conflict_paths) - len(shown_paths)} more)"
                    if len(conflict_paths) > len(shown_paths)
                    else ""
                )
                detail_parts.append(
                    "unmerged index path(s): " + ", ".join(shown_paths) + suffix
                )
            elif unmerged.returncode != 0:
                inspect_detail = (unmerged.stderr or unmerged.stdout or "").strip()
                detail_parts.append(
                    "could not inspect the candidate's unmerged index"
                    + (f": {inspect_detail[:2000]}" if inspect_detail else "")
                )

            if applied.returncode != 0:
                apply_output = "\n".join(
                    part.strip()
                    for part in (applied.stderr or "", applied.stdout or "")
                    if part and part.strip()
                )
                detail_parts.append(
                    f"git apply exited {applied.returncode}"
                    + (f": {apply_output[:4000]}" if apply_output else "")
                )

            detail = "; ".join(detail_parts) or "git apply --3way failed"
            raise RuntimeError(
                f"Release payload {label} conflicts in isolated candidate: "
                f"{detail}"
            )
    finally:
        try:
            tmp_path.unlink()
        except FileNotFoundError:
            pass


def _merge_release_into_candidate(
    git_cmd: list[str],
    candidate: Path,
    *,
    maintenance_sha: str,
    target_sha: str,
    release_tag: str,
) -> str:
    """Incrementally merge one upstream release into an isolated candidate."""

    merged = subprocess.run(
        git_cmd
        + [
            "-c",
            "rerere.enabled=true",
            "-c",
            "rerere.autoupdate=true",
            "merge",
            "--no-ff",
            "--no-commit",
            target_sha,
        ],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    unmerged = subprocess.run(
        git_cmd + ["ls-files", "--unmerged"],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    conflict_paths: list[str] = []
    for line in (unmerged.stdout or "").splitlines():
        path = line.split("\t", 1)[1].strip() if "\t" in line else ""
        if path and path not in conflict_paths:
            conflict_paths.append(path)

    merge_head = _git_resolve_commit(git_cmd, candidate, "MERGE_HEAD")
    if (
        unmerged.returncode != 0
        or conflict_paths
        or merge_head != target_sha
        or (merged.returncode != 0 and merge_head is None)
    ):
        detail_parts: list[str] = []
        if conflict_paths:
            shown = conflict_paths[:20]
            suffix = (
                f" (+{len(conflict_paths) - len(shown)} more)"
                if len(conflict_paths) > len(shown)
                else ""
            )
            detail_parts.append("unmerged path(s): " + ", ".join(shown) + suffix)
        if unmerged.returncode != 0:
            detail_parts.append("could not inspect the candidate's unmerged index")
        merge_output = "\n".join(
            part.strip()
            for part in (merged.stderr or "", merged.stdout or "")
            if part and part.strip()
        )
        if merge_output:
            detail_parts.append(merge_output[:4000])
        detail = "; ".join(detail_parts) or "incremental git merge failed"
        raise RuntimeError(
            f"Release {release_tag} conflicts in isolated candidate: {detail}"
        )

    _validate_candidate_gitlinks(git_cmd, candidate)
    commit = subprocess.run(
        git_cmd
        + [
            "commit",
            "--no-gpg-sign",
            "-m",
            f"local: merge upstream release {release_tag}",
        ],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if commit.returncode != 0:
        detail = (commit.stderr or commit.stdout or "").strip()
        raise RuntimeError(
            f"Could not commit incremental release merge: "
            f"{detail.splitlines()[0] if detail else 'git commit failed'}"
        )

    candidate_head = _git_resolve_commit(git_cmd, candidate, "HEAD")
    parents = subprocess.run(
        git_cmd + ["rev-list", "--parents", "-n", "1", "HEAD"],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    fields = (parents.stdout or "").strip().lower().split()
    expected = [candidate_head, maintenance_sha, target_sha]
    if parents.returncode != 0 or fields != expected:
        raise RuntimeError(
            "Incremental release candidate did not preserve the expected maintenance "
            "and upstream parents."
        )
    assert candidate_head is not None
    return candidate_head


def _validate_candidate_gitlinks(git_cmd: list[str], candidate: Path) -> None:
    """Reject submodule-like index entries before a candidate is applied."""

    staged = subprocess.run(
        git_cmd + ["ls-files", "--stage", "-z"],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if staged.returncode != 0:
        detail = (staged.stderr or staged.stdout or "").strip()
        raise RuntimeError(
            "Could not inspect candidate index for gitlinks"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    for record in staged.stdout.split("\x00"):
        if not record:
            continue
        metadata, separator, path = record.partition("\t")
        if not separator:
            raise RuntimeError("Candidate index contained an invalid staged entry.")
        mode = metadata.split(maxsplit=1)[0]
        if mode == "160000":
            raise RuntimeError(
                f"Candidate validation rejected gitlink/submodule entry {path!r}; "
                "release candidates must contain ordinary files."
            )


def _validate_candidate_self_hosting(candidate: Path) -> None:
    """Run the production release-candidate self-hosting checks."""

    syntax_ok, failing_path, syntax_error = _validate_critical_files_syntax(candidate)
    if not syntax_ok:
        raise RuntimeError(
            f"Candidate validation failed at {failing_path}: "
            f"{syntax_error or 'syntax error'}"
        )
    imports_ok, failing_module, import_error = _validate_critical_modules_import(candidate)
    if not imports_ok:
        raise RuntimeError(
            f"Candidate import validation failed at {failing_module}: "
            f"{import_error or 'import error'}"
        )


def _validate_candidate_tree(
    git_cmd: list[str],
    candidate: Path,
    *,
    candidate_validator=None,
) -> str:
    _validate_candidate_gitlinks(git_cmd, candidate)
    status = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    unmerged = subprocess.run(
        git_cmd + ["ls-files", "--unmerged"],
        cwd=candidate,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if status.returncode != 0 or status.stdout.strip() or unmerged.stdout.strip():
        raise RuntimeError("Candidate worktree is not clean after replay.")
    # This private seam is used only by real-Git tests whose temporary
    # repositories intentionally are not Hermes-shaped. The production path
    # never supplies it and therefore always runs strict self-hosting checks.
    if candidate_validator is None:
        _validate_candidate_self_hosting(candidate)
    else:
        candidate_validator(git_cmd, candidate)
    resolved = _git_resolve_commit(git_cmd, candidate, "HEAD")
    if resolved is None:
        raise RuntimeError("Candidate HEAD did not resolve to an immutable commit.")
    return resolved


def _upgrade_release_transaction(
    git_cmd: list[str],
    cwd: Path | str,
    release_tag: str,
    target_sha: str,
    *,
    transaction_context: ReleaseUpgradeContext | None = None,
    candidate_validator=None,
) -> ReleaseUpgradeResult:
    """Merge an upstream release in isolation, then CAS-promote the candidate."""

    root = Path(cwd).resolve()
    _validate_release_tag_name(git_cmd, root, release_tag)
    if not _SHA_RE.fullmatch(target_sha):
        raise RuntimeError("Release target is not an immutable commit SHA.")
    target_sha = target_sha.lower()
    exact_target = _git_resolve_commit(
        git_cmd, root, f"refs/tags/{release_tag}"
    ) or _git_resolve_commit(git_cmd, root, f"refs/hermes-upgrade/tags/{release_tag}")
    if exact_target != target_sha:
        raise RuntimeError(
            f"Release target {release_tag} is not the exact fetched tag commit; refusing to guess."
        )

    branch = "hermes-release"
    maintenance_ref = f"refs/heads/{branch}"
    current_snapshot = _capture_release_git_snapshot(
        git_cmd, root, maintenance_ref=maintenance_ref
    )
    if (
        transaction_context is not None
        and "post_stash_snapshot" in transaction_context.journal
    ):
        try:
            current_snapshot = _validate_release_git_snapshot(
                git_cmd,
                root,
                _release_context_snapshot(
                    transaction_context,
                    "maintenance_snapshot"
                    if "maintenance_snapshot" in transaction_context.journal
                    else "post_stash_snapshot",
                ),
                label="release transaction post-stash entry",
            )
        except Exception as exc:
            _release_mark_manual_interference(transaction_context, exc)
            raise
    old_sha = current_snapshot.maintenance_ref_sha
    if current_snapshot.symbolic_head != maintenance_ref or current_snapshot.head_sha != old_sha:
        message = "Release transaction requires the explicit hermes-release branch at its recorded HEAD."
        if transaction_context is not None:
            _release_mark_manual_interference(transaction_context, message)
        raise ReleaseGitStateError(message)
    if not _release_snapshot_is_clean(current_snapshot):
        message = "Live maintenance worktree must be clean after user changes are stashed."
        if transaction_context is not None:
            _release_mark_manual_interference(transaction_context, message)
        raise ReleaseGitStateError(message)

    context = transaction_context
    if context is None:
        metadata = _read_release_base_metadata(root)
        _validate_incremental_release_target(
            git_cmd,
            root,
            previous_base_sha=metadata.base_sha,
            target_sha=target_sha,
        )
        payload = _generate_runtime_local_payload(git_cmd, root, metadata, head_sha=old_sha)
        context = _create_release_upgrade_context(
            git_cmd,
            root,
            original_branch=branch,
            original_head_sha=current_snapshot.head_sha,
            maintenance_old_sha=old_sha,
            release_tag=release_tag,
            base_sha=metadata.base_sha,
            target_sha=target_sha,
            payload=payload,
        )
        _release_snapshot_with_journal(context, "original_snapshot", current_snapshot)
        _release_snapshot_with_journal(context, "original_clean_snapshot", current_snapshot)
    journal = context.journal
    recorded_old_sha = journal.get("maintenance_old_sha")
    if recorded_old_sha != old_sha:
        message = "Live maintenance ref changed before promotion; candidate and backup were preserved."
        _release_mark_manual_interference(context, message)
        raise ReleaseGitStateError(message)
    recorded_base_sha = journal.get("base_sha")
    if not isinstance(recorded_base_sha, str) or not _SHA_RE.fullmatch(recorded_base_sha):
        raise RuntimeError("Release transaction has no valid recorded upstream base.")
    _validate_incremental_release_target(
        git_cmd,
        root,
        previous_base_sha=recorded_base_sha.lower(),
        target_sha=target_sha,
    )
    if "pre_cas_snapshot" in journal:
        try:
            current_snapshot = _validate_release_git_snapshot(
                git_cmd,
                root,
                _release_context_snapshot(context, "pre_cas_snapshot"),
                label="release transaction entry",
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
    else:
        _release_snapshot_with_journal(context, "pre_cas_snapshot", current_snapshot)
    payload_path = Path(journal["payload_path"])
    payload = payload_path.read_bytes()
    if (
        len(payload) != journal.get("payload_bytes")
        or hashlib.sha256(payload).hexdigest() != journal.get("payload_sha256")
    ):
        _journal_update(context, "payload-integrity-failed")
        raise RuntimeError("Durable release payload changed before candidate replay.")

    backup_ref = journal["backup_ref"]
    candidate_branch = journal["candidate_branch"]
    candidate_path = Path(journal["candidate_path"])
    backup = subprocess.run(
        git_cmd + ["update-ref", backup_ref, old_sha],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if backup.returncode != 0:
        _journal_update(context, "backup-failed")
        raise RuntimeError("Could not create durable pre-promotion backup ref.")
    _journal_update(context, "backup-created", backup_created=True)

    try:
        worktree = subprocess.run(
            git_cmd + ["worktree", "add", "-b", candidate_branch, str(candidate_path), old_sha],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if worktree.returncode != 0:
            _journal_update(context, "candidate-create-failed")
            detail = (worktree.stderr or worktree.stdout or "").strip()
            raise RuntimeError(
                "Could not create isolated upgrade candidate: "
                f"{detail.splitlines()[0] if detail else 'git worktree add failed'}"
            )
        _journal_update(context, "candidate-created", candidate_created=True)
        _validate_candidate_gitlinks(git_cmd, candidate_path)
        candidate_head = _merge_release_into_candidate(
            git_cmd,
            candidate_path,
            maintenance_sha=old_sha,
            target_sha=target_sha,
            release_tag=release_tag,
        )
        artifact_dir = _candidate_artifact_directory(candidate_path)
        artifact_payload = _git_diff_bytes(git_cmd, candidate_path, target_sha, candidate_head)
        artifact_sha = hashlib.sha256(artifact_payload).hexdigest()
        artifact_metadata = {
            "format_version": 2,
            "integration_mode": "incremental-merge",
            "tag": release_tag,
            "base_sha": target_sha,
            "target_sha": target_sha,
            "patch_sha256": artifact_sha,
            "patch_bytes": len(artifact_payload),
        }
        _atomic_write_bytes(artifact_dir / "0001-local-maintenance.patch", artifact_payload)
        _atomic_write_bytes(artifact_dir / "README.md", _LOCAL_PATCHES_README.encode())
        _atomic_write_bytes(
            artifact_dir / ".release_base",
            (json.dumps(artifact_metadata, sort_keys=True, indent=2) + "\n").encode(),
        )
        stage = subprocess.run(
            git_cmd
            + [
                "add",
                "--",
                f"{_LOCAL_PATCHES_DIRNAME}/0001-local-maintenance.patch",
                f"{_LOCAL_PATCHES_DIRNAME}/README.md",
                f"{_LOCAL_PATCHES_DIRNAME}/.release_base",
            ],
            cwd=candidate_path,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if stage.returncode != 0:
            raise RuntimeError("Could not stage expected local-patches artifacts in candidate.")
        _commit_candidate_changes(
            git_cmd, candidate_path, f"local: refresh maintenance artifacts for {release_tag}"
        )
        candidate_sha = _validate_candidate_tree(
            git_cmd, candidate_path, candidate_validator=candidate_validator
        )
        _journal_update(context, "candidate-validated", candidate_sha=candidate_sha)

        pre_cas_snapshot = _release_context_snapshot(context, "pre_cas_snapshot")
        try:
            _validate_release_git_snapshot(
                git_cmd, root, pre_cas_snapshot, label="promotion CAS precondition"
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
        _journal_update(context, "promoting")
        promote = subprocess.run(
            git_cmd + ["update-ref", f"refs/heads/{branch}", candidate_sha, old_sha],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if promote.returncode != 0:
            try:
                _validate_release_git_snapshot(
                    git_cmd, root, pre_cas_snapshot, label="failed promotion CAS"
                )
            except Exception as exc:
                _release_mark_manual_interference(context, exc)
                raise
            _journal_update(context, "promotion-cas-failed")
            raise RuntimeError(
                "Live maintenance branch changed before promotion; candidate and backup were preserved."
            )

        candidate_tree_sha = _release_git_sha(
            git_cmd,
            root,
            ["rev-parse", "--verify", f"{candidate_sha}^{{tree}}"],
            label="candidate tree",
        )
        post_cas = _capture_release_git_snapshot(
            git_cmd, root, maintenance_ref=maintenance_ref
        )
        try:
            _release_assert_identity(
                post_cas,
                symbolic_head=maintenance_ref,
                head_sha=candidate_sha,
                maintenance_ref_sha=candidate_sha,
                label="promotion CAS postcondition",
            )
            if (
                post_cas.index_tree_sha != pre_cas_snapshot.index_tree_sha
                or post_cas.tracked_diff_sha256 != pre_cas_snapshot.tracked_diff_sha256
                or post_cas.untracked_count != pre_cas_snapshot.untracked_count
                or post_cas.head_tree_sha != candidate_tree_sha
            ):
                raise ReleaseGitStateError(
                    "Live checkout differs from the expected temporary ref/worktree skew."
                )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
        _release_snapshot_with_journal(context, "post_cas_snapshot", post_cas)

        candidate_expected = _release_clean_snapshot(
            post_cas,
            symbolic_head=maintenance_ref,
            head_sha=candidate_sha,
            maintenance_ref_sha=candidate_sha,
            head_tree_sha=candidate_tree_sha,
        )
        try:
            _validate_release_git_snapshot(
                git_cmd, root, post_cas, label="live synchronization precondition"
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
        reset = subprocess.run(
            git_cmd + ["reset", "--hard", candidate_sha],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if reset.returncode != 0:
            try:
                current_after_reset_failure = _capture_release_git_snapshot(
                    git_cmd, root, maintenance_ref=maintenance_ref
                )
                if current_after_reset_failure != post_cas:
                    raise ReleaseGitStateError(
                        "Live checkout changed while candidate synchronization failed."
                    )
            except Exception as exc:
                _release_mark_manual_interference(context, exc)
                raise
            _journal_update(context, "promotion-needs-recovery")
            raise RuntimeError(
                f"Promotion did not verify. Recover with backup ref {backup_ref} "
                f"and candidate {candidate_path}. Journal: {context.journal_path}. "
                f"If refs/heads/{branch} still resolves to {candidate_sha}, use the "
                f"pinned compare-and-swap rollback: git update-ref refs/heads/{branch} "
                f"{old_sha} {candidate_sha}. Immutable stash: "
                f"{journal.get('stash_sha') or '<none>'} "
                f"(stash_pending={journal.get('stash_pending')})."
            )
        try:
            _validate_release_git_snapshot(
                git_cmd, root, candidate_expected, label="live synchronization postcondition"
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise RuntimeError(
                f"Promotion did not verify. Recover with backup ref {backup_ref} "
                f"and candidate {candidate_path}. Journal: {context.journal_path}."
            ) from exc
        _release_snapshot_with_journal(
            context,
            "candidate_snapshot",
            candidate_expected,
            phase="promoted",
            candidate_sha=candidate_sha,
        )
        # The candidate is isolated transaction evidence, not live user state.
        # Once promotion and the live synchronization postcondition have both
        # been verified, clean it at the end of this transaction so callers
        # that defer outer finalization do not strand a successful candidate.
        # A failed cleanup remains recoverable through the outer finalizer.
        if journal.get("candidate_created") and not journal.get("candidate_cleanup"):
            try:
                candidate_cleaned = _release_cleanup_candidate(git_cmd, root, context)
            except (KeyboardInterrupt, SystemExit):
                raise
            except BaseException as exc:
                logger.warning(
                    "Release candidate cleanup could not be completed; recovery journal kept at %s: %s",
                    context.journal_path,
                    exc,
                )
                candidate_cleaned = False
            if candidate_cleaned:
                _journal_update(context, "candidate-cleanup", candidate_cleanup=True)
            else:
                _journal_update(context, "candidate-cleanup-failed", candidate_cleanup=False)
    except BaseException:
        # Candidate and journal are recovery evidence across every exception,
        # including KeyboardInterrupt/SystemExit.  The outer finalizer decides
        # whether it is safe to return user state or must leave guidance.
        raise

    return ReleaseUpgradeResult(
        old_sha,
        target_sha,
        candidate_sha,
        backup_ref,
        candidate_path,
        context,
    )


def _prepare_and_promote_release(
    git_cmd: list[str],
    cwd: Path | str,
    release_tag: str,
    target_sha: str,
    *,
    input_fn=None,
    candidate_validator=None,
) -> ReleaseUpgradeResult:
    """Capture user state, promote a release, and defer restore to finalization."""

    root = Path(cwd).resolve()
    branch = "hermes-release"
    maintenance_ref = f"refs/heads/{branch}"
    _validate_release_tag_name(git_cmd, root, release_tag)
    _reject_unfinished_release_transaction(git_cmd, root)
    original_snapshot = _capture_release_git_snapshot(
        git_cmd, root, maintenance_ref=maintenance_ref
    )
    original_branch = _release_snapshot_branch(original_snapshot)
    original_head_sha = original_snapshot.head_sha
    maintenance_sha = original_snapshot.maintenance_ref_sha
    metadata = _read_release_base_metadata_at_commit(git_cmd, root, maintenance_sha)
    _validate_release_base_metadata(git_cmd, root, metadata, maintenance_sha)
    _validate_incremental_release_target(
        git_cmd,
        root,
        previous_base_sha=metadata.base_sha,
        target_sha=target_sha.lower(),
    )
    payload = _git_diff_bytes(git_cmd, root, metadata.base_sha, maintenance_sha)
    context = _create_release_upgrade_context(
        git_cmd,
        root,
        original_branch=original_branch,
        original_head_sha=original_head_sha,
        maintenance_old_sha=maintenance_sha,
        release_tag=release_tag,
        base_sha=metadata.base_sha,
        target_sha=target_sha.lower(),
        payload=payload,
    )
    _release_snapshot_with_journal(context, "original_snapshot", original_snapshot)
    original_clean_snapshot = _release_clean_snapshot(
        original_snapshot,
        symbolic_head=original_snapshot.symbolic_head,
        head_sha=original_snapshot.head_sha,
        maintenance_ref_sha=maintenance_sha,
        head_tree_sha=original_snapshot.head_tree_sha,
    )
    _release_snapshot_with_journal(
        context, "original_clean_snapshot", original_clean_snapshot
    )
    try:
        try:
            _validate_release_git_snapshot(
                git_cmd, root, original_snapshot, label="stash capture precondition"
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
        pre_capture_status = subprocess.run(
            git_cmd + ["status", "--porcelain"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        local_state_present = (
            bool(pre_capture_status.stdout.strip())
            if pre_capture_status.returncode == 0
            else None
        )
        _journal_update(
            context,
            "stash-capture-started",
            local_state_present=local_state_present,
            stash_capture_required=(
                local_state_present if local_state_present is not None else None
            ),
            stash_capture_confirmed=False,
            stash_capture_uncertain=local_state_present is None,
            stash_pending=False,
        )
        if local_state_present is None:
            detail = (pre_capture_status.stderr or pre_capture_status.stdout or "").strip()
            raise RuntimeError(
                "Could not inspect local state before stash capture"
                + (f": {detail.splitlines()[0]}" if detail else "")
            )
        try:
            _validate_release_git_snapshot(
                git_cmd, root, original_snapshot, label="stash mutation precondition"
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
        _journal_update(context, "stash-capture-started")
        stash_ref = _stash_local_changes_if_needed(
            git_cmd, root, marker=context.journal["stash_marker"]
        )
        if local_state_present:
            _journal_update(
                context,
                "stash-capture-verifying",
                stash_sha=None,
                stash_pending=False,
                stash_capture_confirmed=False,
                stash_capture_uncertain=True,
            )
            verified_stash_ref = _verify_release_stash_capture(
                git_cmd, root, context, stash_ref
            )
            if verified_stash_ref is None:
                _journal_update(
                    context,
                    "stash-capture-uncertain",
                    stash_sha=None,
                    stash_pending=False,
                    stash_capture_confirmed=False,
                    stash_capture_uncertain=True,
                )
                raise RuntimeError(
                    "Stash capture could not independently verify one unique "
                    "immutable stash SHA."
                )
            _journal_update(
                context,
                "stashed",
                stash_sha=verified_stash_ref,
                stash_pending=True,
                stash_capture_confirmed=True,
                stash_capture_uncertain=False,
            )
        else:
            if stash_ref is not None:
                raise RuntimeError(
                    "Stash capture returned an entry even though the pre-capture "
                    "worktree was clean."
                )
            _journal_update(
                context,
                "no-stash",
                stash_pending=False,
                stash_capture_required=False,
                stash_capture_confirmed=False,
                stash_capture_uncertain=False,
            )
        post_stash_expected = _release_clean_snapshot(
            original_snapshot,
            symbolic_head=original_snapshot.symbolic_head,
            head_sha=original_snapshot.head_sha,
            maintenance_ref_sha=maintenance_sha,
            head_tree_sha=original_snapshot.head_tree_sha,
        )
        try:
            post_stash_snapshot = _validate_release_git_snapshot(
                git_cmd,
                root,
                post_stash_expected,
                label="post-stash clean checkout",
            )
        except Exception as exc:
            _release_mark_manual_interference(context, exc)
            raise
        _release_snapshot_with_journal(
            context,
            "post_stash_snapshot",
            post_stash_snapshot,
        )
        maintenance_tree_sha = _release_git_sha(
            git_cmd,
            root,
            ["rev-parse", "--verify", f"{maintenance_sha}^{{tree}}"],
            label="maintenance tree before checkout",
        )
        if original_branch != branch:
            checkout = subprocess.run(
                git_cmd + ["checkout", branch],
                cwd=root,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
            if checkout.returncode != 0:
                detail = (checkout.stderr or checkout.stdout or "").strip()
                raise RuntimeError(
                    f"Could not switch to local maintenance branch '{branch}': "
                    f"{detail.splitlines()[0] if detail else 'git checkout failed'}"
                )
            maintenance_expected = _release_clean_snapshot(
                post_stash_snapshot,
                symbolic_head=maintenance_ref,
                head_sha=maintenance_sha,
                maintenance_ref_sha=maintenance_sha,
                head_tree_sha=maintenance_tree_sha,
            )
            try:
                _validate_release_git_snapshot(
                    git_cmd,
                    root,
                    maintenance_expected,
                    label="maintenance branch checkout",
                )
            except Exception as exc:
                _release_mark_manual_interference(context, exc)
                raise
            _release_snapshot_with_journal(
                context,
                "maintenance_snapshot",
                maintenance_expected,
            )
        result = _upgrade_release_transaction(
            git_cmd,
            root,
            release_tag,
            target_sha,
            transaction_context=context,
            candidate_validator=candidate_validator,
        )
        return result
    except BaseException:
        capture_required = context.journal.get("stash_capture_required")
        capture_confirmed = bool(context.journal.get("stash_capture_confirmed"))
        if capture_required is not False and not capture_confirmed:
            recovered_stash = None
            if not context.journal.get("stash_capture_uncertain"):
                try:
                    # A helper can raise after `git stash push` has created the
                    # entry.  Only a unique marker match can promote this
                    # uncertain state to a confirmed immutable capture.
                    recovered_stash = _refresh_transaction_stash_identity(
                        git_cmd, root, context
                    )
                except BaseException as exc:
                    logger.warning("Could not refresh stash identity after capture fault: %s", exc)
            if recovered_stash:
                try:
                    _journal_update(
                        context,
                        "stashed",
                        stash_sha=recovered_stash,
                        stash_pending=True,
                        stash_capture_confirmed=True,
                        stash_capture_uncertain=False,
                    )
                    if "post_stash_snapshot" not in context.journal:
                        recovered_original = _release_context_snapshot(
                            context, "original_snapshot"
                        )
                        recovered_expected = _release_clean_snapshot(
                            recovered_original,
                            symbolic_head=recovered_original.symbolic_head,
                            head_sha=recovered_original.head_sha,
                            maintenance_ref_sha=recovered_original.maintenance_ref_sha,
                            head_tree_sha=recovered_original.head_tree_sha,
                        )
                        recovered_snapshot = _validate_release_git_snapshot(
                            git_cmd,
                            root,
                            recovered_expected,
                            label="recovered post-stash clean checkout",
                        )
                        _release_snapshot_with_journal(
                            context,
                            "post_stash_snapshot",
                            recovered_snapshot,
                        )
                except BaseException as exc:
                    logger.warning("Could not durably confirm recovered stash: %s", exc)
            else:
                try:
                    _journal_update(
                        context,
                        "stash-capture-uncertain",
                        stash_sha=None,
                        stash_pending=False,
                        stash_capture_confirmed=False,
                        stash_capture_uncertain=True,
                    )
                except BaseException as exc:
                    logger.warning("Could not durably mark stash capture uncertain: %s", exc)
        else:
            try:
                _refresh_transaction_stash_identity(git_cmd, root, context)
            except BaseException as exc:
                logger.warning("Could not refresh stash identity during finalization: %s", exc)
        if (
            context.journal.get("local_state_present") is False
            and not _release_has_manual_interference(context)
        ):
            # Preserve the ordinary update contract for a checkout that was
            # proven to contain no user state.  This fallback runs before the
            # transactional release finalizer; the finalizer itself never
            # resets or cleans the live checkout.
            subprocess.run(
                git_cmd + ["reset", "--hard", "HEAD"],
                cwd=root,
                capture_output=True,
            )
            subprocess.run(
                git_cmd + ["clean", "-fd"],
                cwd=root,
                capture_output=True,
            )
        _finalize_release_upgrade_for_orchestration(
            git_cmd,
            root,
            context,
            input_fn=input_fn,
            primary_exc_info=sys.exc_info(),
        )
        raise


def _verify_release_stash_capture(
    git_cmd: list[str],
    root: Path,
    context: ReleaseUpgradeContext,
    stash_ref: object,
) -> str | None:
    """Independently verify the helper's stash SHA against this transaction marker."""

    if not isinstance(stash_ref, str) or not _SHA_RE.fullmatch(stash_ref):
        return None
    marker = context.journal.get("stash_marker")
    if not isinstance(marker, str) or not marker:
        return None
    listing = subprocess.run(
        git_cmd + ["stash", "list", "--format=%H%x00%gs"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if listing.returncode != 0:
        return None
    matches: list[str] = []
    for line in listing.stdout.splitlines():
        commit, separator, subject = line.partition("\0")
        commit = commit.strip()
        if separator and marker in subject and _SHA_RE.fullmatch(commit):
            matches.append(commit.lower())
    if len(matches) != 1 or matches[0] != stash_ref.lower():
        return None
    verified = subprocess.run(
        git_cmd + ["cat-file", "-e", f"{matches[0]}^{{commit}}"],
        cwd=root,
        capture_output=True,
    )
    if verified.returncode != 0:
        return None
    return matches[0]


def _refresh_transaction_stash_identity(
    git_cmd: list[str], root: Path, context: ReleaseUpgradeContext
) -> str | None:
    """Resolve and durably confirm one immutable stash for this marker."""

    journal = context.journal
    current = journal.get("stash_sha")
    if journal.get("stash_capture_confirmed") and isinstance(current, str):
        if _SHA_RE.fullmatch(current):
            verified = subprocess.run(
                git_cmd + ["cat-file", "-e", f"{current}^{{commit}}"],
                cwd=root,
                capture_output=True,
            )
            if verified.returncode == 0:
                return current.lower()
        return None

    marker = journal.get("stash_marker")
    if not marker:
        return None
    listing = subprocess.run(
        git_cmd + ["stash", "list", "--format=%H%x00%gs"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if listing.returncode != 0:
        return None
    matches: list[str] = []
    for line in listing.stdout.splitlines():
        commit, separator, subject = line.partition("\0")
        commit = commit.strip()
        if separator and marker in subject and _SHA_RE.fullmatch(commit):
            matches.append(commit.lower())
    if len(matches) != 1:
        return None
    commit = matches[0]
    verified = subprocess.run(
        git_cmd + ["cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=root,
        capture_output=True,
    )
    if verified.returncode != 0:
        return None
    _journal_update(
        context,
        "stash-pending",
        stash_sha=commit,
        stash_pending=True,
        stash_capture_confirmed=True,
        stash_capture_uncertain=False,
    )
    return commit


def _transaction_is_promoted(
    git_cmd: list[str], root: Path, context: ReleaseUpgradeContext
) -> bool:
    journal = context.journal
    candidate_sha = journal.get("candidate_sha")
    if not isinstance(candidate_sha, str) or not _SHA_RE.fullmatch(candidate_sha):
        return False
    live_sha = _git_resolve_commit(
        git_cmd, root, f"refs/heads/{journal.get('maintenance_branch', 'hermes-release')}"
    )
    return live_sha == candidate_sha and journal.get("phase") in {
        "promoting",
        "promoted",
        "promotion-needs-recovery",
        "candidate-cleanup",
        "candidate-cleanup-failed",
        "finalizing",
        "stash-restore-conflict",
        "finalized",
    }


def _restore_original_release_checkout(
    git_cmd: list[str], root: Path, context: ReleaseUpgradeContext
) -> None:
    journal = context.journal
    original_branch = journal.get("original_branch")
    original_sha = journal["original_head_sha"]
    promoted = _transaction_is_promoted(git_cmd, root, context)
    if original_branch:
        checkout = subprocess.run(
            git_cmd + ["checkout", "--force", original_branch],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if checkout.returncode != 0:
            detail = (checkout.stderr or checkout.stdout or "").strip()
            raise RuntimeError(
                f"Could not restore original branch '{original_branch}'"
                + (f": {detail.splitlines()[0]}" if detail else "")
            )
        expected_sha = original_sha
        if promoted and original_branch == journal.get("maintenance_branch"):
            expected_sha = journal.get("candidate_sha") or original_sha
        actual_branch = subprocess.run(
            git_cmd + ["symbolic-ref", "--quiet", "--short", "HEAD"],
            cwd=root,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        actual_sha = _git_resolve_commit(git_cmd, root, "HEAD")
        if (
            actual_branch.returncode != 0
            or actual_branch.stdout.strip() != original_branch
            or actual_sha != expected_sha
        ):
            raise RuntimeError(
                f"Original branch restore did not verify: expected "
                f"{original_branch}@{expected_sha}, got "
                f"{actual_branch.stdout.strip() or 'detached'}@{actual_sha}"
            )
        return

    checkout = subprocess.run(
        git_cmd + ["checkout", "--detach", original_sha],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if checkout.returncode != 0:
        detail = (checkout.stderr or checkout.stdout or "").strip()
        raise RuntimeError(
            "Could not restore original detached HEAD"
            + (f": {detail.splitlines()[0]}" if detail else "")
        )
    actual_sha = _git_resolve_commit(git_cmd, root, "HEAD")
    actual_branch = subprocess.run(
        git_cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if actual_sha != original_sha or actual_branch.stdout.strip() != "HEAD":
        raise RuntimeError(
            f"Detached HEAD restore did not verify: expected {original_sha}, "
            f"got {actual_branch.stdout.strip()}@{actual_sha}"
        )


def _release_finalization_marker_state(journal: dict) -> str:
    """Classify the context marker before any live-worktree inspection."""

    phase = journal.get("phase")
    state = journal.get("state")
    verified = journal.get("final_state_verified")
    finalized = journal.get("finalized")
    if phase == "finalized":
        if verified is not True:
            return "inconsistent"
        if state is not None and state != "finalized":
            return "inconsistent"
        if finalized is not None and finalized is not True:
            return "inconsistent"
        return "verified"
    if verified is True or finalized is True or state == "finalized":
        return "inconsistent"
    return "unfinalized"


_TERMINAL_RECONCILIATION_REQUIRED_FIELDS = frozenset(
    {
        "version",
        "transaction_id",
        "phase",
        "state",
        "original_branch",
        "original_ref",
        "original_head_sha",
        "maintenance_branch",
        "maintenance_old_sha",
        "old_sha",
        "release_tag",
        "base_sha",
        "target_sha",
        "backup_ref",
        "backup_created",
        "candidate_branch",
        "candidate_path",
        "candidate_sha",
        "candidate_cleanup",
        "payload_path",
        "payload_sha256",
        "payload_bytes",
        "stash_marker",
        "stash_sha",
        "local_state_present",
        "stash_capture_required",
        "stash_capture_confirmed",
        "stash_capture_uncertain",
        "stash_pending",
        "stash_apply_attempted",
        "stash_applied",
        "checkout_restored",
        "final_state_verified",
    }
)
_TERMINAL_RECONCILIATION_BOOL_FIELDS = frozenset(
    {
        "backup_created",
        "candidate_cleanup",
        "stash_capture_confirmed",
        "stash_capture_uncertain",
        "stash_pending",
        "stash_apply_attempted",
        "stash_applied",
        "checkout_restored",
        "final_state_verified",
        "candidate_created",
        "finalized",
    }
)
_TERMINAL_RECONCILIATION_SHA_FIELDS = (
    "original_head_sha",
    "maintenance_old_sha",
    "old_sha",
    "base_sha",
    "target_sha",
    "candidate_sha",
)


def _terminal_journal_digest(journal: dict) -> str:
    """Return the digest of the exact in-memory pre-terminal journal."""

    encoded = json.dumps(
        journal,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_terminal_reconciliation_paths(
    context: ReleaseUpgradeContext,
) -> tuple[Path, Path]:
    """Validate the fixed transaction/journal path topology without Git."""

    common = Path(context.common_dir)
    transactions = common / "hermes-upgrade-transactions"
    transaction_dir = Path(context.transaction_dir)
    journal_path = Path(context.journal_path)
    if not common.is_absolute() or not transaction_dir.is_absolute() or not journal_path.is_absolute():
        raise RuntimeError("Uncertain terminal journal paths must be absolute.")
    if transaction_dir.parent != transactions:
        raise RuntimeError("Uncertain terminal journal transaction directory escaped its root.")
    if journal_path != transaction_dir / "journal.json":
        raise RuntimeError("Uncertain terminal journal path escaped its transaction directory.")

    for directory, label in (
        (common, "Git common directory"),
        (transactions, "release transactions directory"),
    ):
        try:
            info = directory.lstat()
        except OSError as exc:
            raise RuntimeError(f"Could not inspect {label}: {directory}") from exc
        if stat.S_ISLNK(info.st_mode) or not stat.S_ISDIR(info.st_mode):
            raise RuntimeError(f"Refusing non-directory {label}: {directory}")

    try:
        transaction_info = transaction_dir.lstat()
    except FileNotFoundError:
        return transactions, transaction_dir
    except OSError as exc:
        raise RuntimeError(
            f"Could not inspect uncertain release transaction directory: {transaction_dir}"
        ) from exc
    if stat.S_ISLNK(transaction_info.st_mode) or not stat.S_ISDIR(transaction_info.st_mode):
        raise RuntimeError(
            f"Refusing non-directory uncertain release transaction: {transaction_dir}"
        )
    return transactions, transaction_dir


def _validate_terminal_reconciliation_journal(
    context: ReleaseUpgradeContext,
    journal: object,
    *,
    terminal: bool,
) -> dict:
    """Validate the strict schema needed before filesystem-only reconciliation."""

    if not isinstance(journal, dict):
        raise RuntimeError("Uncertain terminal journal is not a JSON object.")
    missing = _TERMINAL_RECONCILIATION_REQUIRED_FIELDS.difference(journal)
    if missing:
        raise RuntimeError(
            "Uncertain terminal journal is missing control fields: "
            + ", ".join(sorted(missing))
        )
    marker_state = _release_finalization_marker_state(journal)
    expected_state = "verified" if terminal else "unfinalized"
    if marker_state != expected_state:
        raise RuntimeError(
            f"Uncertain terminal journal has invalid marker state {marker_state!r}; "
            f"expected {expected_state!r}."
        )
    if terminal:
        if (
            journal.get("phase") != "finalized"
            or journal.get("state") != "finalized"
            or journal.get("final_state_verified") is not True
            or journal.get("finalized") is not True
        ):
            raise RuntimeError("Uncertain terminal journal terminal controls are not exact.")
    elif journal.get("final_state_verified") is not False:
        raise RuntimeError("Uncertain terminal journal prior marker is not explicitly nonterminal.")
    if journal.get("version") != 2:
        raise RuntimeError("Uncertain terminal journal has an unsupported version.")

    transaction_dir = Path(context.transaction_dir)
    transaction_id = journal.get("transaction_id")
    if not isinstance(transaction_id, str) or transaction_id != transaction_dir.name:
        raise RuntimeError("Uncertain terminal journal transaction identity mismatch.")

    original_branch = journal.get("original_branch")
    original_ref = journal.get("original_ref")
    if original_branch is not None and (
        not isinstance(original_branch, str)
        or not original_branch
        or original_ref != f"refs/heads/{original_branch}"
    ):
        raise RuntimeError("Uncertain terminal journal original branch/ref is invalid.")
    if original_branch is None and original_ref is not None:
        raise RuntimeError("Uncertain terminal journal has a ref without an original branch.")

    for field in _TERMINAL_RECONCILIATION_SHA_FIELDS:
        value = journal.get(field)
        if not isinstance(value, str) or _SHA_RE.fullmatch(value) is None:
            raise RuntimeError(f"Uncertain terminal journal has an invalid {field}.")
    stash_sha = journal.get("stash_sha")
    if stash_sha is not None and (
        not isinstance(stash_sha, str) or _SHA_RE.fullmatch(stash_sha) is None
    ):
        raise RuntimeError("Uncertain terminal journal has an invalid stash SHA.")
    payload_sha256 = journal.get("payload_sha256")
    if not isinstance(payload_sha256, str) or re.fullmatch(r"[0-9a-fA-F]{64}", payload_sha256) is None:
        raise RuntimeError("Uncertain terminal journal has an invalid payload digest.")
    payload_bytes = journal.get("payload_bytes")
    if not isinstance(payload_bytes, int) or isinstance(payload_bytes, bool) or payload_bytes < 0:
        raise RuntimeError("Uncertain terminal journal has an invalid payload length.")

    payload_path = journal.get("payload_path")
    expected_payload = transaction_dir / "runtime-local-maintenance.patch"
    if not isinstance(payload_path, str) or Path(payload_path) != expected_payload:
        raise RuntimeError("Uncertain terminal journal payload path escaped its transaction directory.")
    for field in (
        "maintenance_branch",
        "release_tag",
        "stash_marker",
        "backup_ref",
        "candidate_branch",
        "candidate_path",
    ):
        if not isinstance(journal.get(field), str) or not journal[field]:
            raise RuntimeError(f"Uncertain terminal journal has an invalid {field}.")
    if journal["backup_ref"] != f"refs/hermes-upgrade/backups/{transaction_id}":
        raise RuntimeError("Uncertain terminal journal backup ref does not belong to this transaction.")
    if journal["candidate_branch"] != f"hermes-upgrade-candidate/{transaction_id}":
        raise RuntimeError("Uncertain terminal journal candidate branch does not belong to this transaction.")
    if not Path(journal["candidate_path"]).is_absolute():
        raise RuntimeError("Uncertain terminal journal candidate path is not absolute.")

    if journal.get("local_state_present") not in (None, True, False):
        raise RuntimeError("Uncertain terminal journal has an invalid local-state flag.")
    if journal.get("stash_capture_required") not in (None, True, False):
        raise RuntimeError("Uncertain terminal journal has an invalid stash-capture flag.")
    for field in _TERMINAL_RECONCILIATION_BOOL_FIELDS:
        if field in journal and not isinstance(journal[field], bool):
            raise RuntimeError(f"Uncertain terminal journal has an invalid {field} flag.")
    if journal.get("state") != journal.get("phase"):
        raise RuntimeError("Uncertain terminal journal phase/state controls disagree.")
    return journal


def _read_uncertain_terminal_journal(context: ReleaseUpgradeContext) -> dict:
    """Read the journal for reconciliation while rejecting unsafe entries."""

    transactions, transaction_dir = _validate_terminal_reconciliation_paths(context)
    journal_path = Path(context.journal_path)
    try:
        return _read_release_transaction_journal(
            transaction_dir,
            journal_path=journal_path,
            expected_transactions_dir=transactions,
        )
    except _ReleaseJournalReadError as exc:
        if exc.kind == "missing":
            raise RuntimeError(
                "Uncertain terminal journal is missing; refusing to infer completion."
            ) from None
        if exc.kind == "oversize":
            raise RuntimeError(
                "Uncertain terminal journal exceeds the maximum allowed size; refusing to infer completion."
            ) from None
        if exc.kind == "unsafe":
            raise RuntimeError(
                "Refusing non-regular, symlinked, or unsafe uncertain terminal journal."
            ) from None
        if exc.kind == "changed":
            raise RuntimeError(
                "Uncertain terminal journal changed while being read; refusing to infer completion."
            ) from None
        if exc.kind == "invalid":
            raise RuntimeError(
                "Uncertain terminal journal is unreadable or invalid JSON."
            ) from None
        raise RuntimeError(
            "Could not read uncertain terminal journal; refusing to infer completion."
        ) from None


def _reconcile_uncertain_final_marker(context: ReleaseUpgradeContext) -> bool:
    """Reconcile a possibly-replaced terminal marker without touching Git."""

    prior = context.journal
    candidate = context.final_marker_candidate
    if context.final_marker_write_uncertain is not True:
        raise RuntimeError("Terminal marker reconciliation was requested without an uncertainty latch.")
    if type(prior) is not dict or type(candidate) is not dict:
        raise RuntimeError("Uncertain terminal marker state is incomplete; preserving evidence.")
    _validate_terminal_reconciliation_journal(context, prior, terminal=False)
    if not isinstance(context.final_marker_prior_digest, str):
        raise RuntimeError("Uncertain terminal marker has no immutable prior-journal digest.")
    if _terminal_journal_digest(prior) != context.final_marker_prior_digest:
        raise RuntimeError("Uncertain terminal marker prior journal was mutated in memory.")

    expected_candidate = dict(prior)
    expected_candidate.update(
        {
            "phase": "finalized",
            "state": "finalized",
            "final_state_verified": True,
            "finalized": True,
        }
    )
    if candidate != expected_candidate:
        raise RuntimeError("Uncertain terminal marker candidate no longer matches its prior journal.")
    _validate_terminal_reconciliation_journal(context, candidate, terminal=True)

    disk = _read_uncertain_terminal_journal(context)
    if disk != prior and disk != candidate:
        raise RuntimeError(
            "Uncertain terminal journal does not match either the exact prior or terminal candidate; "
            "preserving evidence."
        )

    try:
        _write_transaction_journal(
            context.common_dir, candidate, path=context.journal_path
        )
    except BaseException:
        # The context must remain nonterminal and latched even if this retry
        # reaches replace before the required child-directory fsync fails.
        context.final_marker_write_uncertain = True
        raise

    context.journal.clear()
    context.journal.update(candidate)
    context.final_marker_write_uncertain = False
    context.final_marker_candidate = None
    context.final_marker_prior_digest = None
    return _acknowledge_finalized_release(context)



def _acknowledge_finalized_release(context: ReleaseUpgradeContext) -> bool:
    """Remove known terminal evidence and durably acknowledge its directory entries."""

    transaction_dir = context.transaction_dir
    transactions = transaction_dir.parent
    try:
        transaction_info = transaction_dir.lstat()
    except FileNotFoundError:
        # A prior attempt may have removed the child before losing the root
        # directory fsync.  Retrying this filesystem-only acknowledgment is
        # safe and repairs that final durability step.
        _fsync_directory(transactions, required=True)
        return True
    if stat.S_ISLNK(transaction_info.st_mode) or not stat.S_ISDIR(
        transaction_info.st_mode
    ):
        raise RuntimeError(f"Refusing non-directory release transaction: {transaction_dir}")

    known_entries = (
        context.journal_path,
        transaction_dir / "runtime-local-maintenance.patch",
    )
    for entry in known_entries:
        try:
            entry_info = entry.lstat()
        except FileNotFoundError:
            continue
        if stat.S_ISLNK(entry_info.st_mode) or not stat.S_ISREG(entry_info.st_mode):
            raise RuntimeError(f"Refusing non-regular release transaction evidence: {entry}")
        entry.unlink()

    # The known evidence is gone, but unknown entries are deliberately not
    # removed.  If one remains, rmdir fails closed and the terminal context is
    # retained for a later acknowledgment retry.
    _fsync_directory(transaction_dir, required=True)
    try:
        transaction_dir.rmdir()
    except FileNotFoundError:
        pass
    except OSError as exc:
        logger.warning(
            "Release transaction acknowledgment retained unknown evidence in %s: %s",
            transaction_dir,
            exc,
        )
        return False
    _fsync_directory(transactions, required=True)
    return True



def _mark_release_finalized(context: ReleaseUpgradeContext) -> None:
    """Persist the terminal marker before exposing it to retry logic."""

    prior_journal = dict(context.journal)
    terminal_journal = dict(prior_journal)
    terminal_journal.update(
        {
            "phase": "finalized",
            "state": "finalized",
            "final_state_verified": True,
            "finalized": True,
        }
    )
    context.final_marker_candidate = dict(terminal_journal)
    context.final_marker_prior_digest = _terminal_journal_digest(prior_journal)
    try:
        _write_transaction_journal(
            context.common_dir, terminal_journal, path=context.journal_path
        )
    except BaseException:
        # os.replace may already have made the terminal bytes visible when the
        # required child-directory fsync fails.  Keep the live context
        # nonterminal, but latch the exact candidate before propagating.
        context.final_marker_write_uncertain = True
        raise
    context.journal.clear()
    context.journal.update(terminal_journal)
    context.final_marker_write_uncertain = False
    context.final_marker_candidate = None
    context.final_marker_prior_digest = None


def _release_expected_original_clean_snapshot(
    original: ReleaseGitSnapshot,
    candidate: ReleaseGitSnapshot | None,
) -> ReleaseGitSnapshot:
    """Build the clean checkout expected immediately before stash apply."""
    if candidate is not None and original.symbolic_head == candidate.maintenance_ref:
        head_sha = candidate.head_sha
        head_tree_sha = candidate.head_tree_sha
    else:
        head_sha = original.head_sha
        head_tree_sha = original.head_tree_sha
    maintenance_ref_sha = candidate.maintenance_ref_sha if candidate else original.maintenance_ref_sha
    return _release_clean_snapshot(
        original,
        symbolic_head=original.symbolic_head,
        head_sha=head_sha,
        maintenance_ref_sha=maintenance_ref_sha,
        head_tree_sha=head_tree_sha,
    )


def _release_cleanup_candidate(
    git_cmd: list[str], root: Path, context: ReleaseUpgradeContext
) -> bool:
    journal = context.journal
    transaction_id = journal.get("transaction_id")
    transactions = context.common_dir / "hermes-upgrade-transactions"
    transaction_dir = Path(context.transaction_dir)
    if (
        not isinstance(transaction_id, str)
        or not re.fullmatch(r"[0-9a-f]{32}", transaction_id)
        or transaction_dir.parent != transactions
        or transaction_dir.name != transaction_id
    ):
        logger.warning(
            "Refusing release candidate cleanup for an untrusted transaction topology: %s",
            context.journal_path,
        )
        return False

    expected_candidate_path = context.common_dir.parent.parent / (
        f"hermes-upgrade-candidate-{transaction_id}"
    )
    candidate_path_value = journal.get("candidate_path")
    candidate_branch = journal.get("candidate_branch")
    if (
        not isinstance(candidate_path_value, str)
        or Path(candidate_path_value) != expected_candidate_path
        or not isinstance(candidate_branch, str)
        or candidate_branch != f"hermes-upgrade-candidate/{transaction_id}"
    ):
        logger.warning(
            "Refusing release candidate cleanup for an escaped candidate path or branch: %s",
            context.journal_path,
        )
        return False
    candidate_path = expected_candidate_path
    try:
        candidate_info = candidate_path.lstat()
    except FileNotFoundError:
        return False
    except OSError:
        return False
    if stat.S_ISLNK(candidate_info.st_mode) or not stat.S_ISDIR(candidate_info.st_mode):
        logger.warning("Refusing non-directory release candidate cleanup path: %s", candidate_path)
        return False
    cleanup = subprocess.run(
        git_cmd + ["worktree", "remove", "--force", str(candidate_path)],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if cleanup.returncode != 0:
        return False
    branch_cleanup = subprocess.run(
        git_cmd + ["branch", "-D", candidate_branch],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (
        branch_cleanup.returncode == 0
        and _git_resolve_commit(git_cmd, root, f"refs/heads/{candidate_branch}") is None
        and not candidate_path.exists()
    )


def _finalize_release_upgrade(
    git_cmd: list[str],
    cwd: Path | str,
    context: ReleaseUpgradeContext,
    *,
    input_fn=None,
) -> bool:
    """Restore the original checkout and apply/drop the immutable stash once."""

    root = Path(cwd).resolve()
    if context.final_marker_write_uncertain is True:
        # A terminal marker write may have replaced the file before its
        # required child-directory fsync failed.  Reconcile only the journal;
        # never inspect or mutate the live Git checkout on this retry.
        try:
            return _reconcile_uncertain_final_marker(context)
        except (KeyboardInterrupt, SystemExit):
            raise
        except BaseException as exc:
            logger.warning(
                "Uncertain terminal marker reconciliation did not complete; journal retained at %s: %s",
                context.journal_path,
                exc,
            )
            print(f"⚠ Release user-state recovery is incomplete; journal: {context.journal_path}")
            return False
    if (
        context.final_marker_write_uncertain is not False
        or context.final_marker_candidate is not None
        or context.final_marker_prior_digest is not None
    ):
        logger.error(
            "Refusing release finalization with invalid in-memory terminal marker state: %s",
            context.journal_path,
        )
        print(f"✗ Release finalization marker state is invalid; journal: {context.journal_path}")
        return False
    journal = context.journal
    try:
        marker_state = _release_finalization_marker_state(journal)
        if marker_state == "verified":
            # The final live state was already verified.  Only acknowledge the
            # durable journal; never re-inspect or mutate the checkout.
            return _acknowledge_finalized_release(context)
        if marker_state == "inconsistent":
            logger.error(
                "Refusing to replay release finalization with an inconsistent "
                "terminal journal marker: %s",
                context.journal_path,
            )
            print(
                "✗ Release finalization has an inconsistent terminal marker; "
                "refusing destructive cleanup."
            )
            print(f"  Recovery journal retained at {context.journal_path}")
            return False
        try:
            context.journal_path.lstat()
        except FileNotFoundError:
            # Journal absence is not proof that first finalization completed.
            print(
                "✗ Release finalization journal is missing before terminal "
                "verification; refusing destructive cleanup."
            )
            print(f"  Expected recovery journal at {context.journal_path}")
            return False
        except OSError as exc:
            logger.warning(
                "Could not inspect release finalization journal %s: %s",
                context.journal_path,
                exc,
            )
            print(f"⚠ Release user-state recovery is incomplete; journal: {context.journal_path}")
            return False

        if _release_has_manual_interference(context):
            print(
                "✗ Release finalization detected manual Git interference; "
                "refusing destructive recovery."
            )
            print(f"  Recovery journal retained at {context.journal_path}")
            return False

        if journal.get("stash_capture_uncertain") and not journal.get(
            "stash_capture_confirmed"
        ):
            stash_sha = None
        elif journal.get("stash_pending"):
            stash_sha = _refresh_transaction_stash_identity(git_cmd, root, context)
        else:
            stash_sha = journal.get("stash_sha") if journal.get("stash_capture_confirmed") else None
        stash_pending = bool(journal.get("stash_pending"))
        stash_capture_required = journal.get("stash_capture_required")
        stash_capture_confirmed = bool(journal.get("stash_capture_confirmed"))
        if stash_capture_required is not False and not stash_capture_confirmed:
            _journal_update(
                context,
                "stash-capture-uncertain",
                stash_sha=None,
                stash_pending=False,
                stash_capture_confirmed=False,
                stash_capture_uncertain=True,
            )
            print(
                "✗ Stash capture is unconfirmed; refusing reset, clean, checkout, "
                "stash apply/drop, and journal deletion."
            )
            print(f"  Recovery journal retained at {context.journal_path}")
            print(
                f"  Inspect marker {journal.get('stash_marker')!r} with: "
                "git stash list --format='%gd %H %s'"
            )
            print("  Do not retry destructive cleanup until one immutable SHA is confirmed.")
            return False
        if stash_pending and not stash_sha:
            print(
                "✗ Release update still has an unidentified pending stash; "
                f"journal: {context.journal_path}"
            )
            print(
                f"  Search for marker {journal.get('stash_marker')!r} with: "
                "git stash list --format='%gd %H %s'"
            )
            return False

        if stash_pending and journal.get("stash_apply_attempted") and not journal.get(
            "stash_applied"
        ):
            assert stash_sha is not None
            _journal_update(context, "stash-restore-conflict")
            print(
                "✗ Release user-state restore was already attempted and remains "
                "unresolved; leaving the immutable stash untouched."
            )
            print(f"  Recovery journal retained at {context.journal_path}")
            _print_stash_cleanup_guidance(stash_sha)
            return False

        # The exact snapshot is the preparation step.  Never reset or clean a
        # release checkout to make it fit an expectation: any mismatch is
        # manual interference and therefore a zero-mutator failure.
        candidate_snapshot: ReleaseGitSnapshot | None = None
        try:
            guard_identity_only = False
            if journal.get("checkout_restored"):
                guard_key = (
                    "restored_snapshot" if journal.get("stash_applied")
                    else "restore_clean_snapshot"
                )
                guard_snapshot = _release_context_snapshot(context, guard_key)
                guard_identity_only = bool(journal.get("stash_applied"))
                if not guard_identity_only and not _release_snapshot_is_clean(guard_snapshot):
                    raise ReleaseGitStateError(
                        "Stored pre-apply release checkout is not clean."
                    )
            elif "candidate_snapshot" in journal:
                candidate_snapshot = _release_context_snapshot(context, "candidate_snapshot")
                if not _release_snapshot_is_clean(candidate_snapshot):
                    raise ReleaseGitStateError(
                        "Stored promoted release checkout is not clean."
                    )
                guard_snapshot = candidate_snapshot
            elif "post_cas_snapshot" in journal:
                # The CAS has moved the maintenance ref, but a failed live
                # synchronization can leave the old index/worktree temporarily
                # skewed from that ref.  This exact snapshot is the only
                # authorized recovery entry for that phase; it is intentionally
                # not classified as a clean checkout.
                guard_snapshot = _release_context_snapshot(context, "post_cas_snapshot")
            elif "maintenance_snapshot" in journal:
                guard_snapshot = _release_context_snapshot(context, "maintenance_snapshot")
                if not _release_snapshot_is_clean(guard_snapshot):
                    raise ReleaseGitStateError(
                        "Stored maintenance checkout is not clean."
                    )
            elif "post_stash_snapshot" in journal:
                guard_snapshot = _release_context_snapshot(context, "post_stash_snapshot")
                if not _release_snapshot_is_clean(guard_snapshot):
                    raise ReleaseGitStateError(
                        "Stored post-stash checkout is not clean."
                    )
            else:
                raise ReleaseGitStateError(
                    "No bound clean release snapshot is available for finalization."
                )
            if guard_identity_only:
                _validate_release_restored_snapshot_identity(
                    git_cmd,
                    root,
                    guard_snapshot,
                    label="finalization entry",
                )
            else:
                _validate_release_git_snapshot(
                    git_cmd,
                    root,
                    guard_snapshot,
                    label="finalization entry",
                )
        except Exception as exc:
            _release_mark_manual_interference(context, f"finalization incomplete: {exc}")
            print(
                "✗ Release finalization detected live checkout interference; "
                "no reset, clean, checkout, stash apply, or stash drop was run."
            )
            print(f"  Recovery journal retained at {context.journal_path}")
            return False

        prior_phase = journal.get("phase")
        prior_stash_applied = bool(journal.get("stash_applied"))
        _journal_update(context, "finalizing")
        original_snapshot = _release_context_snapshot(context, "original_snapshot")
        original_branch = journal.get("original_branch")
        maintenance_branch = journal.get("maintenance_branch", "hermes-release")
        was_checkout_restored = bool(journal.get("checkout_restored"))
        candidate_promotion_incomplete = False
        if (
            "post_cas_snapshot" in journal
            and "candidate_snapshot" not in journal
            and _transaction_is_promoted(git_cmd, root, context)
        ):
            post_cas_snapshot = _release_context_snapshot(context, "post_cas_snapshot")
            candidate_sha = journal.get("candidate_sha")
            if not isinstance(candidate_sha, str) or _SHA_RE.fullmatch(candidate_sha) is None:
                raise ReleaseGitStateError("Promotion recovery candidate SHA is malformed.")
            candidate_tree_sha = _release_git_sha(
                git_cmd,
                root,
                ["rev-parse", "--verify", f"{candidate_sha}^{{tree}}"],
                label="promotion recovery candidate tree",
            )
            candidate_snapshot = _release_clean_snapshot(
                post_cas_snapshot,
                symbolic_head=maintenance_branch,
                head_sha=candidate_sha,
                maintenance_ref_sha=candidate_sha,
                head_tree_sha=candidate_tree_sha,
            )
            candidate_promotion_incomplete = True
        if not was_checkout_restored:
            # A promoted checkout on the maintenance branch is already the
            # correct clean state; do not replay even a same-branch checkout.
            needs_checkout = (
                candidate_promotion_incomplete
                or (
                    candidate_snapshot is not None
                    and original_branch != maintenance_branch
                )
            ) or (
                candidate_snapshot is None
                and original_branch != maintenance_branch
            ) or (
                candidate_snapshot is None
                and "maintenance_snapshot" in journal
            )
            if needs_checkout:
                if guard_identity_only:
                    _validate_release_restored_snapshot_identity(
                        git_cmd,
                        root,
                        guard_snapshot,
                        label="immediate pre-original-checkout restoration",
                    )
                else:
                    _validate_release_git_snapshot(
                        git_cmd,
                        root,
                        guard_snapshot,
                        label="immediate pre-original-checkout restoration",
                    )
                _restore_original_release_checkout(git_cmd, root, context)
            restore_clean_snapshot = _release_expected_original_clean_snapshot(
                original_snapshot,
                candidate_snapshot,
            )
            try:
                _validate_release_git_snapshot(
                    git_cmd,
                    root,
                    restore_clean_snapshot,
                    label="pre-stash-apply checkout",
                )
            except Exception as exc:
                _release_mark_manual_interference(context, f"finalization incomplete: {exc}")
                print(f"  Recovery journal retained at {context.journal_path}")
                return False
            _release_snapshot_with_journal(
                context,
                "restore_clean_snapshot",
                restore_clean_snapshot,
                checkout_restored=True,
            )
            was_checkout_restored = True
        else:
            restore_clean_snapshot = _release_context_snapshot(
                context, "restore_clean_snapshot"
            )

        final_snapshot: ReleaseGitSnapshot = restore_clean_snapshot
        if stash_pending:
            assert stash_sha is not None
            if not journal.get("stash_apply_attempted"):
                if (
                    subprocess.run(
                        git_cmd + ["cat-file", "-e", f"{stash_sha}^{{commit}}"],
                        cwd=root,
                        capture_output=True,
                    ).returncode
                    != 0
                ):
                    raise RuntimeError(
                        f"Immutable stash {stash_sha} is no longer reachable; "
                        f"journal: {context.journal_path}"
                    )
                # This is the last read-only guard before the first necessary
                # live mutator, stash apply.  The preceding checkout was
                # separately verified against the clean restore snapshot.
                _validate_release_git_snapshot(
                    git_cmd,
                    root,
                    restore_clean_snapshot,
                    label="immediate pre-stash-apply checkout",
                )
                _journal_update(context, stash_apply_attempted=True)
                restored = _restore_stashed_changes(
                    git_cmd,
                    root,
                    stash_sha,
                    prompt_user=False,
                    input_fn=input_fn,
                    restore_index=True,
                    drop_stash=False,
                    preserve_conflict_state=True,
                )
                if not restored:
                    _journal_update(context, "stash-restore-conflict")
                    print(f"  Recovery journal retained at {context.journal_path}")
                    _print_stash_cleanup_guidance(
                        stash_sha, _resolve_stash_selector(git_cmd, root, stash_sha)
                    )
                    return False
                try:
                    restored_snapshot = _validate_release_restored_snapshot_identity(
                        git_cmd,
                        root,
                        restore_clean_snapshot,
                        label="post-stash-apply restoration",
                    )
                except Exception as exc:
                    _release_mark_manual_interference(
                        context, f"finalization incomplete: {exc}"
                    )
                    print(f"  Recovery journal retained at {context.journal_path}")
                    return False
                _release_snapshot_with_journal(
                    context,
                    "restored_snapshot",
                    restored_snapshot,
                    stash_applied=True,
                )
                final_snapshot = restored_snapshot
            else:
                final_snapshot = _release_context_snapshot(context, "restored_snapshot")
                try:
                    _validate_release_restored_snapshot_identity(
                        git_cmd,
                        root,
                        final_snapshot,
                        label="pre-stash-drop restoration",
                    )
                except Exception as exc:
                    _release_mark_manual_interference(
                        context, f"finalization incomplete: {exc}"
                    )
                    print(f"  Recovery journal retained at {context.journal_path}")
                    return False

            # Stash deletion is allowed only after the captured user-state
            # snapshot still has the same immutable checkout identity.
            selector = _resolve_stash_selector(git_cmd, root, stash_sha)
            _validate_release_restored_snapshot_identity(
                git_cmd,
                root,
                final_snapshot,
                label="immediate pre-stash-drop restoration",
            )
            if selector is not None:
                drop = subprocess.run(
                    git_cmd + ["stash", "drop", selector],
                    cwd=root,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                if drop.returncode != 0:
                    print(
                        "⚠ Local changes were restored, but the immutable stash "
                        "could not be dropped."
                    )
                    _print_stash_cleanup_guidance(stash_sha, selector)
                    _journal_update(context, "stash-drop-failed")
                    return False
                if _resolve_stash_selector(git_cmd, root, stash_sha) is not None:
                    _journal_update(context, "stash-drop-unverified")
                    return False
            _journal_update(context, stash_pending=False)
        else:
            if journal.get("stash_applied"):
                final_snapshot = _release_context_snapshot(context, "restored_snapshot")
                _validate_release_restored_snapshot_identity(
                    git_cmd,
                    root,
                    final_snapshot,
                    label="final restored checkout",
                )
            else:
                _validate_release_git_snapshot(
                    git_cmd,
                    root,
                    final_snapshot,
                    label="final restored checkout",
                )
            _journal_update(context, stash_pending=False)

        # Candidate evidence is retained until user bytes/index/status are
        # verified restored.  Manual interference returned above before this
        # point, so cleanup cannot erase evidence after a mismatch.
        if journal.get("stash_applied"):
            _validate_release_restored_snapshot_identity(
                git_cmd,
                root,
                final_snapshot,
                label="pre-candidate-cleanup restored checkout",
            )
        else:
            _validate_release_git_snapshot(
                git_cmd,
                root,
                final_snapshot,
                label="pre-candidate-cleanup checkout",
            )
        candidate_cleanup_failed_before_restore = (
            prior_phase == "candidate-cleanup-failed"
            and not prior_stash_applied
        )
        if candidate_promotion_incomplete or (
            journal.get("candidate_created") and "candidate_snapshot" not in journal
        ):
            _journal_update(
                context,
                "promotion-incomplete",
                candidate_cleanup=False,
            )
            logger.warning(
                "Release promotion did not reach a verified candidate; recovery journal kept at %s",
                context.journal_path,
            )
            return False
        if candidate_cleanup_failed_before_restore:
            _journal_update(
                context,
                "candidate-cleanup-failed",
                candidate_cleanup=False,
            )
            logger.warning(
                "Release candidate cleanup remains incomplete; recovery journal kept at %s",
                context.journal_path,
            )
            return False
        if journal.get("candidate_created") and not journal.get("candidate_cleanup"):
            if not _release_cleanup_candidate(git_cmd, root, context):
                _journal_update(context, "candidate-cleanup-failed", candidate_cleanup=False)
                logger.warning(
                    "Release candidate cleanup failed; recovery journal kept at %s",
                    context.journal_path,
                )
                return False
            _journal_update(context, "candidate-cleanup", candidate_cleanup=True)

        if journal.get("stash_applied"):
            _validate_release_restored_snapshot_identity(
                git_cmd,
                root,
                final_snapshot,
                label="terminal restored checkout",
            )
        else:
            _validate_release_git_snapshot(
                git_cmd,
                root,
                final_snapshot,
                label="terminal restored checkout",
            )
        _mark_release_finalized(context)
        # Mark the verified terminal state before acknowledging journal removal.
        # Any retry now takes the filesystem-only path above.
        return _acknowledge_finalized_release(context)
    except (KeyboardInterrupt, SystemExit):
        # Preserve process-control exceptions after the terminal write latch has
        # captured the candidate and kept the journal as recovery evidence.
        raise
    except BaseException as exc:
        # A second Ctrl-C must not erase the only recovery evidence.  Preserve
        # the journal and let an already-active exception continue propagating.
        logger.warning(
            "Release user-state finalization did not complete; journal retained at %s: %s",
            context.journal_path,
            exc,
        )
        print(f"⚠ Release user-state recovery is incomplete; journal: {context.journal_path}")
        return False


def _finalize_release_upgrade_for_orchestration(
    git_cmd: list[str],
    cwd: Path | str,
    context: ReleaseUpgradeContext,
    *,
    input_fn=None,
    primary_exc_info=None,
) -> bool:
    """Apply finalization precedence at the release orchestration boundary.

    The transaction finalizer owns durable cleanup and its detailed warning.
    This boundary is the sole owner of converting an otherwise-successful
    ``False`` result into a command failure.  When a primary exception is
    already active, every finalizer failure is logged and suppressed so the
    original exception (including process-control exceptions) remains the one
    that propagates.
    """

    primary = primary_exc_info[1] if primary_exc_info is not None else None
    journal_path = getattr(context, "journal_path", "<unknown>")
    try:
        finalized = _finalize_release_upgrade(
            git_cmd,
            cwd,
            context,
            input_fn=input_fn,
        )
    except (KeyboardInterrupt, SystemExit) as exc:
        if primary is None:
            # With no primary operation failure, finalizer process-control
            # semantics are authoritative and must not be converted to a
            # normal upgrade error.
            raise
        logger.warning(
            "Release finalizer raised %s while preserving the primary exception; "
            "journal retained at %s",
            type(exc).__name__,
            journal_path,
        )
        return False
    except BaseException as exc:
        if primary is None:
            raise ReleaseFinalizationIncompleteError(journal_path) from exc
        logger.warning(
            "Release finalizer raised %s while preserving the primary exception; "
            "journal retained at %s",
            type(exc).__name__,
            journal_path,
        )
        return False

    if finalized:
        return True
    if primary is not None:
        logger.warning(
            "Release finalization incomplete; preserving the primary exception; "
            "journal retained at %s",
            journal_path,
        )
        return False
    raise ReleaseFinalizationIncompleteError(journal_path)


_UPDATE_RUNTIME_RELOAD_MODULES = (
    "hermes_constants",
    "tools.environments.local",
    "tools.lazy_deps",
)

def _reload_updated_runtime_modules() -> None:
    """Reload update-sensitive modules after the checkout changes in-place.

    ``hermes update`` keeps running in the pre-pull Python process. After a
    large update, modules already present in ``sys.modules`` can still expose
    old symbols even though their source files on disk are new. Refresh the
    small module set used by lazy-backend refresh before that step imports
    newly-updated code paths.
    """
    try:
        import importlib

        importlib.invalidate_caches()
        for module_name in _UPDATE_RUNTIME_RELOAD_MODULES:
            module = _m().sys.modules.get(module_name)
            if module is None:
                continue
            try:
                importlib.reload(module)
            except Exception as exc:
                logger.debug("Could not reload updated module %s: %s", module_name, exc)
    except Exception as exc:
        logger.debug("Could not refresh update runtime modules: %s", exc)

def _reload_config_modules() -> None:
    """Force-reload modules from disk after git pull.

    ``hermes update`` runs in the PRE-pull Python process. After ``git pull``
    updates the source files on disk, modules already in ``sys.modules``
    still hold the OLD code. Function-level imports return the cached module,
    so ``DEFAULT_CONFIG["_config_version"]`` is the OLD value and
    ``check_config_version()`` reports ``(33, 33)`` — "up to date" — even
    though the freshly-pulled code has v34 with a migration to run.

    This function force-reloads ``hermes_cli.config_defaults``,
    ``hermes_cli.config``, and ``hermes_cli.config_migrations`` from disk
    so subsequent imports read the UPDATED code.

    It also reloads ``hermes_cli._subprocess_compat`` and
    ``hermes_cli.dashboard_procs`` so that post-update dashboard cleanup
    (``_finish_dashboard_update_cleanup`` → ``_scan_dashboard_processes``)
    uses the freshly-pulled code. Without this, a new symbol added to
    ``_subprocess_compat`` (e.g. ``bounded_probe_run``) is invisible to the
    cached module object, causing ``ImportError`` during the cleanup step
    that runs later in the same process.
    """
    import importlib

    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli.config_defaults",
        "hermes_cli.config",
        "hermes_cli.config_migrations",
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                logger.debug("Could not reload %s for fresh post-update code: %s", mod_name, exc)


def _run_config_check_fresh() -> tuple:
    """Check config version using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns ``(current_ver, latest_ver)``.
    """
    _reload_config_modules()
    from hermes_cli.config import check_config_version

    return check_config_version()


def _run_migrate_config_fresh(*, interactive: bool = False, quiet: bool = False) -> dict:
    """Run config migration using freshly-reloaded modules.

    See ``_reload_config_modules`` for why this is necessary.
    Returns the migration results dict.
    """
    _reload_config_modules()
    from hermes_cli.config import migrate_config

    return migrate_config(interactive=interactive, quiet=quiet)


# Critical files that Hermes must be able to import immediately after an
# update/install. Most are imported on every CLI startup; ``web_server.py``
# is the desktop/dashboard backend path that a fresh Windows install launches
# right away. If any of these fail to parse after a pull, the user can be
# left with a bricked CLI or desktop backend. The post-pull syntax guard
# validates these and auto-rolls-back on failure.
_UPDATE_CRITICAL_FILES = (
    "hermes_cli/main.py",
    "hermes_cli/update_cmd.py",
    "hermes_cli/config.py",
    "hermes_cli/__init__.py",
    "hermes_cli/web_server.py",
    "cli.py",
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "hermes_constants.py",
)

def _capture_head_sha(git_cmd, cwd) -> str | None:
    """Return the current HEAD SHA, or None if it can't be resolved."""
    try:
        result = subprocess.run(
            git_cmd + ["rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=True,
        )
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


_LOCAL_PATCHES_DIRNAME = "local-patches"
_LOCAL_PATCHES_README = """# Hermes local release patches

These artifacts are maintained on the ``hermes-release`` branch.

``hermes upgrade`` captures user changes before any cleanup, generates the
maintenance payload from committed Git history (not this file), incrementally
merges the new upstream release into the long-lived maintenance history in a
temporary candidate worktree, validates the candidate, and only then promotes
it.  Git rerere is enabled for the isolated merge so a previously recorded
resolution can be reused.  The generated patch is a portable, byte-safe
snapshot for inspection and recovery; it is never the upgrade input or sole
source of truth.

The payload excludes this directory so the series cannot patch itself.  The
JSON ``.release_base`` file records the integration mode, human release tag,
immutable upstream base SHA, and patch hash/size.  A missing or stale patch
artifact therefore cannot make a committed local change disappear.

## Regenerate after editing local customizations

Run ``hermes upgrade``.  It atomically refreshes this directory after the
candidate has been committed and validated.  Manual edits to these artifacts
are stashed and restored like every other user edit.

## Keep out of this series

- Mem0 sync_turns / infer_turns / api_url → ``$HERMES_HOME/plugins/mem0-local``
  with ``memory.provider: mem0-local`` in config.yaml
"""


def _repo_local_patches_dir(cwd: Path | str | None = None) -> Path:
    """In-repo ``local-patches/`` on the hermes-release branch."""
    root = Path(cwd) if cwd is not None else Path(_m().PROJECT_ROOT)
    return root / _LOCAL_PATCHES_DIRNAME


def _home_local_patches_dir() -> Path:
    """Legacy ``$HERMES_HOME/local-patches`` (migration / emergency fallback)."""
    return Path(get_hermes_home()) / _LOCAL_PATCHES_DIRNAME


def _patch_series_signature(patches: list[Path]) -> tuple[tuple[str, str], ...]:
    """Return ordered patch names and byte hashes without exposing contents."""
    signature = []
    for path in patches:
        try:
            payload = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"Could not read local patch {path}: {exc}") from exc
        signature.append((path.name, hashlib.sha256(payload).hexdigest()))
    return tuple(signature)


def _select_local_patch_files(cwd: Path | str | None = None) -> list[Path]:
    """Select one compatible repo/home patch series or fail closed on conflict."""
    root = Path(cwd) if cwd is not None else Path(_m().PROJECT_ROOT)
    repo_dir = _repo_local_patches_dir(root)
    home_dir = _home_local_patches_dir()
    repo_patches = list(_iter_patch_files(repo_dir))
    home_patches = list(_iter_patch_files(home_dir))

    if repo_patches and home_patches:
        if _patch_series_signature(repo_patches) != _patch_series_signature(home_patches):
            raise RuntimeError(
                "Conflicting local patch series found in "
                f"{repo_dir} and {home_dir}. Migrate to one location, or make "
                "both series byte-identical, then re-run the update."
            )
        return repo_patches
    return repo_patches or home_patches


def _local_patches_dir(cwd: Path | str | None = None) -> Path:
    """Prefer the repo patch series, checking legacy home state for conflicts."""
    selected = _select_local_patch_files(cwd)
    if selected:
        return selected[0].parent
    return _repo_local_patches_dir(cwd)


def _iter_patch_files(patches_dir: Path):
    if not patches_dir.is_dir():
        return
    for path in sorted(patches_dir.iterdir()):
        try:
            info = path.lstat()
        except OSError:
            continue
        if (
            stat.S_ISREG(info.st_mode)
            and path.suffix == ".patch"
            and not path.name.startswith(".")
            and info.st_size > 0
        ):
            yield path


def _list_local_release_patches(cwd: Path | str | None = None) -> list[Path]:
    """Sorted ``*.patch`` files used to reapply local customizations after a release reset."""
    return _select_local_patch_files(cwd)


def _snapshot_local_release_patches(cwd: Path | str) -> list[tuple[str, str]]:
    """Read the selected patch series before ``reset --hard`` removes its copies."""
    snapshots: list[tuple[str, str]] = []
    for path in _select_local_patch_files(cwd):
        try:
            content = path.read_text(encoding="utf-8")
        except OSError as exc:
            raise RuntimeError(f"Could not read local patch {path}: {exc}") from exc
        if not content.strip():
            continue
        snapshots.append((path.name, content))
    return snapshots


def _git_working_tree_dirty(git_cmd, cwd) -> bool:
    result = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return bool(result.stdout.strip())


def _apply_patch_text(git_cmd, cwd, name: str, content: str) -> None:
    """Apply one patch from an in-memory snapshot via a temp file."""
    import tempfile

    with tempfile.TemporaryDirectory(prefix="hermes-local-patch-") as tmp:
        patch_path = Path(tmp) / name
        patch_path.write_text(content, encoding="utf-8")
        print(f"→ Applying local patch {name}...")
        apply_result = subprocess.run(
            git_cmd
            + [
                "apply",
                "--3way",
                "--whitespace=nowarn",
                str(patch_path),
            ],
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        if apply_result.returncode != 0:
            apply_result = subprocess.run(
                git_cmd + ["apply", "--whitespace=nowarn", str(patch_path)],
                cwd=cwd,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
            )
        if apply_result.returncode != 0:
            err = (apply_result.stderr or apply_result.stdout or "").strip()
            detail = err.splitlines()[0] if err else "git apply failed"
            raise RuntimeError(
                f"Could not apply local patch {name}: {detail}. "
                f"Update files under {_LOCAL_PATCHES_DIRNAME}/ against the new "
                "release, then re-run: hermes upgrade"
            )


def _refresh_local_release_patch_file(
    git_cmd,
    cwd,
    release_tag: str,
    *,
    patch_body: str,
) -> Path:
    """Write the versioned ``local-patches/`` directory for ``patch_body``."""
    patches_dir = _repo_local_patches_dir(cwd)
    patches_dir.mkdir(parents=True, exist_ok=True)
    primary = patches_dir / "0001-local-maintenance.patch"
    primary.write_text(patch_body, encoding="utf-8")
    (patches_dir / "README.md").write_text(_LOCAL_PATCHES_README, encoding="utf-8")
    (patches_dir / ".release_base").write_text(f"{release_tag}\n", encoding="utf-8")
    for stale in _iter_patch_files(patches_dir):
        if stale.resolve() != primary.resolve():
            stale.unlink(missing_ok=True)
    try:
        home_dir = _home_local_patches_dir()
        home_dir.mkdir(parents=True, exist_ok=True)
        (home_dir / primary.name).write_text(patch_body, encoding="utf-8")
        (home_dir / "README.md").write_text(_LOCAL_PATCHES_README, encoding="utf-8")
        (home_dir / ".release_base").write_text(f"{release_tag}\n", encoding="utf-8")
    except OSError:
        pass
    return primary


def _diff_local_payload_against_release(git_cmd, cwd, release_tag: str) -> str:
    """Diff release tag vs current index/worktree, excluding ``local-patches/``."""
    # Stage payload so newly created files (from git apply) are included.
    add_payload = subprocess.run(
        git_cmd
        + [
            "add",
            "-A",
            "--",
            ".",
            f":!({_LOCAL_PATCHES_DIRNAME})",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if add_payload.returncode != 0:
        raise RuntimeError("Failed to stage reapplied local patch payload.")

    diff_result = subprocess.run(
        git_cmd
        + [
            "diff",
            "--cached",
            release_tag,
            "--",
            ".",
            f":!({_LOCAL_PATCHES_DIRNAME})",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=True,
    )
    return diff_result.stdout


def _apply_local_release_patches(
    git_cmd,
    cwd,
    release_tag: str,
    *,
    patch_snapshots: list[tuple[str, str]] | None = None,
) -> None:
    """Reapply local patches after reset, commit, and refresh the in-repo series.

    Customizations stay out of upstream merge history: upgrade resets to the
    release tag, replays these patches, and commits onto ``hermes-release``.
    """
    snapshots = patch_snapshots
    if snapshots is None:
        snapshots = _snapshot_local_release_patches(cwd)
    if not snapshots:
        print(
            f"→ No local patches under {_LOCAL_PATCHES_DIRNAME}/ "
            "(or $HERMES_HOME/local-patches); staying on clean release."
        )
        return

    for name, content in snapshots:
        _apply_patch_text(git_cmd, cwd, name, content)

    try:
        patch_body = _diff_local_payload_against_release(git_cmd, cwd, release_tag)
        _refresh_local_release_patch_file(
            git_cmd, cwd, release_tag, patch_body=patch_body
        )
    except (OSError, subprocess.CalledProcessError, RuntimeError) as exc:
        raise RuntimeError(f"Failed to refresh local patch series: {exc}") from exc

    add_patches = subprocess.run(
        git_cmd + ["add", "-A", "--", _LOCAL_PATCHES_DIRNAME],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if add_patches.returncode != 0:
        raise RuntimeError("Failed to stage refreshed local-patches directory.")

    cached = subprocess.run(
        git_cmd + ["diff", "--cached", "--quiet"],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if cached.returncode == 0:
        print("→ Local patches applied with no tree changes.")
        return

    commit_result = subprocess.run(
        git_cmd
        + [
            "commit",
            "--no-gpg-sign",
            "-m",
            f"local: reapply patches onto {release_tag}",
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if commit_result.returncode != 0:
        err = (commit_result.stderr or commit_result.stdout or "").strip()
        detail = err.splitlines()[0] if err else "git commit failed"
        raise RuntimeError(f"Failed to commit reapplied local patches: {detail}")

    print(f"→ Refreshed in-repo local-patches against {release_tag}.")


def _upgrade_release_with_local_patches(git_cmd, cwd, release_tag: str) -> subprocess.CompletedProcess:
    """Reset hermes-release to ``release_tag``, then reapply local patches."""
    # Snapshot before reset — in-repo local-patches disappear with --hard.
    try:
        snapshots = _snapshot_local_release_patches(cwd)
    except RuntimeError as exc:
        return subprocess.CompletedProcess(
            ["snapshot-local-patches"],
            returncode=1,
            stdout="",
            stderr=str(exc),
        )

    reset_result = subprocess.run(
        git_cmd
        + [
            "-c",
            "gpg.ssh.allowedSignersFile=/dev/null",
            "reset",
            "--hard",
            release_tag,
        ],
        cwd=cwd,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    if reset_result.returncode != 0:
        return reset_result

    try:
        _apply_local_release_patches(
            git_cmd, cwd, release_tag, patch_snapshots=snapshots
        )
    except RuntimeError as exc:
        return subprocess.CompletedProcess(
            reset_result.args,
            returncode=1,
            stdout=reset_result.stdout,
            stderr=str(exc),
        )
    return subprocess.CompletedProcess(
        reset_result.args,
        returncode=0,
        stdout=reset_result.stdout,
        stderr=reset_result.stderr,
    )


def _validate_critical_files_syntax(root) -> tuple[bool, str | None, str | None]:
    """Strictly compile every required production file in a candidate.

    Release-candidate validation is fail-closed: every path in
    ``_UPDATE_CRITICAL_FILES`` must exist, be a contained regular file, and
    compile successfully.  Keep this strict entry point separate from the
    compatibility guard used after an ordinary update.
    """
    return _validate_critical_files_syntax_impl(root, allow_missing=False)


def _validate_post_pull_critical_files_syntax(
    root,
) -> tuple[bool, str | None, str | None]:
    """Compile existing critical files after an ordinary post-pull update.

    Older installations can legitimately predate one of the current critical
    paths.  Missing files are therefore tolerated here, but every path that
    does exist still receives the same regular-file, containment, and syntax
    checks as strict candidate validation.
    """
    return _validate_critical_files_syntax_impl(root, allow_missing=True)


def _validate_critical_files_syntax_impl(
    root, *, allow_missing: bool
) -> tuple[bool, str | None, str | None]:
    """Shared syntax guard implementation with an explicit missing-file mode."""
    import py_compile

    root = Path(root)
    try:
        candidate_root = root.resolve(strict=True)
    except OSError as exc:
        return False, str(root), f"candidate root is not accessible: {exc}"

    with tempfile.TemporaryDirectory(prefix="hermes-syntax-check-") as tmpdir:
        for relpath in _UPDATE_CRITICAL_FILES:
            relative = Path(relpath)
            path = root / relative
            if relative.is_absolute() or ".." in relative.parts:
                return False, str(path), "critical path escapes candidate root"
            try:
                info = path.lstat()
            except FileNotFoundError:
                if allow_missing:
                    continue
                return False, str(path), "required critical file is missing"
            except OSError as exc:
                return False, str(path), f"could not inspect required file: {exc}"
            if stat.S_ISLNK(info.st_mode):
                return False, str(path), "required critical file is a symlink"
            if not stat.S_ISREG(info.st_mode):
                return False, str(path), "required critical file is not a regular file"
            try:
                resolved = path.resolve(strict=True)
                resolved.relative_to(candidate_root)
            except ValueError:
                return False, str(path), "required critical file resolves outside candidate root"
            except OSError as exc:
                return False, str(path), f"could not resolve required file: {exc}"

            # Mirror the relative path under the temp directory so different
            # source files cannot collide in the compile output.
            cfile = Path(tmpdir) / (str(relative).replace("/", "__") + "c")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, str(path), str(exc)
            except OSError as exc:
                return False, str(path), f"could not compile: {exc}"
    return True, None, None


# Modules imported on every agent startup. Unlike _UPDATE_CRITICAL_FILES (which
# is only parsed), these are actually *imported* so that cross-module breakage
# is caught — a file can be syntactically perfect and still fail to import
# because a name it pulls from a sibling module no longer exists.
_UPDATE_CRITICAL_MODULES = (
    "hermes_cli.update_cmd",
    "hermes_cli.main",
    "run_agent",
    "model_tools",
    "toolsets",
)


def _validate_critical_modules_import(root) -> tuple[bool, str | None, str | None]:
    """Import every critical module from the candidate in an isolated process.

    A missing dependency is deferred to the existing dependency-sync stage only
    when it is an explicitly named, non-first-party module. Every other import
    or probe failure remains fatal before candidate promotion.
    """

    root = Path(root)
    try:
        candidate_root = root.resolve(strict=True)
    except OSError as exc:
        return False, "candidate import probe", f"candidate root is not accessible: {exc}"

    # Inject the canonical root set into the child probe. The parent still uses
    # is_first_party_module() for names such as hermes_constants, whose
    # first-party family is represented by that canonical helper as well.
    first_party_roots = tuple(sorted(FIRST_PARTY_MODULE_ROOTS))
    probe = (
        "import importlib, sys, traceback\n"
        "from pathlib import Path\n"
        f"_root = Path({str(candidate_root)!r})\n"
        "_root = _root.resolve(strict=True)\n"
        "sys.path.insert(0, str(_root))\n"
        "_FIRST_PARTY_MODULE_ROOTS = frozenset(%r)\n"
        "for _name in %r:\n"
        "    try:\n"
        "        _module = importlib.import_module(_name)\n"
        "        _origin = getattr(_module, '__file__', None)\n"
        "        if not _origin:\n"
        "            raise RuntimeError('critical module has no __file__')\n"
        "        _resolved = Path(_origin).resolve(strict=True)\n"
        "        try:\n"
        "            _resolved.relative_to(_root)\n"
        "        except ValueError as _exc:\n"
        "            raise RuntimeError(\n"
        "                f'critical module imported from outside candidate: {_resolved}'\n"
        "            ) from _exc\n"
        "    except ModuleNotFoundError as _exc:\n"
        "        _missing = getattr(_exc, 'name', None) or ''\n"
        "        _missing_root = _missing.split('.', 1)[0]\n"
        "        if _missing_root in _FIRST_PARTY_MODULE_ROOTS:\n"
        "            _marker = 'HERMES_FIRST_PARTY_MISSING:'\n"
        "        else:\n"
        "            _marker = 'HERMES_MODULE_NOT_FOUND:'\n"
        "        print(_marker + _name + ':' + _missing)\n"
        "        traceback.print_exc()\n"
        "        continue\n"
        "    except BaseException:\n"
        "        print('HERMES_IMPORT_FAILURE:' + _name)\n"
        "        traceback.print_exc()\n"
        "        raise\n"
        "print('HERMES_IMPORT_OK')\n"
        % (first_party_roots, _UPDATE_CRITICAL_MODULES)
    )
    try:
        interpreter = sys.executable
        try:
            venv_python = venv_python_path(
                root / "venv", windows=_m()._is_windows()
            )
            if venv_python.exists():
                interpreter = str(venv_python)
        except Exception:
            pass  # fall back to the running interpreter
        result = subprocess.run(
            [interpreter, "-I", "-c", probe],
            cwd=str(candidate_root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return False, "candidate import probe", f"could not execute import probe: {exc}"

    stdout_lines = (result.stdout or "").splitlines()
    output = "\n".join(
        part for part in ((result.stdout or "").strip(), (result.stderr or "").strip()) if part
    )
    if result.returncode != 0 or "HERMES_IMPORT_OK" not in stdout_lines:
        module = "candidate import probe"
        for line in stdout_lines:
            if line.startswith("HERMES_IMPORT_FAILURE:"):
                module = line.partition(":")[2].strip() or module
                break
        detail = output or f"import probe exited with status {result.returncode}"
        return False, module, detail

    missing: list[tuple[str, str, bool]] = []
    for line in stdout_lines:
        if line.startswith("HERMES_FIRST_PARTY_MISSING:"):
            payload = line.partition(":")[2]
            module, _, missing_name = payload.partition(":")
            missing.append((module, missing_name, True))
        elif line.startswith("HERMES_MODULE_NOT_FOUND:"):
            payload = line.partition(":")[2]
            module, _, missing_name = payload.partition(":")
            missing.append((module, missing_name, False))

    for module, missing_name, marked_first_party in missing:
        if marked_first_party or not missing_name or is_first_party_module(missing_name):
            return False, module or "candidate import probe", output or missing_name

    # A clearly third-party ModuleNotFoundError is deferred, but retain the
    # child traceback so callers that expose diagnostics can report it.
    if missing:
        return True, None, output or None
    return True, None, None


def _gateway_prompt(prompt_text: str, default: str = "", timeout: float = 300.0) -> str:
    """File-based IPC prompt for gateway mode.

    Writes a prompt marker file so the gateway can forward the question to the
    user, then polls for a response file.  Falls back to *default* on timeout.

    Used by ``hermes update --gateway`` so interactive prompts (stash restore,
    config migration) are forwarded to the messenger instead of being silently
    skipped.
    """
    import json as _json
    import uuid as _uuid
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    prompt_path = home / ".update_prompt.json"
    response_path = home / ".update_response"

    # Clean any stale response file
    response_path.unlink(missing_ok=True)

    payload = {
        "prompt": prompt_text,
        "default": default,
        "id": str(_uuid.uuid4()),
    }
    tmp = prompt_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload), encoding="utf-8")
    tmp.replace(prompt_path)

    # Poll for response
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if response_path.exists():
            try:
                answer = response_path.read_text(encoding="utf-8").strip()
                response_path.unlink(missing_ok=True)
                prompt_path.unlink(missing_ok=True)
                return answer if answer else default
            except (OSError, ValueError):
                pass
        _time.sleep(0.5)

    # Timeout — clean up and use default
    prompt_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    print(f"  (no response after {int(timeout)}s, using default: {default!r})")
    return default

def _npm_bin_exists(bin_dir: Path, name: str) -> bool:
    """True when an npm bin shim for *name* exists (POSIX or Windows)."""
    return any(
        (bin_dir / candidate).exists()
        for candidate in (name, f"{name}.cmd", f"{name}.ps1", f"{name}.exe")
    )

def _web_build_toolchain_ready(*roots: Path) -> bool:
    """True when ``tsc`` and ``vite`` shims are reachable from any of *roots*.

    Callers must pass every root the build would search; checking only one
    reports a healthy tree as broken.
    """
    bin_dirs = [
        bin_dir
        for bin_dir in (root / "node_modules" / ".bin" for root in roots)
        if bin_dir.is_dir()
    ]
    return bool(bin_dirs) and all(
        any(_npm_bin_exists(bin_dir, tool) for bin_dir in bin_dirs)
        for tool in ("tsc", "vite")
    )

def _web_toolchain_roots(web_dir: Path) -> tuple[Path, ...]:
    """Roots whose ``node_modules/.bin`` can satisfy the web build.

    ``npm run build`` prepends ``node_modules/.bin`` for the package and each
    of its ancestors, so shims hoisted to the workspace root and shims nested
    under a package that owns its lockfile (#42973) are equally valid.
    """
    return (web_dir, web_dir.parent)

def _print_curator_first_run_notice() -> None:
    """Print a short heads-up about the skill curator after `hermes update`.

    Only fires when the curator is enabled AND has no recorded run yet, which
    is exactly the window where the gateway ticker used to fire Curator
    against a fresh skill library immediately after an update. We defer the
    first real pass by one ``interval_hours``; this notice tells the user how
    to preview or disable before then. Silent on steady state.
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        if not curator.is_enabled():
            return
        state = curator.load_state()
    except Exception:
        return
    if state.get("last_run_at"):
        # Curator has run before (real or already seeded) — no notice needed.
        return
    try:
        hours = curator.get_interval_hours()
    except Exception:
        hours = 24 * 7
    days = max(1, hours // 24)
    print()
    print("ℹ Skill curator")
    print(
        f"  Background skill maintenance is enabled. First pass is deferred "
        f"~{days}d after installation; only agent-created skills are in "
        f"scope and nothing is ever auto-deleted (archive is recoverable)."
    )
    print("  Preview now:  hermes curator run --dry-run")
    print("  Pause it:     hermes curator pause")
    print(
        "  Docs:         https://hermes-agent.nousresearch.com/docs/user-guide/features/curator"
    )

def _print_fts_optimize_available_notice() -> None:
    """Advertise the opt-in v23 search-index optimization after `hermes update`.

    Only fires when the current profile's state.db is still on the legacy
    (pre-v23) inline FTS layout. Leads with the reclaimable-space figure and
    points at the exact command. Honors ``sessions.fts_optimize_notice``:
    ``advise`` (default) prints an advisory notice, ``require`` prints a
    firmer required-upgrade notice, ``off`` suppresses it. Silent for
    fresh/already-optimized installs.
    """
    mode = "advise"
    try:
        from hermes_cli.config import load_config

        mode = str(
            ((load_config() or {}).get("sessions") or {}).get(
                "fts_optimize_notice", "advise"
            )
        ).strip().lower()
    except Exception:
        mode = "advise"
    if mode == "off":
        return

    try:
        from hermes_constants import get_hermes_home
        from hermes_state import SessionDB
    except Exception:
        return
    db_path = get_hermes_home() / "state.db"
    if not db_path.exists():
        return
    try:
        size_gb = db_path.stat().st_size / (1024 ** 3)
    except OSError:
        return
    # Skip the notice for trivially small DBs — the win isn't worth the nag.
    if size_gb < 0.5:
        return
    db = None
    interrupted = False
    try:
        db = SessionDB(db_path=db_path, read_only=True)
        # read_only opens skip schema init, so probe the layout directly.
        row = db._conn.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name = 'messages_fts'"
        ).fetchone()
        # An interrupted `optimize-storage` run: the table is already the
        # v23 shape, but backfill markers / demoted trash tables remain.
        # Offer the command again — re-running resumes and finishes it.
        interrupted = bool(
            db._conn.execute(
                "SELECT 1 FROM state_meta "
                "WHERE key = 'fts_rebuild_high_water' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' "
                "AND name LIKE 'fts\\_v22\\_trash\\_%' ESCAPE '\\' LIMIT 1"
            ).fetchone()
            or db._conn.execute(
                "SELECT 1 FROM state_meta WHERE key IN "
                "('fts_cjk_rebuild_high_water', 'fts_cjk_stale') LIMIT 1"
            ).fetchone()
        )
    except Exception:
        return
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass
    sql = (row[0] if row else "") or ""
    if not sql or ("tool_name" in sql and not interrupted):
        # v23 layout already present (fresh/optimized) — nothing to offer.
        return

    if interrupted:
        print()
        print("◆ Session database optimization incomplete")
        print(
            "  A previous `hermes sessions optimize-storage` run was "
            "interrupted. Search still works; re-run the command to resume "
            "and finish reclaiming disk:"
        )
        print("    hermes sessions optimize-storage")
        return

    # Concrete size framing — lead with the savings the user cares about.
    est_reclaim = size_gb * 0.6
    print()
    if mode == "require":
        print("◆ Session database upgrade required")
        print(
            f"  Your search index uses the OLD storage layout and should be "
            f"upgraded. The new layout typically frees ~60% of state.db "
            f"(≈{est_reclaim:.1f} GB of your current {size_gb:.1f} GB) and is "
            f"required for continued optimal operation."
        )
    else:
        print("◆ Reclaim ~60% of your session database disk")
        print(
            f"  Your search index uses the old storage layout. Upgrading it "
            f"typically frees ~60% of state.db — about {est_reclaim:.1f} GB "
            f"of your current {size_gb:.1f} GB."
        )
    print("  Run when convenient:  hermes sessions optimize-storage")
    print(
        "  It runs in the foreground with a progress bar, is safe to "
        "interrupt/re-run, and never changes your conversations."
    )

def _print_curator_recent_run_notice() -> None:
    """Print the most recent curator run summary, exactly once.

    The curator runs in the background (gateway tick + CLI session start),
    so users learn about skill consolidations only by stumbling into a
    rename. ``hermes update`` is a high-attention surface — surface the
    most recent run's rename map here, once.

    Show-once: state stamps ``last_run_summary_shown_at`` after printing.
    Subsequent ``hermes update`` invocations skip the block until a newer
    curator run lands. Silent when the curator has never run, when the
    most recent summary has already been shown, or when the summary has
    no rename information to display (no archives).
    """
    try:
        from agent import curator
    except Exception:
        return
    try:
        state = curator.load_state()
    except Exception:
        return

    last_run_at = state.get("last_run_at")
    if not last_run_at:
        return  # no curator run yet — first-run notice handles this case

    if state.get("last_run_summary_shown_at") == last_run_at:
        return  # already shown for this run

    summary = state.get("last_run_summary") or ""
    if not summary:
        return

    # Only print when there's something interesting to show — i.e. the
    # rename map block was appended (multi-line summary). A bare "auto:
    # no changes; llm: no change" doesn't warrant interrupting the
    # update flow.
    if "\n" not in summary:
        # Still stamp it shown so we don't reconsider it on every update.
        try:
            state["last_run_summary_shown_at"] = last_run_at
            curator.save_state(state)
        except Exception:
            pass
        return

    # Format the timestamp as "Xh ago" for readability.
    when = _format_time_ago(last_run_at)
    print()
    print(f"ℹ Skill curator — last run {when}")
    for line in summary.splitlines():
        print(f"  {line}")
    print(
        "  (This message shows once per curator run. "
        "View anytime: hermes curator status)"
    )

    # Stamp shown so we don't repeat on the next update.
    try:
        state["last_run_summary_shown_at"] = last_run_at
        curator.save_state(state)
    except Exception:
        pass

def _format_time_ago(iso_ts: str) -> str:
    """Render an ISO timestamp as `Xh ago` / `Xd ago` / `Xm ago`. Best effort."""
    try:
        from datetime import datetime, timezone
        ts = datetime.fromisoformat(iso_ts.replace("Z", "+00:00"))
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        delta = datetime.now(timezone.utc) - ts
        secs = int(delta.total_seconds())
        if secs < 60:
            return "just now"
        if secs < 3600:
            return f"{secs // 60}m ago"
        if secs < 86400:
            return f"{secs // 3600}h ago"
        return f"{secs // 86400}d ago"
    except Exception:
        return "recently"

def _reload_process_scan_modules() -> None:
    """Force-reload the process-scan modules from disk after an update.

    ``_finish_dashboard_update_cleanup`` runs in the PRE-update Python
    process, but ``_scan_dashboard_processes`` does a function-level
    ``from hermes_cli._subprocess_compat import bounded_probe_run``. If the
    update added a new symbol to ``_subprocess_compat`` (as #87134 did with
    ``bounded_probe_run``), the cached OLD module object doesn't have it and
    the cleanup step crashes with ImportError — after the code update itself
    already succeeded. Reload dependency-first so ``dashboard_procs`` binds
    against the fresh ``_subprocess_compat``.

    Lives here (called from the cleanup entry point) rather than only in
    ``_reload_config_modules`` so EVERY caller — the git-update path, the
    Windows ZIP fallback path, and any future one — is covered.
    """
    import importlib

    importlib.invalidate_caches()
    for mod_name in (
        "hermes_cli._subprocess_compat",
        "hermes_cli.dashboard_procs",
    ):
        mod = sys.modules.get(mod_name)
        if mod is not None:
            try:
                importlib.reload(mod)
            except Exception as exc:
                # warning, not debug: a failed reload here surfaces seconds
                # later as an ImportError in the same process — leave a trail.
                logger.warning(
                    "Could not reload %s for post-update cleanup: %s",
                    mod_name,
                    exc,
                )


def _finish_dashboard_update_cleanup(
    node_failures: list[str], already_restarted_units: "set[str] | None" = None
) -> None:
    """Refresh managed dashboards or stop stale manual ones after an update.

    *already_restarted_units* forwards the systemd unit names (no
    ``.service`` suffix) that the fleet-restart loop already restarted
    directly, so a Serve-only install's freshly restarted process isn't
    found and restarted a second time here (review on #83595).
    """
    if node_failures:
        print()
        print("  ℹ Leaving running dashboard process(es) untouched because the")
        print("    Node.js dependency refresh did not complete.")
        return

    # The scan path lazy-imports symbols from _subprocess_compat; make sure
    # both modules reflect the freshly-updated source before touching them.
    _reload_process_scan_modules()

    stop_result = _m()._kill_stale_dashboard_processes(
        restart_managed=True, already_restarted_units=already_restarted_units
    )
    if not stop_result.get("unrecovered"):
        return

    print()
    print(
        "⚠ A web dashboard/serve process was stopped during update and could "
        "not be auto-restarted."
    )
    print("  Re-launch it when you want the web UI back:")
    print("    hermes dashboard --port <port>")

def _atomic_replace_dir(src: str, dst: str) -> None:
    """Replace directory *dst* with *src* without leaving *dst* half-deleted.

    The naive ``rmtree(dst); copytree(src, dst)`` has a destructive window: if
    the copy fails partway (common on the Windows ZIP-update path, which only
    runs because file I/O is already flaky on that machine), the old directory
    is already gone and nothing replaced it — the install is left with a
    deleted tree (issue #49145, where ``ui-tui/`` vanished and broke the TUI).

    Now a thin single-entry alias over the two-phase helpers below, which
    generalise the same stage-then-swap discipline across every entry the ZIP
    update touches (#76104). Retained because it is part of the mechanical
    ``hermes_cli.main`` re-export surface and guards the #49145 regression.
    """
    _commit_staged_replacements([(_stage_replacement(src, dst), dst)])


def _stage_replacement(src: str, dst: str) -> str:
    """Copy *src* to a sibling staging path for *dst*; return the staging path.

    Phase 1 of the two-phase replace. Handles both directories and plain
    files. Touches nothing live, so a failure here leaves the whole install
    untouched.
    """
    staging = f"{dst}.hermes-update-staging"
    backup = f"{dst}.hermes-update-old"
    # A previous run may have died between "move dst aside" and "move staging
    # in" — leaving dst missing and the backup as the ONLY copy of that entry.
    # Restore it before clearing leftovers: deleting the backup first and then
    # failing to stage (disk exhaustion is likely right after writing a full
    # staging copy) would leave a hole in the install with nothing to roll
    # back to. The restore is a same-filesystem rename — instant and safe.
    if not os.path.exists(dst) and os.path.exists(backup):
        os.rename(backup, dst)
    for leftover in (staging, backup):
        if os.path.isdir(leftover):
            shutil.rmtree(leftover, ignore_errors=True)
        elif os.path.exists(leftover):
            os.remove(leftover)
    if os.path.isdir(src):
        shutil.copytree(src, staging)
    else:
        shutil.copy2(src, staging)
    return staging


def _discard_staged(staged) -> None:
    """Remove staging paths for entries that were never committed.

    Without this a phase-1 failure (typically disk exhaustion) orphans one
    staging copy per entry already processed — up to a full second copy of
    the tree. The user then follows the "re-run `hermes update`" advice with
    *less* free space than before and the retry fails harder than the
    original attempt.
    """
    for staging, _dst in staged:
        try:
            if os.path.isdir(staging):
                shutil.rmtree(staging, ignore_errors=True)
            elif os.path.exists(staging):
                os.remove(staging)
        except OSError as exc:  # best-effort cleanup, never fatal
            logger.warning("could not remove staging path %s: %s", staging, exc)


def _commit_staged_replacements(staged) -> None:
    """Phase 2: swap every staged entry into place, rolling back all on failure.

    ``_atomic_replace_dir`` makes each *individual* directory swap safe, but
    the ZIP update replaces ~90 top-level entries in a loop, and nothing made
    the loop atomic *as a whole*. A failure partway left some entries at the
    new version and the rest at the old one — every file valid Python, the
    combination unbootable (issue #76104; the ``ImportError`` in #76091 and
    the field report in #63717 are both this).

    This covers plain files as well as directories: the repo root holds 20
    first-party modules (``run_agent.py``, ``cli.py``, ``hermes_constants.py``
    …), so a files-only failure reproduces exactly the bug class we are
    closing. Every swap is an ``os.rename`` onto a path that was just moved
    aside — a same-filesystem rename is atomic on POSIX and NTFS alike, so a
    file swap can never leave a half-written module the way ``copy2`` onto a
    live path can.

    Splitting stage-all-then-swap-all shrinks the failure window from "the
    duration of a full tree copy" to "the duration of N renames", and makes
    the remaining window recoverable: if a swap fails we restore every entry
    already swapped, so the tree lands wholly new or wholly old.
    """
    swapped: list[tuple[str, str]] = []  # (dst, backup) in swap order; "" = absent
    try:
        for staging, dst in staged:
            backup = f"{dst}.hermes-update-old"
            if os.path.exists(dst):
                os.rename(dst, backup)
                swapped.append((dst, backup))
            else:
                swapped.append((dst, ""))
            os.rename(staging, dst)
    except OSError:
        # Undo every swap already made so the install stays self-consistent.
        for dst, backup in reversed(swapped):
            try:
                if os.path.isdir(dst):
                    shutil.rmtree(dst, ignore_errors=True)
                elif os.path.exists(dst):
                    os.remove(dst)
                if backup and os.path.exists(backup):
                    os.rename(backup, dst)
            except OSError as exc:
                # Keep restoring the rest — a silent failure here is the one
                # thing that turns a recoverable rollback into a mixed tree,
                # so say so rather than swallowing it.
                logger.warning("rollback failed for %s: %s", dst, exc)
        raise
    # All swaps succeeded — drop the backups (best-effort, never fatal).
    for _dst, backup in swapped:
        if backup and os.path.isdir(backup):
            shutil.rmtree(backup, ignore_errors=True)
        elif backup and os.path.exists(backup):
            try:
                os.remove(backup)
            except OSError:
                pass


def _branch_head_label(git_cmd=None, cwd=None) -> str | None:
    """``"<branch> @ <short-sha>"`` for the checkout, or None when unknown.

    Appended to the update summary lines so branch drift is visible at a
    glance (live incident 2026-08-17: a checkout parked on a stale feature
    branch got "✓ Update complete!" with nothing on the line saying WHERE
    the checkout actually sat). Never raises — summary decoration must not
    break an update.
    """
    try:
        cmd = list(git_cmd) if git_cmd else ["git"]
        root = cwd if cwd is not None else _m().PROJECT_ROOT
        branch = subprocess.run(
            cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        sha = subprocess.run(
            cmd + ["rev-parse", "--short", "HEAD"],
            cwd=root, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        branch_name = branch.stdout.strip()
        sha_text = sha.stdout.strip()
        if branch.returncode != 0 or sha.returncode != 0 or not sha_text:
            return None
        if not branch_name:
            return None
        label = "detached" if branch_name == "HEAD" else branch_name
        return f"{label} @ {sha_text}"
    except Exception:
        return None


def _branch_head_suffix(git_cmd=None, cwd=None) -> str:
    """`` [<branch> @ <sha>]`` suffix for summary lines ("" when unknown)."""
    label = _branch_head_label(git_cmd, cwd)
    return f" [{label}]" if label else ""


def _assess_parked_branch_switch(
    git_cmd: list[str], cwd: Path, current_branch: str, target_branch: str
) -> tuple[bool, str]:
    """Decide whether it is safe to auto-switch a parked feature branch back
    to the update target.

    Live incident (2026-08-17, Teknium's box): the source checkout sat on a
    stale feature branch left behind by earlier tooling; ``hermes update``
    autostashed, ran its post-update steps and printed "✓ Code updated!"
    while the running code stayed days behind main. The guard's contract:

    - safe (True, "") only when the working tree + index are clean AND every
      commit on the parked branch is already contained in
      ``origin/<target_branch>`` (``git cherry`` reports no ``+`` lines).
    - anything else — dirty tree, unmerged commits, git errors, or the
      ``updates.auto_switch_parked_branch: false`` config opt-out — returns
      (False, <reason>) and the caller must NOT touch the branch.

    Reasons: "disabled", "dirty", "unmerged:<count>", "unverifiable".
    """
    try:
        from hermes_cli.config import load_config

        _update_cfg = (load_config() or {}).get("updates", {})
        if isinstance(_update_cfg, dict) and not bool(
            _update_cfg.get("auto_switch_parked_branch", True)
        ):
            return False, "disabled"
    except Exception as exc:
        # A config read failure must not disable the guard's safety checks —
        # fall through to them with the default (auto-switch allowed).
        logger.debug("Could not read updates.auto_switch_parked_branch: %s", exc)

    status = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=cwd, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if status.returncode != 0:
        return False, "unverifiable"
    if status.stdout.strip():
        return False, "dirty"

    cherry = subprocess.run(
        git_cmd + ["cherry", f"origin/{target_branch}"],
        cwd=cwd, capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if cherry.returncode != 0:
        return False, "unverifiable"
    unmerged = [
        line for line in cherry.stdout.splitlines() if line.startswith("+")
    ]
    if unmerged:
        return False, f"unmerged:{len(unmerged)}"
    return True, ""


def _print_parked_branch_skip_warning(
    git_cmd: list[str],
    cwd: Path,
    current_branch: str,
    target_branch: str,
    reason: str,
) -> None:
    """LOUD block explaining why the code update was skipped on a parked
    branch, with the behind-count and the exact commands to resolve."""
    behind = None
    try:
        behind_result = subprocess.run(
            git_cmd + ["rev-list", f"HEAD..origin/{target_branch}", "--count"],
            cwd=cwd, capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if behind_result.returncode == 0 and behind_result.stdout.strip():
            behind = int(behind_result.stdout.strip())
    except Exception:
        behind = None

    if reason == "dirty":
        why = "the working tree has uncommitted changes"
    elif reason.startswith("unmerged:"):
        count = reason.split(":", 1)[1]
        why = (
            f"the branch has {count} commit(s) not merged into "
            f"origin/{target_branch}"
        )
    elif reason == "disabled":
        why = "updates.auto_switch_parked_branch is set to false in config.yaml"
    else:
        why = (
            f"the branch state could not be verified against "
            f"origin/{target_branch}"
        )

    bar = "=" * 68
    print()
    print(bar)
    print(f"⚠ CODE UPDATE SKIPPED — checkout is parked on '{current_branch}'")
    print(f"  Not auto-switching to {target_branch}: {why}.")
    if behind is not None and behind > 0:
        print(
            f"  This checkout is {behind} commit(s) BEHIND "
            f"origin/{target_branch} — the code you are running is stale."
        )
    print()
    print("  To resolve, inspect the branch and switch back yourself:")
    print(f"    git -C {cwd} status")
    print(f"    git -C {cwd} checkout {target_branch} && hermes update")
    print(
        "  (commit or stash your work on the branch first if you want to "
        "keep it)"
    )
    print(bar)


def _print_update_completion(message: str) -> None:
    """Print an update outcome plus, when the dashboard launched this run
    with an action id, a terminal receipt line the Desktop can match after
    the dashboard restarts (see #47359 / #58764).

    The outcome line carries the checkout's actual branch + HEAD short-sha
    so branch drift is visible at a glance (2026-08-17 parked-branch
    incident)."""
    print(f"{message}{_branch_head_suffix()}")
    action_id = os.environ.get("HERMES_ACTION_ID", "")
    if len(action_id) == 32 and all(char in "0123456789abcdef" for char in action_id):
        print(f"=== hermes-update completed {action_id} ===")


def _read_project_version() -> str | None:
    """Read the ``version`` field from the checkout's pyproject.toml.

    Reads the on-disk file (not importlib.metadata) because after a git
    pull the installed distribution metadata still describes the OLD
    version; the file is the only source that reflects what was just
    pulled. Returns None on any failure — version reporting is cosmetic
    and must never break an update.
    """
    try:
        import tomllib

        with open(_m().PROJECT_ROOT / "pyproject.toml", "rb") as fh:  # windows-footgun: ok — binary mode, tomllib requires bytes
            version = tomllib.load(fh).get("project", {}).get("version")
        return str(version) if version else None
    except Exception:
        return None


def _update_complete_message(pre_version: str | None) -> str:
    """Completion line with the version transition when it is known.

    Ported from PrimeIntellect-ai/prime-agent#630: after a successful
    self-update, show both versions (``v0.19.4 → v0.20.0``) so the user
    can see what they actually got. Falls back to the plain message when
    either side is unknown or the version did not change (e.g. several
    commits landed within one release).
    """
    post_version = _read_project_version()
    if pre_version and post_version and pre_version != post_version:
        return f"✓ Update complete! (v{pre_version} → v{post_version})"
    if post_version:
        return f"✓ Update complete! (v{post_version})"
    return "✓ Update complete!"


def _update_via_zip(args, *, had_desktop_app_before_update: bool = False):
    """Update Hermes Agent by downloading a ZIP archive.

    Used on Windows when git file I/O is broken (antivirus, NTFS filter
    drivers causing 'Invalid argument' errors on file creation).
    """
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    import tempfile
    import zipfile
    from urllib.request import urlretrieve

    # Snapshot the pre-update version before files are replaced so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()

    # The ZIP fallback exists for Windows git-file-I/O breakage. It pulls a
    # static archive from GitHub, which is fine for the default "main"
    # channel but would silently ignore --branch and update from main even
    # if the user asked for something else — exactly the silent-divergence
    # bug --branch was added to prevent. Refuse to proceed in that case
    # rather than lie.
    branch = _m()._resolve_update_branch(args)
    if branch != "main":
        print(
            f"✗ --branch={branch} is not supported on the Windows ZIP-fallback "
            "update path."
        )
        print(
            "  This path runs when git file I/O is broken on the system. "
            "Either resolve the git-side breakage (typically an antivirus "
            "or NTFS filter holding files open) and rerun `hermes update "
            f"--branch {branch}`, or update against main with `hermes update`."
        )
        _m().sys.exit(1)
    zip_url = (
        f"https://github.com/NousResearch/hermes-agent/archive/refs/heads/{branch}.zip"
    )

    print("→ Downloading latest version...")
    tmp_dir = tempfile.mkdtemp(prefix="hermes-update-")
    try:
        zip_path = os.path.join(tmp_dir, f"hermes-agent-{branch}.zip")
        urlretrieve(zip_url, zip_path)

        print("→ Extracting...")
        import stat as _stat
        with zipfile.ZipFile(zip_path, "r") as zf:
            # Validate paths to prevent zip-slip (path traversal) AND reject
            # symlink members. A GitHub source ZIP for hermes-agent itself
            # should never contain symlinks — they'd point outside the
            # extracted tree and let an attacker who can compromise the
            # update mirror plant arbitrary files via the update path.
            tmp_dir_real = os.path.realpath(tmp_dir)
            for member in zf.infolist():
                member_path = os.path.realpath(os.path.join(tmp_dir, member.filename))
                if (
                    not member_path.startswith(tmp_dir_real + os.sep)
                    and member_path != tmp_dir_real
                ):
                    raise ValueError(
                        f"Zip-slip detected: {member.filename} escapes extraction directory"
                    )
                # Unix mode lives in the upper 16 bits of external_attr;
                # mask to the file-type bits.
                mode = (member.external_attr >> 16) & 0o170000
                if _stat.S_ISLNK(mode):
                    raise ValueError(
                        f"ZIP contains unsupported symlink member: {member.filename}"
                    )
            zf.extractall(tmp_dir)

        # GitHub ZIPs extract to hermes-agent-<branch>/
        extracted = os.path.join(tmp_dir, f"hermes-agent-{branch}")
        if not os.path.isdir(extracted):
            # Try to find it
            for d in os.listdir(tmp_dir):
                candidate = os.path.join(tmp_dir, d)
                if os.path.isdir(candidate) and d != "__MACOSX":
                    extracted = candidate
                    break

        # Copy updated files over existing installation, preserving venv/node_modules/.git
        preserve = {"venv", "node_modules", ".git", ".env"}
        entries = [i for i in os.listdir(extracted) if i not in preserve]

        # Two-phase replace (#76104). Phase 1 copies every entry — directories
        # AND top-level files — to a sibling staging path without touching
        # anything live; phase 2 swaps them all in with same-filesystem
        # renames and rolls back every swap if any one fails. Replacing
        # entries one-at-a-time (the previous shape) meant an interruption
        # partway left `agent/` new and `tools/` stale — all files valid, the
        # tree unbootable. Files matter as much as directories here: the repo
        # root holds 20 first-party modules (run_agent.py, cli.py,
        # hermes_constants.py, ...).
        #
        # Staging costs one extra copy of the tree on disk. Check up front so
        # we fail with a clear message instead of running out mid-copy.
        need = sum(
            os.path.getsize(os.path.join(dirpath, f))
            for entry in entries
            for dirpath, _dirs, files in os.walk(os.path.join(extracted, entry))
            for f in files
        ) + sum(
            os.path.getsize(os.path.join(extracted, e))
            for e in entries
            if os.path.isfile(os.path.join(extracted, e))
        )
        # Only the staging copy is new — the live tree already occupies its
        # space and the swaps are renames, not copies. Ask for the staging
        # copy plus 20% headroom rather than a full 2x, which would block
        # updates that would have succeeded on exactly the space-constrained
        # machines most likely to hit this path.
        required = int(need * 1.2)
        free = shutil.disk_usage(str(_m().PROJECT_ROOT)).free
        if free < required:
            raise RuntimeError(
                f"not enough free disk space to stage the update safely "
                f"(need ~{required // (1024 * 1024)} MB, have "
                f"{free // (1024 * 1024)} MB)"
            )

        staged: list[tuple[str, str]] = []
        try:
            for item in entries:
                src = os.path.join(extracted, item)
                dst = os.path.join(str(_m().PROJECT_ROOT), item)
                staged.append((_stage_replacement(src, dst), dst))
        except Exception:
            # Nothing is live yet; drop the partial staging copies so a retry
            # starts from the same free space this attempt did.
            _discard_staged(staged)
            raise

        try:
            _commit_staged_replacements(staged)
        except Exception:
            # The rollback already restored every swapped entry, but staging
            # copies for the not-yet-swapped entries (potentially most of a
            # full tree) are still on disk. Drop them, or the retry's
            # up-front free-space check — which runs BEFORE the lazy
            # per-entry leftover cleanup — fails on litter this attempt
            # left behind: the exact "retry fails harder" failure mode
            # _discard_staged exists to prevent. Safe post-rollback: swapped
            # entries' staging paths were renamed away, and _discard_staged
            # skips paths that no longer exist.
            _discard_staged(staged)
            raise
        update_count = len(staged)

        print(f"✓ Updated {update_count} items from ZIP")

    except Exception as e:
        print(f"✗ ZIP update failed: {e}")
        # The two-phase replace either commits every entry or rolls them all
        # back, so a failure here does not leave a mixed-version tree — don't
        # scare the user toward a reinstall they don't need.
        print("  Your existing install was left in place.")
        print(
            "  Re-run `hermes update` to retry; if the agent won't start, "
            "reinstall from https://hermes-agent.nousresearch.com"
        )
        _m().sys.exit(1)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    # Clear stale bytecode after ZIP extraction
    removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
    if removed:
        print(
            f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
        )
    _m()._record_bytecode_fingerprint()
    _m()._refresh_bootstrap_cache_scripts(branch)

    # Reinstall Python dependencies. Prefer .[all], but if one optional extra
    # breaks on this machine, keep base deps and reinstall the remaining extras
    # individually so update does not silently strip working capabilities.
    #
    # Self-lock deferral (relocated preflight — #86735): the ZIP code swap
    # above is already committed; defer only the dependency sync when this
    # process holds a native extension the sync must rewrite.
    _m()._abort_dependency_sync_if_self_locked()
    print("→ Updating Python dependencies...")

    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current — runs `uv self update` if we already have one.
    update_managed_uv()

    uv_bin = ensure_uv()

    pip_cmd = [_m().sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    if uv_bin:
        uv_env = {**os.environ, "VIRTUAL_ENV": str(_m().PROJECT_ROOT / "venv")}
        if _m()._is_termux_env(uv_env):
            uv_env.pop("PYTHONPATH", None)
            uv_env.pop("PYTHONHOME", None)
        _m()._install_python_dependencies_with_optional_fallback([uv_bin, "pip"], env=uv_env)
    else:
        # Use sys.executable to explicitly call the venv's pip module,
        # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
        # Some environments lose pip inside the venv; bootstrap it back with
        # ensurepip before trying the editable install.
        try:
            subprocess.run(
                pip_cmd + ["--version"],
                cwd=_m().PROJECT_ROOT,
                check=True,
                capture_output=True,
            )
        except subprocess.CalledProcessError:
            subprocess.run(
                [_m().sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                cwd=_m().PROJECT_ROOT,
                check=True,
            )
        _m()._install_python_dependencies_with_optional_fallback(pip_cmd)

    install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
    install_env = uv_env if uv_bin else None
    _m()._restore_active_tool_dependencies(
        active_tool_dependencies,
        install_prefix,
        env=install_env,
    )

    # ZIP path parity: heal the active memory provider's bridge packages
    # after the dependency reinstall, same as the git-pull path (#53272,
    # #70636).
    _m()._refresh_active_memory_provider_dependencies()

    # Now that dependencies are installed, verify the tree actually imports.
    # The copy loop above replaces top-level entries one at a time in
    # os.listdir order, so an interruption between (say) `agent/` and `tools/`
    # leaves a tree whose files all parse but cannot be imported together —
    # the ImportError-on-startup class this guard exists to catch. Deliberately
    # placed *after* the dependency reinstall so a genuinely-new third-party
    # requirement isn't misreported as a partial copy. There is no SHA to roll
    # back to here, so surface it with a concrete recovery step rather than
    # reporting a successful update over a bricked install.
    import_ok, failing_module, import_error = _validate_critical_modules_import(
        _m().PROJECT_ROOT
    )
    if not import_ok:
        print()
        print("✗ Update left the install in an unimportable state:")
        print(f"  {failing_module}: {import_error}")
        print()
        print("  This usually means the copy was interrupted partway through.")
        print("  Re-run `hermes update` to complete it.")
        _m().sys.exit(1)

    node_failures = _update_node_dependencies()
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update,
    )

    # Sync skills
    try:
        from tools.skills_sync import sync_skills

        print("→ Syncing bundled skills...")
        result = sync_skills(quiet=True)
        if result["copied"]:
            print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
        if result.get("updated"):
            print(
                f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
            )
        if result.get("user_modified"):
            print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
            print(
                "    → see them: hermes skills list-modified  "
                "(diff/reset to resume updates)"
            )
        if result.get("cleaned"):
            print(f"  − {len(result['cleaned'])} removed from manifest")
        if result.get("relocated"):
            print(
                f"  → {len(result['relocated'])} moved to new upstream paths: "
                f"{', '.join(result['relocated'])}"
            )
        if not result["copied"] and not result.get("updated"):
            print("  ✓ Skills are up to date")
    except Exception:
        pass

    # Seed the model-catalog disk cache from the freshly-unpacked checkout
    # (same rationale as the git-pull path in _cmd_update_impl). Non-fatal.
    try:
        from hermes_cli.model_catalog import seed_cache_from_checkout

        if seed_cache_from_checkout(_m().PROJECT_ROOT):
            print("  ✓ Model catalog cache refreshed from checkout")
    except Exception as e:
        logger.debug("Model catalog seed during zip update failed: %s", e)

    # ── Post-update state.db integrity guard (#68474) ─────────────────
    # Same as the git-pull path: verify state.db survived the ZIP update
    # and auto-restore from the most recent pre-update snapshot if needed.
    try:
        from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity

        _state_path = get_hermes_home() / "state.db"
        if _state_path.exists():
            _state_ok = verify_sqlite_integrity(
                _state_path, check_header=True, run_pragma=True
            )
            if not _state_ok.get("valid"):
                print()
                print(
                    "⚠ state.db is corrupted after update: "
                    + _state_ok.get("message", "unknown error")
                )
                _snap_root = _quick_snapshot_root(get_hermes_home())
                if _snap_root.exists():
                    _snap_dirs = sorted(
                        (d for d in _snap_root.iterdir() if d.is_dir()),
                        reverse=True,
                    )
                    for _snap_dir in _snap_dirs:
                        _snap_state = _snap_dir / "state.db"
                        if _snap_state.exists():
                            _snap_ok = verify_sqlite_integrity(
                                _snap_state, check_header=True, run_pragma=True
                            )
                            if _snap_ok.get("valid"):
                                try:
                                    import shutil as _shutil

                                    _shutil.copy2(_snap_state, _state_path)
                                    _restored_ok = verify_sqlite_integrity(
                                        _state_path,
                                        check_header=True,
                                        run_pragma=True,
                                    )
                                    if _restored_ok.get("valid"):
                                        print(
                                            "  ✓ Auto-restored from snapshot "
                                            f"{_snap_dir.name}"
                                        )
                                    else:
                                        print(
                                            "  ✗ Auto-restore FAILED — restored "
                                            "copy also failed integrity"
                                        )
                                    break
                                except OSError as _exc:
                                    print(
                                        f"  ✗ Auto-restore file copy failed: {_exc}"
                                    )
                                    break
    except Exception as exc:
        logger.debug(
            "Post-update state.db integrity check (zip path) failed: %s", exc
        )

    print()
    if node_failures:
        print(
            "⚠ Update partially complete — Node.js dependencies for "
            f"{', '.join(node_failures)} did not refresh."
        )
        print("  Code and Python deps are updated, but the dashboard/TUI may")
        print("  be in a mixed state until the Node deps are rebuilt.")
    else:
        _print_update_completion(_update_complete_message(pre_update_version))
    try:
        _print_curator_first_run_notice()
    except Exception as e:
        logger.debug("Curator first-run notice failed: %s", e)
    try:
        _print_curator_recent_run_notice()
    except Exception as e:
        logger.debug("Curator recent-run notice failed: %s", e)
    # Don't stop a working dashboard when the Node refresh failed — see the
    # git-update path for rationale (#30271).
    _finish_dashboard_update_cleanup(node_failures)

def _stash_local_changes_if_needed(
    git_cmd: list[str], cwd: Path, *, marker: str | None = None
) -> Optional[str]:
    # An unmerged index contains stage 1/2/3 entries whose identity must survive
    # an update refusal.  Never reset, stash, or otherwise mutate an unresolved
    # merge/rebase state; require the user to resolve it first.
    unmerged = subprocess.run(
        git_cmd + ["ls-files", "--unmerged"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if unmerged.returncode != 0:
        print("✗ Could not inspect the Git index safely; update aborted.")
        print("  Resolve conflicts or inspect the repository manually, then re-run the update.")
        raise RuntimeError(
            "Could not inspect the Git index safely; resolve conflicts before updating."
        )
    if unmerged.stdout.strip():
        print("✗ Unresolved Git index conflicts detected; update aborted.")
        print("  Resolve conflicts in the working tree, then re-run the update.")
        raise RuntimeError(
            "Unresolved Git index conflicts detected; resolve conflicts before updating."
        )

    status = subprocess.run(
        git_cmd + ["status", "--porcelain"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    if not status.stdout.strip():
        return None

    stash_name = marker or (
        f"hermes-update-autostash-{os.getpid()}-{uuid.uuid4().hex}"
    )
    print("→ Local changes detected — stashing before update...")
    push = subprocess.run(
        git_cmd + ["stash", "push", "--include-untracked", "-m", stash_name],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if push.stdout.strip():
        print(push.stdout.strip())
    stash_matches: list[str] = []
    stash_list = subprocess.run(
        git_cmd + ["stash", "list", "--format=%H%x00%gs"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if stash_list.returncode == 0:
        for line in stash_list.stdout.splitlines():
            commit, separator, subject = line.partition("\0")
            commit = commit.strip()
            if separator and stash_name in subject and _SHA_RE.fullmatch(commit):
                stash_matches.append(commit.lower())
    stash_created = len(stash_matches) == 1

    if push.returncode != 0:
        if push.stderr.strip():
            print(push.stderr.strip())
        if stash_created:
            print(
                "  ⚠ The stash command failed after creating a uniquely marked "
                f"entry; it remains preserved as immutable stash {stash_matches[0]}."
            )
            if marker is None:
                # Preserve ordinary update behavior: Git may leave tracked
                # edits in place after saving the stash while an untracked
                # path cannot be removed.  The ordinary caller's existing
                # restore path expects a clean checkout window here.
                subprocess.run(
                    git_cmd + ["reset", "--hard", "HEAD"],
                    cwd=cwd,
                    capture_output=True,
                )
        else:
            print("✗ Could not stash local changes — update aborted.")
            print(
                "  Commit, stash, or clean up your local changes manually, "
                "then re-run `hermes update`."
            )
            # Never reset/clean here.  The transaction boundary will either
            # durably confirm this exact SHA or retain an uncertain-capture journal.
            raise subprocess.CalledProcessError(
                push.returncode, push.args, output=push.stdout, stderr=push.stderr
            )

    if not stash_created:
        print("✗ Could not identify one exact autostash entry — update aborted.")
        raise RuntimeError(
            "The local changes may not be safely stashed; inspect `git stash list` "
            "and recover before retrying."
        )
    return stash_matches[0]

def _resolve_stash_selector(
    git_cmd: list[str], cwd: Path, stash_ref: str
) -> Optional[str]:
    stash_list = subprocess.run(
        git_cmd + ["stash", "list", "--format=%gd %H"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    for line in stash_list.stdout.splitlines():
        selector, _, commit = line.partition(" ")
        if commit.strip() == stash_ref:
            return selector.strip()
    return None

def _print_stash_cleanup_guidance(
    stash_ref: str, stash_selector: Optional[str] = None
) -> None:
    print(
        "  Check `git status` first so you don't accidentally reapply the same change twice."
    )
    print("  Find the saved entry with: git stash list --format='%gd %H %s'")
    if stash_selector:
        print(f"  Remove it with: git stash drop {stash_selector}")
    else:
        print(
            f"  Look for commit {stash_ref}, then drop its selector with: git stash drop stash@{{N}}"
        )

def _stash_apply_failed_only_on_existing_untracked(stderr: str) -> bool:
    """True when a ``git stash apply`` failure is ONLY about untracked files
    that already exist in the working tree.

    This is the tail end of the permission-denied autostash class: ``git stash
    push --include-untracked`` swept undeletable files (e.g. a root-owned
    ``packaging/`` directory) into the stash but could not remove them from
    disk.  On restore, git applies all tracked changes, then refuses to
    overwrite those still-present files (``already exists, no checkout`` /
    ``could not restore untracked files from stash``) and exits non-zero even
    though nothing was lost.  Any other error line (e.g. ``would be
    overwritten by merge`` / ``Aborting``) means the tracked apply itself
    failed and this returns False.
    """
    lines = [ln.strip() for ln in (stderr or "").splitlines() if ln.strip()]
    if not lines:
        return False
    saw_untracked_error = False
    for ln in lines:
        if "already exists, no checkout" in ln:
            saw_untracked_error = True
        elif "could not restore untracked files from stash" in ln:
            saw_untracked_error = True
        elif ln.startswith(("warning:", "hint:")):
            continue
        else:
            return False
    return saw_untracked_error

def _restore_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
    prompt_user: bool = False,
    input_fn=None,
    restore_index: bool = False,
    drop_stash: bool = True,
    preserve_conflict_state: bool = False,
) -> bool:
    if prompt_user:
        print()
        print("⚠ Local changes were stashed before updating.")
        print(
            "  Restoring them may reapply local customizations onto the updated codebase."
        )
        print("  Review the result afterward if Hermes behaves unexpectedly.")
        print("Restore local changes now? [Y/n]")
        if input_fn is not None:
            response = input_fn("Restore local changes now? [Y/n]", "y")
        else:
            try:
                response = input().strip().lower()
            except (EOFError, UnicodeDecodeError):
                # Mirror the config-migration prompt's fix: don't let a
                # terminal-encoding issue or a closed stdin crash the
                # update mid-restore. Falls through to the existing
                # skip-restore path below, which already explains how to
                # restore manually from git stash.
                response = "n"
        if response not in {"", "y", "yes"}:
            print("Skipped restoring local changes.")
            print("Your changes are still preserved in git stash.")
            print(f"Restore manually with: git stash apply {stash_ref}")
            return False

    print("→ Restoring local changes...")
    stash_apply = git_cmd + ["stash", "apply"]
    if restore_index:
        stash_apply.append("--index")
    stash_apply.append(stash_ref)
    restore = subprocess.run(
        stash_apply,
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )

    # Check for unmerged (conflicted) files — can happen even when returncode is 0
    unmerged = subprocess.run(
        git_cmd + ["diff", "--name-only", "--diff-filter=U"],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    has_conflicts = bool(unmerged.stdout.strip())

    if restore.returncode != 0 and not has_conflicts and (
        _stash_apply_failed_only_on_existing_untracked(restore.stderr)
    ):
        # Permission-denied autostash tail end: the tracked changes applied
        # cleanly; the only "failure" is untracked files that never left the
        # working tree (git could not delete them at stash time, so it now
        # refuses to overwrite them). Their content was never touched —
        # nothing is lost. Treat as restored.
        print(
            "  ⚠ Some stashed untracked files already exist in the working "
            "tree and were kept as-is."
        )
    elif restore.returncode != 0 or has_conflicts:
        print("✗ Update pulled new code, but restoring local changes hit conflicts.")
        if restore.stdout.strip():
            print(restore.stdout.strip())
        if restore.stderr.strip():
            print(restore.stderr.strip())

        # Show which files conflicted
        conflicted_files = unmerged.stdout.strip()
        if conflicted_files:
            print("\nConflicted files:")
            for f in conflicted_files.splitlines():
                print(f"  • {f}")

        print("\nYour stashed changes are preserved — nothing is lost.")
        print(f"  Stash ref: {stash_ref}")

        if not preserve_conflict_state:
            # Ordinary update recovery retains its historical behavior.  The
            # release transaction opts into preserving the conflict state so
            # finalization never replays a destructive reset.
            subprocess.run(
                git_cmd + ["reset", "--hard", "HEAD"],
                cwd=cwd,
                capture_output=True,
            )
            print("Working tree reset to clean state.")
        else:
            print("  Conflict state is preserved for manual resolution.")
        print(f"Restore your changes later with: git stash apply {stash_ref}")
        # Don't sys.exit — the code update itself succeeded, only the stash
        # restore had conflicts.  Let cmd_update continue with pip install,
        # skill sync, and gateway restart.
        return False

    if not drop_stash:
        print("⚠ Local changes were restored on top of the updated codebase.")
        print("  Review `git diff` / `git status` if Hermes behaves unexpectedly.")
        return True

    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Local changes were restored, but Hermes couldn't find the stash entry to drop."
        )
        print(
            "  The stash was left in place. You can remove it manually after checking the result."
        )
        _print_stash_cleanup_guidance(stash_ref)
    else:
        drop = subprocess.run(
            git_cmd + ["stash", "drop", stash_selector],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if drop.returncode != 0:
            print(
                "⚠ Local changes were restored, but Hermes couldn't drop the saved stash entry."
            )
            if drop.stdout.strip():
                print(drop.stdout.strip())
            if drop.stderr.strip():
                print(drop.stderr.strip())
            print(
                "  The stash was left in place. You can remove it manually after checking the result."
            )
            _print_stash_cleanup_guidance(stash_ref, stash_selector)

    print("⚠ Local changes were restored on top of the updated codebase.")
    print("  Review `git diff` / `git status` if Hermes behaves unexpectedly.")
    return True

def _discard_stashed_changes(
    git_cmd: list[str],
    cwd: Path,
    stash_ref: str,
) -> bool:
    """Throw away a stash created before an update, without applying it.

    Used only on a NON-interactive update when the user has set
    ``updates.non_interactive_local_changes: discard`` — i.e. they've opted out
    of keeping local source edits on this machine. Drops the stash entry
    instead of re-applying it, so the working tree stays clean at the freshly
    pulled HEAD. Unlike ``git reset --hard`` + ``git clean -fd``, this only
    affects what was stashed (tracked changes + the untracked files we
    explicitly captured) — ignored paths like node_modules/venv/build outputs
    are never touched, since they were never stashed.

    Returns True if the stash was dropped, False on a git failure (in which
    case the stash is left in place for safety).
    """
    stash_selector = _resolve_stash_selector(git_cmd, cwd, stash_ref)
    if stash_selector is None:
        print(
            "⚠ Configured to discard local changes on non-interactive update, "
            "but Hermes couldn't find the stash entry to drop."
        )
        _print_stash_cleanup_guidance(stash_ref)
        return False

    drop = subprocess.run(
        git_cmd + ["stash", "drop", stash_selector],
        cwd=cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if drop.returncode != 0:
        print(
            "⚠ Configured to discard local changes, but Hermes couldn't drop "
            "the saved stash entry."
        )
        if drop.stderr.strip():
            print(f"  {drop.stderr.strip().splitlines()[0]}")
        _print_stash_cleanup_guidance(stash_ref, stash_selector)
        return False

    print("→ Discarded local source changes (updates.non_interactive_local_changes=discard).")
    return True

OFFICIAL_REPO_URLS = {
    "https://github.com/NousResearch/hermes-agent.git",
    "git@github.com:NousResearch/hermes-agent.git",
    "https://github.com/NousResearch/hermes-agent",
    "git@github.com:NousResearch/hermes-agent",
}

OFFICIAL_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"

SKIP_UPSTREAM_PROMPT_FILE = ".skip_upstream_prompt"

def _get_origin_url(git_cmd: list[str], cwd: Path) -> Optional[str]:
    """Get the URL of the origin remote, or None if not set."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "get-url", "origin"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None

def _is_fork(origin_url: Optional[str]) -> bool:
    """Check if the origin remote points to a fork (not the official repo)."""
    if not origin_url:
        return False
    # Normalize URL for comparison (strip trailing .git if present)
    normalized = origin_url.rstrip("/")
    if normalized.endswith(".git"):
        normalized = normalized[:-4]
    for official in OFFICIAL_REPO_URLS:
        official_normalized = official.rstrip("/")
        if official_normalized.endswith(".git"):
            official_normalized = official_normalized[:-4]
        if normalized == official_normalized:
            return False
    return True

def _has_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Check if an 'upstream' remote already exists."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "get-url", "upstream"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0
    except Exception:
        return False

def _add_upstream_remote(git_cmd: list[str], cwd: Path) -> bool:
    """Add the official repo as the 'upstream' remote. Returns True on success."""
    try:
        result = subprocess.run(
            git_cmd + ["remote", "add", "upstream", OFFICIAL_REPO_URL],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0
    except Exception:
        return False

def _count_commits_between(git_cmd: list[str], cwd: Path, base: str, head: str) -> int:
    """Count commits on `head` that are not on `base`. Returns -1 on error."""
    try:
        result = subprocess.run(
            git_cmd + ["rev-list", "--count", f"{base}..{head}"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        if result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return -1

def _should_skip_upstream_prompt() -> bool:
    """Check if user previously declined to add upstream."""
    from hermes_constants import get_hermes_home

    return (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).exists()

def _mark_skip_upstream_prompt():
    """Create marker file to skip future upstream prompts."""
    try:
        from hermes_constants import get_hermes_home

        (get_hermes_home() / SKIP_UPSTREAM_PROMPT_FILE).touch()
    except Exception:
        pass

def _sync_fork_with_upstream(git_cmd: list[str], cwd: Path) -> bool:
    """Attempt to push updated main to origin (sync fork).

    Returns True if push succeeded, False otherwise.
    """
    try:
        result = subprocess.run(
            git_cmd + ["push", "origin", "main", "--force-with-lease"],
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        return result.returncode == 0
    except Exception:
        return False

def _sync_with_upstream_if_needed(git_cmd: list[str], cwd: Path) -> None:
    """Check if fork is behind upstream and sync if safe.

    This implements the fork upstream sync logic:
    - If upstream remote doesn't exist, ask user if they want to add it
    - Compare origin/main with upstream/main
    - If origin/main is strictly behind upstream/main, pull from upstream
    - Try to sync fork back to origin if possible
    """
    has_upstream = _has_upstream_remote(git_cmd, cwd)

    if not has_upstream:
        # Check if user previously declined
        if _should_skip_upstream_prompt():
            return

        # Ask user if they want to add upstream
        print()
        print("ℹ Your fork is not tracking the official Hermes repository.")
        print("  This means you may miss updates from NousResearch/hermes-agent.")
        print()
        try:
            response = (
                input("Add official repo as 'upstream' remote? [Y/n]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt, UnicodeDecodeError):
            print()
            response = "n"

        if response in {"", "y", "yes"}:
            print("→ Adding upstream remote...")
            if _add_upstream_remote(git_cmd, cwd):
                print(
                    "  ✓ Added upstream: https://github.com/NousResearch/hermes-agent.git"
                )
                has_upstream = True
            else:
                print("  ✗ Failed to add upstream remote. Skipping upstream sync.")
                return
        else:
            print(
                "  Skipped. Run 'git remote add upstream https://github.com/NousResearch/hermes-agent.git' to add later."
            )
            _mark_skip_upstream_prompt()
            return

    # Fetch upstream main only. This sync compares upstream/main with
    # origin/main, so there's no reason to pull every upstream ref — and a bare
    # fetch drags in thousands of auto-generated branches.
    print()
    print("→ Fetching upstream...")
    try:
        subprocess.run(
            git_cmd + ["fetch", "upstream", "main", "--quiet"],
            cwd=cwd,
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError:
        print("  ✗ Failed to fetch upstream. Skipping upstream sync.")
        return

    # Compare origin/main with upstream/main
    origin_ahead = _count_commits_between(git_cmd, cwd, "upstream/main", "origin/main")
    upstream_ahead = _count_commits_between(
        git_cmd, cwd, "origin/main", "upstream/main"
    )

    if origin_ahead < 0 or upstream_ahead < 0:
        print("  ✗ Could not compare branches. Skipping upstream sync.")
        return

    # If origin/main has commits not on upstream, don't trample
    if origin_ahead > 0:
        print()
        print(f"ℹ Your fork has {origin_ahead} commit(s) not on upstream.")
        print("  Skipping upstream sync to preserve your changes.")
        print("  If you want to merge upstream changes, run:")
        print("    git pull upstream main")
        return

    # If upstream is not ahead, fork is up to date
    if upstream_ahead == 0:
        print("  ✓ Fork is up to date with upstream")
        return

    # origin/main is strictly behind upstream/main (can fast-forward)
    print()
    print(f"→ Fork is {upstream_ahead} commit(s) behind upstream")
    print("→ Pulling from upstream...")

    try:
        subprocess.run(
            git_cmd + ["pull", "--ff-only", "upstream", "main"],
            cwd=cwd,
            check=True,
        )
    except subprocess.CalledProcessError:
        print(
            "  ✗ Failed to pull from upstream. You may need to resolve conflicts manually."
        )
        return

    print("  ✓ Updated from upstream")

    # Try to sync fork back to origin
    print("→ Syncing fork...")
    if _sync_fork_with_upstream(git_cmd, cwd):
        print("  ✓ Fork synced with upstream")
    else:
        print(
            "  ℹ Got updates from upstream but couldn't push to fork (no write access?)"
        )
        print("    Your local repo is updated, but your fork on GitHub may be behind.")

def _invalidate_update_cache():
    """Delete the update-check cache for ALL profiles so no banner
    reports a stale "commits behind" count after a successful update.

    The git repo is shared across profiles — when one profile runs
    ``hermes update``, every profile is now current.
    """
    homes = []
    # Default profile home (Docker-aware — uses /opt/data in Docker)
    from hermes_constants import get_default_hermes_root

    default_home = get_default_hermes_root()
    homes.append(default_home)
    # Named profiles under <root>/profiles/
    profiles_root = default_home / "profiles"
    if profiles_root.is_dir():
        for entry in profiles_root.iterdir():
            if entry.is_dir():
                homes.append(entry)
    for home in homes:
        try:
            cache_file = home / ".update_check"
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass

def _write_marker_file(path: Path, *, label: str) -> None:
    """Drop an update-recovery breadcrumb. Never raises."""
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping %s marker under pytest (live checkout)", label)
        return
    try:
        path.write_text(
            f"started={_time.time()}\npid={os.getpid()}\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not write %s marker: %s", label, exc)

def _write_update_incomplete_marker() -> None:
    """Drop the interrupted core-install breadcrumb. Never raises."""
    _write_marker_file(_m()._update_marker_path(), label="update-incomplete")

def _write_lazy_refresh_incomplete_marker() -> None:
    """Drop the interrupted lazy-refresh breadcrumb. Never raises."""
    _write_marker_file(_m()._lazy_refresh_marker_path(), label="lazy-refresh-incomplete")

def _format_concurrent_instances_message(
    matches: list[tuple[int, str]], scripts_dir: Path
) -> str:
    """Build a human-readable explanation + remediation hint for the user."""
    shim = scripts_dir / "hermes.exe"
    lines = ["✗ Another hermes.exe is running:"]
    for pid, name in matches:
        lines.append(f"    PID {pid}  {name}")
    lines.append("")
    lines.append(f"  Updating now would fail to overwrite {shim} because")
    lines.append("  Windows blocks REPLACE on a running executable.")
    lines.append("")
    lines.append("  Close Hermes Desktop, exit any open `hermes` REPLs, and")
    lines.append("  stop the gateway (`hermes gateway stop`) before retrying.")
    lines.append("")
    if matches:
        pid_args = " ".join(f"/PID {pid}" for pid, _ in matches)
        lines.append("  If you've already closed everything and these PIDs are")
        lines.append("  stale, terminate them directly, then retry the update:")
        lines.append(f"      taskkill {pid_args} /F")
        lines.append("")
    lines.append("  Override with `hermes update --force` if you've already")
    lines.append("  confirmed those processes will not write to the venv.")
    return "\n".join(lines)

def _upgrade_pip_before_lazy_refresh(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Upgrade pip before lazy-backend refreshes.

    Older pip (e.g. 24.0 on Python 3.11) can fail setuptools-backed source
    builds during lazy installs and leave a partially-written venv (#57828).
    Never raises.
    """
    try:
        _m()._run_package_only_install(
            install_cmd_prefix + ["install", "--upgrade", "pip"],
            env=env,
        )
    except subprocess.CalledProcessError as exc:
        logger.debug("pip upgrade before lazy refresh failed: %s", exc)


def _capture_active_lazy_features() -> list[str]:
    """Snapshot active lazy backends before a managed runtime is replaced."""
    try:
        from tools import lazy_deps

        return lazy_deps.active_features()
    except Exception as exc:
        logger.debug("Could not snapshot active lazy features: %s", exc)
        return []


def _capture_active_tool_dependencies() -> list[str]:
    """Snapshot Python dependencies installed explicitly through ``hermes tools``."""
    try:
        from hermes_cli import tools_config

        return tools_config.active_restorable_python_tool_dependencies()
    except Exception as exc:
        logger.debug("Could not snapshot active Hermes Tools dependencies: %s", exc)
        return []


def _restore_active_tool_dependencies(
    dependencies: list[str],
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Restore allowlisted ``hermes tools`` dependencies into a rebuilt venv.

    The dependency names came from a pre-rebuild import probe and are resolved
    through a static package allowlist. Never raises: a failed optional tool
    must not block the core update, but the user must be told what stayed
    unavailable.
    """
    if not dependencies:
        return

    try:
        from hermes_cli import tools_config
    except Exception as exc:
        logger.debug("Hermes Tools dependency restore skipped (import failed): %s", exc)
        return

    target_python = _m()._resolve_install_target_python(install_cmd_prefix, env)
    missing: list[tuple[str, tuple[str, ...]]] = []
    for name in dependencies:
        spec = tools_config.restorable_python_tool_dependency(name)
        if spec is None:
            continue
        module_name, install_args = spec
        if target_python is not None:
            try:
                probe = subprocess.run(
                    [
                        str(target_python),
                        "-c",
                        "import importlib.util,sys; "
                        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                        module_name,
                    ],
                    capture_output=True,
                    env=env,
                    check=False,
                )
                if probe.returncode == 0:
                    continue
            except (subprocess.SubprocessError, OSError):
                # An indeterminate probe is safer to repair than to treat as
                # proof that a pre-rebuild dependency survived.
                pass
        missing.append((name, install_args))

    if not missing:
        return

    print()
    print(f"→ Restoring {len(missing)} Hermes Tools dependency set(s)...")
    restored: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, install_args in missing:
        try:
            _m()._run_package_only_install(
                install_cmd_prefix + ["install", *install_args, "--quiet"],
                env=env,
            )
            restored.append(name)
        except Exception as exc:
            # This is best-effort recovery for optional tooling. Unexpected
            # installer failures must be surfaced without aborting the core
            # runtime update.
            failed.append((name, str(exc)))

    if restored:
        print(f"  ✓ {len(restored)} restored: {', '.join(restored)}")
    for name, reason in failed:
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {name} failed to restore: {reason}")


def _refresh_active_lazy_features(
    install_cmd_prefix: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    features: list[str] | None = None,
) -> bool:
    """Refresh lazy-installed backends after a code update.

    When pyproject.toml's ``[all]`` extra was slimmed down (May 2026), most
    optional backends moved to ``tools/lazy_deps.py`` and only install on
    first use. ``hermes update`` runs ``uv pip install -e .[all]`` which
    leaves those packages untouched — so if we bump a pin in
    :data:`LAZY_DEPS` (CVE response, transitive bug fix), users who already
    activated the backend keep the stale version forever.

    This function asks lazy_deps which features the user has previously
    activated and reinstalls them under the current pins. Features the
    user never enabled stay quiet — no churn for cold backends.

    Returns True when the venv is safe to use (refresh succeeded, or no
    active lazy backends, or post-failure import repair succeeded). Returns
    False when a failed lazy install left broken core imports that automatic
    repair could not fix (#57828).

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from tools import lazy_deps
    except Exception as exc:
        logger.debug("Lazy refresh skipped (import failed): %s", exc)
        return True

    if features is None:
        try:
            active = lazy_deps.active_features()
        except Exception as exc:
            logger.debug("Lazy refresh skipped (active_features failed): %s", exc)
            return True
    else:
        active = features

    if not active:
        return True

    print()
    print(f"→ Refreshing {len(active)} active lazy backend(s)...")

    unexpected_failure = False
    try:
        if features is None:
            results = lazy_deps.refresh_active_features(prompt=False)
        else:
            results = lazy_deps.restore_features(active)
    except Exception as exc:
        # refresh_active_features is documented as never-raise, but defend
        # the update flow against future regressions.
        print(f"  ⚠ Lazy refresh failed unexpectedly: {exc}")
        results = {}
        unexpected_failure = True

    refreshed = [f for f, s in results.items() if s in {"refreshed", "restored"}]
    current = [f for f, s in results.items() if s == "current"]
    failed = [(f, s) for f, s in results.items() if s.startswith("failed:")]
    skipped = [(f, s) for f, s in results.items() if s.startswith("skipped:")]

    if refreshed:
        print(f"  ↑ {len(refreshed)} refreshed: {', '.join(refreshed)}")
    if current:
        print(f"  ✓ {len(current)} already current")
    if skipped:
        # Most common reason: security.allow_lazy_installs=false. Show one
        # line so the user knows why; not an error.
        names = ", ".join(f for f, _ in skipped)
        reason = skipped[0][1].split(": ", 1)[-1]
        print(f"  · {len(skipped)} skipped ({reason}): {names}")

    if not failed and not unexpected_failure:
        return True

    for feature, status in failed:
        reason = status.split(": ", 1)[-1]
        # Clip noisy pip stderr to keep update output legible.
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {feature} failed to refresh: {reason}")

    if install_cmd_prefix is None:
        print("  ⚠ Lazy refresh failed; rerun `hermes update` once resolved.")
        return False

    # Immediate import-based recovery — metadata-only verifiers miss the case
    # where DISTRIBUTION-INFO remains but import files were wiped (#57828).
    # Unavailable probes are indeterminate, not healthy — keep the lazy marker.
    status = _m()._repair_venv_via_import_probes(install_cmd_prefix, env=env)
    if status == "repaired":
        print(
            "  Lazy backend(s) keep their previous version until refresh succeeds."
        )
        return True
    if status == "healthy":
        print(
            "  Lazy backend(s) keep their previous version; probed packages look intact."
        )
        print("  Rerun `hermes update` once the upstream issue is resolved.")
        return True
    if status == "indeterminate":
        print(
            "  ⚠ Leaving `.lazy-refresh-incomplete` until import probes can confirm health."
        )
    return False

def _refresh_active_memory_provider_dependencies() -> None:
    """Refresh pip dependencies for the configured external memory provider.

    Memory-provider bridge packages are declared in each provider's
    ``plugin.yaml`` (plus mode-dependent extras like Hindsight's
    ``hindsight-all``), NOT in Hermes' editable-install extras or
    ``LAZY_DEPS`` alone — so the core dependency reinstall above can strip
    or downgrade them (#53272 mem0ai, #70636 hindsight-embed). Re-run the
    provider's declared install for the ACTIVE provider only, after the
    core install and lazy refresh, so the last write to any shared package
    is the one the active provider needs.

    Never raises. A failure here must not block the rest of the update.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (config load failed): %s", exc)
        return

    provider = ""
    if isinstance(cfg, dict):
        memory_cfg = cfg.get("memory")
        if isinstance(memory_cfg, dict):
            if memory_cfg.get("enabled") is False:
                return
            provider = str(memory_cfg.get("provider") or "").strip()

    # "default" / empty is the built-in file-backed store — no pip deps.
    if not provider or provider in {"default", "builtin", "none"}:
        return

    try:
        from hermes_cli.memory_setup import _install_dependencies
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (import failed): %s", exc)
        return

    print()
    print(f"→ Refreshing active memory provider dependencies ({provider})...")

    try:
        _install_dependencies(provider, force=True)
    except Exception as exc:
        print(f"  ⚠ {provider} dependencies failed to refresh: {exc}")

def _is_android_python() -> bool:
    return _m().sys.platform == "android"

def _install_psutil_android_compat(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install psutil on Android by patching upstream platform detection.

    psutil's setup currently gates Linux sources behind
    ``sys.platform.startswith('linux')``. On Termux Python reports
    ``sys.platform == 'android'``, so setup aborts with
    "platform android is not supported" despite compiling fine when using the
    Linux source path.

    We patch only the extracted build tree used for this install attempt;
    nothing is persisted in the repository.

    Stopgap: remove this once https://github.com/giampaolo/psutil/pull/2762
    merges and ships in a release. The standalone installer script uses the
    same shared helper and should be removed together.
    """
    import tempfile
    import urllib.request
    from hermes_cli.psutil_android import PSUTIL_URL, prepare_patched_psutil_sdist

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "psutil.tar.gz"
        urllib.request.urlretrieve(PSUTIL_URL, archive)
        src_root = prepare_patched_psutil_sdist(archive, tmp_path)

        _m()._run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)],
            env=env,
        )

def _ensure_uv_for_termux(pip_cmd: list[str]) -> str | None:
    """Best-effort uv bootstrap on Termux for faster update installs.

    The normal path (``ensure_uv()`` in managed_uv) installs the managed
    standalone uv into ``$HERMES_HOME/bin/uv``, but on Termux the official
    installer may not work (glibc vs bionic).  Prefer a uv already on PATH
    (e.g. ``pkg install uv``); only if there is none do we fall back to a
    wheel-only ``pip install uv`` so we never source-build the Rust crate.
    """
    from hermes_cli.managed_uv import resolve_uv

    existing = resolve_uv()
    if existing:
        return existing
    if not _m()._is_termux_env():
        return None
    # A Termux-packaged uv lands on PATH but not in the managed bin dir, so
    # resolve_uv() misses it. Use it before pip, which has no Android wheel and
    # would otherwise build uv from source on a low-memory device.
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv
    try:
        print("  → Termux detected: trying to install uv for faster dependency updates...")
        result = subprocess.run(
            pip_cmd + ["install", "uv", "--only-binary", ":all:"],
            cwd=_m().PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
    except Exception:
        pass
    # After pip install, check managed path first, then PATH
    return resolve_uv() or shutil.which("uv")

def _npm_manifest_paths() -> tuple[Path, ...]:
    """Manifests whose changes must defeat the update-skip.

    The lockfile alone is NOT a sufficient key: on a local checkout a dev
    can edit package.json (root or a workspace) without running npm — the
    lockfile is then unchanged but `hermes update` is exactly the step
    expected to sync node_modules (via the `npm install` fallback in
    _run_npm_install_deterministic).

    The workspace list is pulled from the root package.json's `workspaces`
    globs (npm's own source of truth) rather than hardcoded, so adding a
    workspace can never silently escape the skip key. Every workspace
    manifest belongs in the key — desktop included, even though the
    install only names ui-tui and web — because the single lockfile spans
    the whole workspace graph, so any manifest edit can put the lockfile
    out of sync and change what the install must do. Falls back to hashing
    just root manifests if package.json is unreadable (never skips more
    than main would have installed).
    """
    root_pkg = _m().PROJECT_ROOT / "package.json"
    paths = [_m().PROJECT_ROOT / "package-lock.json", root_pkg]
    try:
        workspaces = json.loads(root_pkg.read_text(encoding="utf-8")).get(
            "workspaces", []
        )
        if isinstance(workspaces, dict):  # legacy {"packages": [...]} form
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            for match in sorted(_m().PROJECT_ROOT.glob(str(pattern))):
                manifest = match / "package.json"
                if manifest.is_file():
                    paths.append(manifest)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return tuple(paths)

def _npm_manifests_digest() -> str | None:
    """Combined sha256 over the lockfile + all workspace package.json files.

    Returns None when the lockfile is missing (never skip then).
    """
    if not (_m().PROJECT_ROOT / "package-lock.json").exists():
        return None
    h = hashlib.sha256()
    for p in _npm_manifest_paths():
        h.update(str(p.relative_to(_m().PROJECT_ROOT)).encode())
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()

def _npm_lockfile_changed(hermes_root: Path) -> bool:
    current = _npm_manifests_digest()
    if current is None:
        return True
    # Also check that node_modules exists; a matching hash with missing
    # node_modules means the cache was recorded by another checkout.
    if not (_m().PROJECT_ROOT / "node_modules").is_dir():
        return True
    # A matching lockfile hash over a tree whose web build toolchain never
    # landed must NOT skip the reinstall — otherwise every later `hermes
    # update` keeps rebuilding against a half-installed tree and serving a
    # stale dist.
    web_dir = _m().PROJECT_ROOT / "web"
    if (web_dir / "package.json").is_file() and not _web_build_toolchain_ready(
        *_web_toolchain_roots(web_dir)
    ):
        return True
    try:
        # Key the cache by PROJECT_ROOT so parallel worktrees don't collide.
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        if not cache_file.exists():
            return True
        return cache_file.read_text(encoding="utf-8").strip() != current
    except OSError:
        return True

def _record_npm_lockfile_hash(hermes_root: Path) -> None:
    digest = _npm_manifests_digest()
    if digest is None:
        return
    try:
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        cache_file.write_text(digest, encoding="utf-8")
    except OSError:
        logger.debug("Could not write npm lockfile hash cache")

def _repair_node_deps_on_current_checkout(print_completion) -> None:
    """Repair Node deps on the ``commit_count == 0`` path (#77211).

    A current checkout does not imply healthy Node deps: a previous npm
    install may have failed (EBADENGINE from a node/npm mismatch, network
    timeout, interrupted install) and its error message says to "re-run
    hermes update" — but the early return never reached the Node refresh,
    so that repair advice could never work. ``_update_node_dependencies``
    self-gates on the lockfile hash, which is only recorded after a
    SUCCESSFUL npm install (and re-trips when node_modules is missing or
    the web toolchain never landed), so this is a cheap no-op on healthy
    installs and a real repair after a failed one.
    """
    node_failures = _update_node_dependencies()
    if node_failures:
        print(f"  ⚠ Node.js refresh failed for: {', '.join(node_failures)}")
        print("    Fix npm and re-run `hermes update`.")
        print_completion(
            "⚠ Checkout is current, but Node.js dependencies could not be repaired."
        )
        return
    # Pair the refresh with the web build like every other
    # _update_node_dependencies call site; it staleness-checks internally,
    # so this is a no-op when nothing changed.
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    print_completion("✓ Already up to date!")


def _update_node_dependencies() -> list[str]:
    """Refresh Node deps for the ui-tui and web workspaces.

    Returns the list of labels whose npm install failed (empty on success),
    so the caller can treat a Node refresh failure as a partial update rather
    than silently reporting ``Update complete!`` (#30271).
    """
    if not (_m().PROJECT_ROOT / "package.json").exists():
        return []

    npm = _m()._resolve_node_runtime_npm()
    if not npm:
        # If the only npm reachable inside this WSL shell is the Windows one,
        # flag it loudly: silently skipping leaves ui-tui deps stale while the
        # rest of the update proceeds, and running it would corrupt the tree.
        from hermes_constants import is_wsl

        path_npm = shutil.which("npm")
        if is_wsl() and path_npm and _m()._is_windows_npm_path(path_npm):
            print("→ Updating Node.js dependencies...")
            print("  ⚠ Skipped: only a Windows npm is reachable from this WSL shell.")
            print("    Install Node.js inside the WSL distro (nvm, or your distro's")
            print("    package manager), then re-run `hermes update`.")
            failed = []
            if any(
                (_m().PROJECT_ROOT / workspace / "package.json").exists()
                for workspace in ("ui-tui", "web")
            ):
                failed.append("ui-tui, web workspaces")
            return failed
        return []

    from hermes_constants import get_default_hermes_root

    # This cache describes PROJECT_ROOT/node_modules, which is shared by every
    # Hermes profile using this checkout. Keep one per-checkout cache under the
    # shared Hermes root rather than rerunning npm once per named profile.
    shared_hermes_root = get_default_hermes_root()

    # Best-effort: warm npx's cache for agent-browser (#43564). Runs before
    # the lockfile-unchanged early return below since that's the common
    # `hermes update` case. Synchronous and can block ~11s on a true cold
    # cache (~0.4s once warm) — print first so that doesn't look like a hang.
    print("→ Warming npx cache for agent-browser...")
    try:
        from tools.browser_tool import warm_agent_browser_npx_cache
        warm_agent_browser_npx_cache()
    except Exception:
        pass

    if not _m()._npm_lockfile_changed(shared_hermes_root):
        logger.info("npm lockfile unchanged, skipping npm install")
        return []

    # Root package.json has no dependencies of its own (agent-browser and
    # @streamdown/math were moved out — see #43564): agent-browser resolves
    # at runtime via `npx agent-browser` (tools/browser_tool.py), and
    # @streamdown/math is a desktop-only import now declared in
    # apps/desktop/package.json. That means a plain workspace-scoped install
    # can never prune anything root-only, so we only need to name the
    # workspaces the CLI/TUI/web build actually requires. apps/desktop pulls
    # in Electron as a devDependency with a ~200MB postinstall download, so
    # it's deliberately never named here — desktop deps install on demand
    # (see _desktop_build_needed).
    print("→ Updating Node.js dependencies...")

    def _partial_update_failure(*labels: str) -> list[str]:
        print()
        print("  ⚠ Node.js dependency refresh did not complete cleanly; the")
        print("    installation may be in a mixed state (updated code, stale Node")
        print("    deps). Fix npm and re-run `hermes update`.")
        return list(labels)

    install_args = [
        "--no-fund", "--no-audit", "--prefer-offline", "--progress=false",
        "--workspace", "ui-tui", "--workspace", "web",
        # Root package.json's own devDependencies (the shared ESLint flat
        # config every workspace's eslint.config.mjs imports) are otherwise
        # pruned by this scoped install, same as agent-browser/@streamdown
        # math used to be before they moved out of root entirely (#43564).
        # Unlike those, root's devDependencies have nowhere else to live —
        # this flag still excludes apps/desktop, which is never named above.
        "--include-workspace-root",
    ]

    from hermes_constants import with_hermes_node_path

    nixos_env = with_hermes_node_path(_m()._nixos_build_env())

    # NOTE: capture_output=False here is deliberate (#18840) — optional
    # postinstall scripts print download progress, and capturing it makes a
    # long download look hung. The chatty npm-deprecation noise during
    # `hermes update` comes from the *desktop* build, not this step; that
    # one is captured to update.log.
    result = _m()._run_npm_install_deterministic(
        npm,
        _m().PROJECT_ROOT,
        extra_args=tuple(install_args),
        capture_output=False,
        env=nixos_env,
    )
    if result.returncode == 0:
        _record_npm_lockfile_hash(shared_hermes_root)
        print("  ✓ ui-tui, web workspaces installed (desktop skipped)")
        failures: list[str] = []
    else:
        print("  ⚠ npm install failed")
        stderr = (result.stderr or "").strip() if result.stderr else ""
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")
        failures = _partial_update_failure("ui-tui, web workspaces")

    return failures

def _log_only_write(text: str) -> None:
    """Write ``text`` to ``~/.hermes/logs/update.log`` only, never the terminal.

    During ``hermes update`` ``sys.stdout`` is an ``_UpdateOutputStream`` that
    mirrors to both the terminal and ``update.log``. Loud, low-signal
    subprocess output (npm installs, the Electron/vite build, the cua-driver
    installer's "Next steps" wall) should be captured and tucked into the log
    so failures stay debuggable, without flooding the user's terminal. This
    reaches past the mirroring stream straight to the underlying log handle.
    """
    if not text:
        return
    stream = _m().sys.stdout
    log_file = getattr(stream, "_log", None)
    if log_file is None:
        return
    try:
        log_file.write(text if text.endswith("\n") else text + "\n")
        log_file.flush()
    except Exception:
        pass

def _run_logged_subprocess(cmd, *, cwd=None, env=None):
    """Run ``cmd`` capturing combined output into update.log (not the terminal).

    Returns the ``CompletedProcess`` (with ``stdout`` populated) so the caller
    can decide whether to surface the captured output on failure.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _log_only_write(result.stdout or "")
    return result

def _cmd_update_check(branch: str = "main", *, branch_explicit: bool = False):
    """Implement ``hermes update --check``: fetch and report without installing.

    ``branch`` selects which branch the check compares against. Default is
    "main"; callers can pass another branch to ask "are there new commits
    on origin/<branch>?" without performing the update.

    ``branch_explicit`` is True iff the caller passed --branch on the CLI.
    Installs that can't honor non-default branches (e.g. Docker) surface a
    one-line notice instead of silently dropping the flag.
    """
    from hermes_cli.config import detect_install_method, recommended_update_command_for_method
    method = detect_install_method(_m().PROJECT_ROOT)
    if method == "docker":
        # Docker can't ``git fetch`` from within the container.  Surface the
        # same long-form ``docker pull`` guidance ``hermes update`` (apply
        # path) uses — telling the user to "reinstall via curl" or that
        # ".git is missing" would point them at the wrong remediation.
        from hermes_cli.config import format_docker_update_message
        print(format_docker_update_message())
        sys.exit(1)

    if method in {"nix", "nixos", "apt"}:
        print(recommended_update_command_for_method(method))
        sys.exit(1)

    git_dir = _m().PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("✗ Not a git repository — cannot check for updates.")
        sys.exit(1)

    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # A crashed/interrupted fetch can leave .git/shallow.lock (or another git
    # lock file) behind; every later fetch then fails with "File exists" and
    # the check reports a hard failure (or, in the banner path, silently
    # compares stale refs). Self-heal abandoned locks before fetching.
    from hermes_cli.gitlock import clear_stale_git_locks

    cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
    for lock_path in cleared:
        print(f"  (removed stale git lock: {lock_path})")

    # Fetch only the branch we compare against; prefer upstream as the canonical
    # reference. A bare `git fetch <remote>` pulls every ref, and this repo has
    # thousands of auto-generated branches, so scope the fetch to <branch>.
    # Note: upstream/<branch> may not exist for non-main branches (a fork's
    # bb/gui has no upstream counterpart), so when the caller picks a
    # non-default branch we skip the upstream probe and use origin directly.
    # Installer checkouts are shallow (`git clone --depth 1`). A plain
    # `git fetch` would unshallow the repo (dragging in the whole history —
    # the exact cost the shallow clone avoided) and the rev-list count below
    # would then report a huge bogus "behind" number. Detect shallow up front:
    # fetch with --depth 1 to preserve the boundary and report presence-only.
    is_shallow = (
        subprocess.run(
            git_cmd + ["rev-parse", "--is-shallow-repository"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        == "true"
    )
    depth_args = ["--depth", "1"] if is_shallow else []

    if branch == "main":
        # Probe locally (~6 ms) whether an 'upstream' remote exists at all
        # before spending a network fetch on it. Non-fork installs have no
        # 'upstream' remote, and the old flow burned a failed network attempt
        # (~0.3-1 s) on every --check before falling back to origin.
        has_upstream_remote = (
            subprocess.run(
                git_cmd + ["remote", "get-url", "upstream"],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).returncode
            == 0
        )
        fetch_result = None
        if has_upstream_remote:
            print("→ Fetching from upstream...")
            fetch_result = subprocess.run(
                git_cmd + ["fetch"] + depth_args + ["upstream", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
        if fetch_result is not None and fetch_result.returncode == 0:
            upstream_exists = True
            compare_branch = f"upstream/{branch}"
        else:
            # No upstream remote, or the upstream fetch failed — use origin.
            print("→ Fetching from origin...")
            fetch_result = subprocess.run(
                git_cmd + ["fetch"] + depth_args + ["origin", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            upstream_exists = False
            compare_branch = f"origin/{branch}"
    else:
        # Non-default branch: compare against origin/<branch> directly.
        print("→ Fetching from origin...")
        fetch_result = subprocess.run(
            git_cmd + ["fetch"] + depth_args + ["origin", branch],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        )
        upstream_exists = False
        compare_branch = f"origin/{branch}"

    if fetch_result.returncode != 0:
        stderr = fetch_result.stderr.strip()
        if "Could not resolve host" in stderr or "unable to access" in stderr:
            print("✗ Network error — cannot reach the remote repository.")
        elif "Authentication failed" in stderr or "could not read Username" in stderr:
            print("✗ Authentication failed — check your git credentials or SSH key.")
        else:
            print("✗ Failed to fetch.")
            if stderr:
                print(f"  {stderr.splitlines()[0]}")
        sys.exit(1)

    # Verify the compare ref actually exists before asking rev-list about it.
    # Without this, `git rev-list HEAD..origin/<bogus> --count` exits 128 and
    # (with check=True) raises CalledProcessError, surfacing a Python
    # traceback. Friendlier to detect-and-report.
    verify_result = subprocess.run(
        git_cmd + ["rev-parse", "--verify", "--quiet", compare_branch],
        cwd=_m().PROJECT_ROOT,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
    )
    if verify_result.returncode != 0:
        print(f"✗ Branch '{branch}' not found on {compare_branch.split('/', 1)[0]}.")
        sys.exit(1)

    if is_shallow:
        # No history to count across the shallow boundary. Compare tip SHAs
        # (mirrors the banner's _check_via_local_git), then try to recover the
        # exact count via the GitHub compare API — the remote graph is complete
        # even when the local one is truncated.
        head_sha = subprocess.run(
            git_cmd + ["rev-parse", "HEAD"],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        target_sha = subprocess.run(
            git_cmd + ["rev-parse", compare_branch],
            cwd=_m().PROJECT_ROOT, capture_output=True, text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if head_sha and target_sha and head_sha == target_sha:
            print("✓ Already up to date.")
        else:
            from hermes_cli.banner import _github_compare_behind
            from hermes_cli.config import recommended_update_command

            counted = _github_compare_behind(head_sha, target_sha)
            if counted == 0:
                # Local commits on top of the remote tip — not behind.
                print("✓ Already up to date.")
                return
            if counted is not None:
                commits_word = "commit" if counted == 1 else "commits"
                print(f"⚕ Update available: {counted} {commits_word} behind {compare_branch}.")
            else:
                print(f"⚕ Update available (behind {compare_branch}).")
            print(f"  Run '{recommended_update_command()}' to install.")
        return

    rev_result = subprocess.run(
        git_cmd + ["rev-list", f"HEAD..{compare_branch}", "--count"],
        cwd=_m().PROJECT_ROOT,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=True,
    )
    behind = int(rev_result.stdout.strip())

    if behind == 0:
        print("✓ Already up to date.")
    else:
        commits_word = "commit" if behind == 1 else "commits"
        print(f"⚕ Update available: {behind} {commits_word} behind {compare_branch}.")
        from hermes_cli.config import recommended_update_command

        print(f"  Run '{recommended_update_command()}' to install.")

def _ensure_fhs_path_guard() -> None:
    """Ensure /usr/local/bin is on PATH for RHEL-family root non-login shells.

    Mirrors the post-symlink probe added to ``scripts/install.sh`` so that
    existing FHS-layout root installs on RHEL/CentOS/Rocky/Alma 8+ get
    repaired on ``hermes update`` without requiring a reinstall.  The
    installer's assumption that ``/usr/local/bin`` is on PATH for every
    standard shell breaks on those distros in non-login interactive shells
    (su, sudo -s, tmux panes, some web terminals): /etc/bashrc doesn't
    add /usr/local/bin and /root/.bash_profile doesn't either.  Symptom:
    ``hermes`` prints ``command not found`` even though the symlink lives
    at /usr/local/bin/hermes.

    Silent no-op on: non-Linux, non-root, non-FHS installs, and any system
    where ``bash -i -c 'command -v hermes'`` already resolves.  Idempotent.
    """
    if _m().sys.platform != "linux":
        return
    try:
        if os.geteuid() != 0:  # windows-footgun: ok — Linux FHS helper, guarded by sys.platform == "linux" above + AttributeError catch
            return
    except AttributeError:
        return
    # Only act when this is actually an FHS-layout install (command link at
    # /usr/local/bin/hermes, code at /usr/local/lib/hermes-agent).
    fhs_link = Path("/usr/local/bin/hermes")
    if not fhs_link.is_symlink() and not fhs_link.exists():
        return

    # Probe a fresh non-login interactive bash the way the user will use it.
    # ``bash -i -c`` sources ~/.bashrc but NOT ~/.bash_profile or /etc/profile,
    # which is the exact scenario where RHEL root loses /usr/local/bin.
    home = os.environ.get("HOME") or "/root"
    try:
        probe = subprocess.run(
            [
                "env",
                "-i",
                f"HOME={home}",
                f"TERM={os.environ.get('TERM', 'dumb')}",
                "bash",
                "-i",
                "-c",
                "command -v hermes",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=10,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return  # no bash or probe hung — don't block update on this
    if probe.returncode == 0:
        return  # already on PATH, nothing to do

    path_line = 'export PATH="/usr/local/bin:$PATH"'
    path_comment = (
        "# Hermes Agent — ensure /usr/local/bin is on PATH " "(RHEL non-login shells)"
    )
    wrote_any = False
    for candidate in (".bashrc", ".bash_profile"):
        cfg = Path(home) / candidate
        if not cfg.is_file():
            continue
        try:
            existing = cfg.read_text(errors="replace", encoding="utf-8")
        except OSError:
            continue
        # Idempotency: skip if any uncommented PATH= line already references
        # /usr/local/bin.  Mirrors the grep pattern used by install.sh.
        already_guarded = any(
            "/usr/local/bin" in line
            and "PATH" in line
            and not line.lstrip().startswith("#")
            for line in existing.splitlines()
        )
        if already_guarded:
            continue
        try:
            with cfg.open("a", encoding="utf-8") as f:
                f.write("\n" + path_comment + "\n" + path_line + "\n")
        except OSError as e:
            print(f"  ⚠ Could not update {cfg}: {e}")
            continue
        print(f"  ✓ Added /usr/local/bin to PATH in {cfg}")
        wrote_any = True
    if wrote_any:
        print("    (reload your shell or run 'source ~/.bashrc' to pick it up)")

def _ensure_acp_launcher() -> None:
    """Self-heal: install a ``hermes-acp`` launcher next to the ``hermes`` one.

    Mirrors the launcher block in ``scripts/install.sh`` so existing installs
    gain the ACP command on ``hermes update`` without a reinstall.  ACP hosts
    (Zed, JetBrains, Buzz Desktop) spawn the agent by resolving the
    ``hermes-acp`` command name against the login-shell PATH; the console
    script of that name lives inside the install's venv, which is not on that
    PATH, so those hosts report Hermes as not installed even when it is.

    The shim simply delegates to the sibling ``hermes`` launcher with the
    ``acp`` subcommand, which makes it correct for every install layout
    (venv wrapper, FHS symlink, pipx/pip console script) without having to
    reconstruct interpreter/entrypoint paths.

    No-op on Windows (install.ps1 copies ``hermes.exe`` + ``hermes-acp.exe``
    into ``$InstallDir\bin`` and puts THAT on the user PATH — never the whole
    ``venv\Scripts`` dir, which would shadow the user's ``python`` (#83797) —
    so ``hermes-acp.exe`` already resolves) and wherever a ``hermes-acp`` is
    already present next to the ``hermes`` command.  Unwritable directories
    (e.g. ``/usr/local/bin`` as non-root) are skipped silently.  Idempotent.
    """
    if _m().sys.platform == "win32":
        return
    for bin_dir in (Path.home() / ".local" / "bin", Path("/usr/local/bin")):
        hermes_cmd = bin_dir / "hermes"
        acp_cmd = bin_dir / "hermes-acp"
        try:
            if not (hermes_cmd.is_file() or hermes_cmd.is_symlink()):
                continue
            # Already present — a console script (pip/pipx install), an
            # earlier shim, or a symlink. is_symlink() catches broken
            # symlinks that exists() would miss; never follow-and-overwrite
            # (the #21454 failure mode).
            if acp_cmd.exists() or acp_cmd.is_symlink():
                continue
            shim = (
                "#!/usr/bin/env bash\n"
                "# Hermes Agent — ACP launcher (written by `hermes update`).\n"
                "# ACP hosts (Zed, JetBrains, Buzz) resolve the agent by this\n"
                "# command name on the login-shell PATH.\n"
                f'exec "{hermes_cmd}" acp "$@"\n'
            )
            acp_cmd.write_text(shim, encoding="utf-8")
            acp_cmd.chmod(acp_cmd.stat().st_mode | 0o755)
        except OSError:
            continue
        print(f"  ✓ Installed hermes-acp launcher → {acp_cmd}")

_PRE_UPDATE_SNAPSHOT_KEEP = 1

# Per-file size cap for the pre-update quick snapshot. Anything larger is
# skipped with a warning: the snapshot exists to protect small, hard-to-
# regenerate state (pairing JSONs, cron jobs, config, auth) — not to copy a
# multi-GB state.db on every update (observed: a 24 GB state.db added ~60s
# of wall time and silently ate 24 GB of disk per update).
_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE = 1 << 30  # 1 GiB

def _resolve_pre_update_backup_mode(args) -> str:
    """Resolve the pre-update backup mode: ``"off"``, ``"quick"``, or ``"full"``.

    CLI flags win over config; ``--no-backup`` beats ``--backup`` when both
    are set. Config accepts the mode strings plus legacy booleans:
    ``true`` → ``full`` (the old zip behavior), ``false`` → ``off``
    (an explicit opt-out now disables the quick snapshot too — previously
    it ran unconditionally, ignoring the user's setting). A missing key
    defaults to ``quick``.
    """
    if getattr(args, "no_backup", False):
        return "off"
    if getattr(args, "backup", False):
        return "full"

    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logging.getLogger(__name__).debug(
            "Could not load config for pre-update backup: %s", exc
        )
        cfg = {}

    updates_cfg = cfg.get("updates", {}) if isinstance(cfg, dict) else {}
    raw = updates_cfg.get("pre_update_backup", "quick")

    if raw is True:
        return "full"
    if raw is False:
        return "off"
    mode = str(raw).strip().lower()
    if mode in ("off", "false", "none", "disabled"):
        return "off"
    if mode in ("full", "zip", "true"):
        return "full"
    if mode == "quick":
        return "quick"
    logging.getLogger(__name__).warning(
        "Unknown updates.pre_update_backup value %r — using 'quick'", raw
    )
    return "quick"

def _run_pre_update_backup(args) -> Optional[str]:
    """Run the pre-update safety backup and return the quick-snapshot id.

    Single consolidated mechanism gated on ``updates.pre_update_backup``:

    - ``off``   — nothing runs. Explicit user opt-out is honored fully.
    - ``quick`` (default) — a state snapshot of critical small files
      (pairing JSONs, cron jobs, config, auth; see ``_QUICK_STATE_FILES``)
      under ``state-snapshots/``. Files over 1 GiB are skipped with a
      warning so a bloated state.db can never stall the update
      (issues #15733, #34600 are the reason this safety net exists).
    - ``full``  — the quick snapshot PLUS a full zip of HERMES_HOME under
      ``backups/`` (restorable via ``hermes import``; the #48200 wrong-path
      wipe is the reason this level exists).

    ``--backup`` forces ``full`` for one run; ``--no-backup`` forces ``off``.
    Never raises — a backup failure should not block the update itself.

    Returns the quick-snapshot id (used by the post-update cron-jobs
    restore safety net), or ``None`` when mode is ``off`` or the snapshot
    failed.
    """
    mode = _resolve_pre_update_backup_mode(args)

    if mode == "off":
        if getattr(args, "no_backup", False):
            print("◆ Pre-update backup: skipped (--no-backup)")
            print()
        # Config-level off is silent — the user opted out; don't spam them
        # on every update.
        return None

    snapshot_id = None
    try:
        from hermes_cli.backup import (
            _quick_snapshot_root,
            create_quick_snapshot,
            verify_sqlite_integrity,
        )

        # NOTE: this function later does `from hermes_constants import
        # get_hermes_home`, which makes the name function-local — the
        # module-level import is shadowed and unbound here. Alias explicitly.
        from hermes_cli.config import get_hermes_home as _get_home

        snapshot_id = create_quick_snapshot(
            label="pre-update",
            keep=_PRE_UPDATE_SNAPSHOT_KEEP,
            max_file_size=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
        )

        # After the snapshot, verify the source state.db is still intact.
        # The snapshot was taken via _safe_copy_db (read-only SQLite backup
        # API), but a concurrent process (antivirus, force-killed gateway
        # releasing file handles, Windows filter driver) can corrupt the live
        # file at any point. A silent zeroing at this point would proceed with
        # the update and exit code 0 — exactly the #68474 symptom.
        if snapshot_id:
            _src_path = _get_home() / "state.db"
            if _src_path.exists():
                _integrity = verify_sqlite_integrity(
                    _src_path,
                    check_header=True,
                    run_pragma=True,
                    max_bytes=_PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
                )
                if not _integrity.get("valid"):
                    _msg = _integrity.get("message", "unknown error")
                    print(
                        f"  ⚠ state.db integrity check FAILED after snapshot: {_msg}"
                    )
                    # Check if the snapshot itself is valid.
                    _snap_root = _quick_snapshot_root(_get_home())
                    _snap_state = _snap_root / snapshot_id / "state.db"
                    if _snap_state.exists():
                        _snap_ok = verify_sqlite_integrity(
                            _snap_state, check_header=True, run_pragma=True
                        )
                        if _snap_ok.get("valid"):
                            print(
                                "  ✓ Snapshot copy is valid — continuing update."
                            )
                            print(
                                "    If state.db is lost after update it will be auto-restored."
                            )
                        else:
                            print(
                                "  ✗ Snapshot copy ALSO failed integrity — "
                                "the source was already corrupted before the backup."
                            )
                    else:
                        print(
                            "  ⚠ Snapshot does not contain state.db (was skipped or too large)."
                        )
                    print()
        if snapshot_id:
            print(f"◆ Pre-update snapshot: {snapshot_id}")
    except Exception as exc:
        # Never let a snapshot failure block an update.
        logging.getLogger(__name__).debug("Pre-update snapshot failed: %s", exc)

    if mode != "full":
        if snapshot_id:
            print()
        return snapshot_id

    try:
        from hermes_cli.backup import create_pre_update_backup
    except Exception as exc:
        print(
            f"⚠ Pre-update backup: could not load backup module ({exc}); continuing update."
        )
        print()
        return snapshot_id

    try:
        from hermes_cli.config import load_config

        _keep = (load_config() or {}).get("updates", {}).get("backup_keep", 5)
    except Exception:
        _keep = 5

    print("◆ Creating pre-update backup...")
    t0 = _time.monotonic()
    try:
        out_path = create_pre_update_backup(keep=int(_keep))
    except Exception as exc:  # defensive — helper already swallows, but just in case
        print(f"  ⚠ Backup failed: {exc}")
        print("  Continuing with update.")
        print()
        return snapshot_id

    elapsed = _time.monotonic() - t0

    if out_path is None:
        print("  ⚠ Backup skipped (no files found or write failed); continuing update.")
        print()
        return snapshot_id

    try:
        size_bytes = out_path.stat().st_size
    except OSError:
        size_bytes = 0

    # Human-readable size
    from hermes_cli.sizefmt import format_bytes

    size_str = format_bytes(size_bytes)

    # Render path using display_hermes_home so the user sees ~/.hermes/...
    try:
        from hermes_constants import get_hermes_home, display_hermes_home

        home = get_hermes_home()
        try:
            display_path = f"{display_hermes_home()}/{out_path.relative_to(home)}"
        except ValueError:
            display_path = str(out_path)
    except Exception:
        display_path = str(out_path)

    print(f"  Saved:    {display_path} ({size_str}, {elapsed:.1f}s)")
    print(f"  Restore:  hermes import {out_path}")
    print("  Disable:  set updates.pre_update_backup: quick (or off) in config.yaml")
    print()
    return snapshot_id

def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from datetime import timezone

        from gateway.status import _get_process_start_time
        from utils import atomic_json_write

        record = {
            "target_pid": pid,
            "target_start_time": _get_process_start_time(pid),
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            record,
            indent=None,
            separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False

def _wait_for_windows_update_gateway_exit(
    pids: list[int], *, timeout: float
) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()

    from gateway.status import _pid_exists

    remaining = set(pids)
    deadline = _time.monotonic() + max(timeout, 0.0)
    while remaining and _time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                if not _pid_exists(pid):
                    remaining.discard(pid)
            except Exception:
                remaining.discard(pid)
        if remaining:
            _time.sleep(0.25)

    survivors: set[int] = set()
    for pid in remaining:
        try:
            if _pid_exists(pid):
                survivors.add(pid)
        except Exception:
            pass
    return survivors

def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the project venv for the core imports the backend needs to boot.

    Runs a tiny import check inside the venv interpreter (NOT this process —
    ``hermes update`` may be driven by a different Python). Catches the
    half-updated-venv state: git checkout current but a dependency sync that
    failed or was killed partway (e.g. Windows access-denied on a loaded
    .pyd), leaving imports like ``fastapi``'s new transitive deps missing.
    Without this probe, ``hermes update`` on a current checkout prints
    "Already up to date!" and returns without ever re-syncing dependencies —
    the user's install stays broken no matter how many times they update
    (ryanc's incident, July 2026).

    Returns ``(healthy, detail)``. Never raises; unknown states report
    healthy so a probe failure can't force needless reinstalls.
    """
    venv_dir = _m().PROJECT_ROOT / "venv"
    venv_python = venv_python_path(venv_dir, windows=_m()._is_windows())
    if not venv_python.exists():
        # No venv interpreter at all. In a dev checkout that's normal (the
        # dev may run hermes from any interpreter), so report healthy to
        # avoid forcing reinstalls. But on a MANAGED install (the Windows
        # installer / desktop bootstrap stamps `.hermes-bootstrap-complete`,
        # and an interrupted update leaves `.update-incomplete`), the venv
        # IS the install — its absence means a repair got interrupted after
        # the old venv was moved aside, and "Already up to date!" would
        # gaslight the user while nothing can run.
        managed_markers = (
            _m().PROJECT_ROOT / ".hermes-bootstrap-complete",
            _m()._update_marker_path(),
        )
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Core web/serve imports plus their newest transitive deps. Import (not
    # just metadata) — a package can have intact dist-info but a missing
    # module after an interrupted uninstall/install cycle.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=60,
            cwd=_m().PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken (e.g. deleted stdlib) — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""

def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Find live processes running from the project venv's interpreter.

    The hermes.exe shim guard misses the biggest lock-holder class on
    Windows: the Desktop app's backend (``python.exe -m hermes_cli.main
    serve``) and anything else running straight off ``venv\\Scripts\\python
    (w).exe``. Those processes keep native ``.pyd`` extensions mapped, so a
    dependency sync mid-update dies with access-denied and strands the venv
    half-updated (ryanc's brotlicffi/_sodium.pyd incidents, July 2026).

    Killing them from here is pointless — the Desktop app supervises its
    backend and respawns it within seconds — so the caller should refuse and
    tell the user to close the app instead. Returns ``(pid, name, cmdline)``
    tuples; empty off-Windows / without psutil / when nothing matches. The
    calling process and its ancestors are always excluded (a CLI ``hermes
    update`` itself runs from the venv python). Never raises.
    """
    if not _m()._is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(_m().PROJECT_ROOT.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(_m().PROJECT_ROOT).lower().rstrip(os.sep) + os.sep

    skip: set[int] = set(exclude_pids or set())
    skip.add(os.getpid())
    try:
        for anc in psutil.Process().parents():
            skip.add(int(anc.pid))
    except Exception:
        pass

    matches: list[tuple[int, str, str]] = []
    try:
        proc_iter = psutil.process_iter(["pid", "exe", "name", "cmdline", "cwd"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        cmdline_raw = " ".join(info.get("cmdline") or [])
        cmdline_low = cmdline_raw.lower()
        cwd_low = str(info.get("cwd") or "").lower().rstrip(os.sep) + os.sep

        # Primary match: the executable itself lives under this venv
        # (venv\Scripts\python(w).exe — the desktop backend / gateway case).
        is_holder = exe_norm.startswith(venv_prefix)
        # Fallback: uv/base-interpreter trampolines run a python whose exe is
        # OUTSIDE the venv but which still imports from it and holds its .pyd
        # files. Catch those by what they're running: a cmdline that references
        # this venv's path, or a `-m hermes_cli.main ...` invocation tied to
        # this install (install root in the cmdline or as the working dir).
        if not is_holder and venv_prefix in cmdline_low:
            is_holder = True
        if not is_holder and "hermes_cli.main" in cmdline_low:
            if root_prefix in cmdline_low or cwd_low.startswith(root_prefix):
                is_holder = True
        if not is_holder:
            continue
        name = info.get("name") or Path(exe).name
        # Return the FULL cmdline: callers match against it (the Desktop
        # preflight's pausable-gateway exemption parses for `gateway run`).
        # Truncating here cut long managed-runtime interpreter paths before
        # the `-m hermes_cli.main gateway run` argv, so autostarted gateways
        # were misreported as blockers and the update dead-ended. Truncate
        # only at display time.
        matches.append((int(pid), str(name), cmdline_raw))
    return matches

# Native-extension modules that pin files inside the venv once imported.  If
# the updater process itself has any of these loaded, the dependency sync
# below cannot rewrite the backing ``.pyd``/``.dll`` — Windows blocks REPLACE
# on a mapped image — and the update dies with ``os error 5`` between
# uninstall and reinstall, stranding the venv half-updated (#83569).
# ``cryptography`` is the canonical case: ``hermes_cli.main`` used to import
# it at startup while resolving external secret sources; ``PyYAML``'s
# ``_yaml`` C extension is loaded by every CLI process (config parsing).
# Keep this guard as defence-in-depth against future eager imports (new
# secret sources, plugins absorbed into core, refactors of the startup
# order) — but the guard must be HONEST (#86735/#86780/#86781: a preflight
# that fired on every run, before the fetch, re-bricked the exact flow it
# was meant to protect).  Two honesty gates:
#
# 1. It only fires when the dependency sync would actually REWRITE the
#    loaded distribution (``_dependency_sync_would_rewrite``): if the
#    installed version already satisfies the on-disk pyproject pins, uv/pip
#    will not touch the mapped ``.pyd``, so there is no lock to trip.
# 2. It runs AFTER the code swap (git pull / ZIP commit), immediately
#    before the venv rewrite — so the on-disk pyproject is the NEW one
#    (gate 1 compares against the right target) and a deferral no longer
#    strands the user on the old checkout: the next launch's marker
#    recovery completes the dependency install against the already-updated
#    pyproject.
#
# Keys are module prefixes in ``sys.modules``; values are
# ``(display name, PyPI distribution name)``.
_SELF_LOCKING_NATIVE_MODULES: dict[str, tuple[str, str]] = {
    "cryptography.hazmat.bindings._rust": ("cryptography (_rust.pyd)", "cryptography"),
    "yaml._yaml": ("PyYAML (_yaml.pyd)", "pyyaml"),
}


def _dependency_sync_would_rewrite(dist_name: str) -> bool | None:
    """Whether ``uv pip install -e .[all]`` would replace *dist_name*'s files.

    Compares the installed distribution version against every applicable
    requirement for it in the on-disk ``pyproject.toml`` (base dependencies
    plus all optional extras).  Returns:

    - ``False`` — installed version satisfies every pin: the resolver will
      leave the wheel alone, so a mapped extension is NOT at risk.
    - ``True``  — some pin is not satisfied (or the distribution is
      missing): the sync will rewrite it.
    - ``None``  — could not determine (parse failure, unparseable pins).

    Never raises.  Callers treat ``None`` as fail-OPEN (no deferral): a
    module in the registry can be loaded by every process (PyYAML), so
    deferring on uncertainty would recreate the #86735 always-firing loop.
    """
    try:
        from importlib import metadata as _ilmd

        installed = _ilmd.version(dist_name)
    except Exception:
        return True  # not installed → the sync will definitely install it
    try:
        import tomllib

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version

        pyproject = _m().PROJECT_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        req_strings: list[str] = list(project.get("dependencies") or [])
        for extra_reqs in (project.get("optional-dependencies") or {}).values():
            req_strings.extend(extra_reqs or [])

        target = canonicalize_name(dist_name)
        installed_v = Version(installed)
        saw_pin = False
        for req_str in req_strings:
            try:
                req = Requirement(req_str)
            except Exception:
                continue
            if canonicalize_name(req.name) != target:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            saw_pin = True
            if installed_v not in req.specifier:
                return True
        if saw_pin:
            return False
        # Not pinned anywhere in pyproject: the resolver may still move it
        # as a transitive — we cannot cheaply predict that, so stay honest
        # about the uncertainty.
        return None
    except Exception:
        return None


def _detect_self_loaded_native_modules() -> list[str]:
    """Native venv extensions loaded into THIS process that the sync would rewrite.

    Returns display names (empty off Windows — POSIX lets a running process
    keep using an unlinked inode, so self-locking is a Windows-only hazard).
    A loaded module whose installed version already satisfies the on-disk
    pyproject pins is NOT reported: the dependency sync will not touch its
    files, so there is no swap at risk (#86735 — the always-firing variant
    of this preflight bricked every Windows update).  Never raises.
    """
    if not _m()._is_windows():
        return []
    found = []
    for prefix, (display, dist) in _SELF_LOCKING_NATIVE_MODULES.items():
        if prefix not in sys.modules:
            continue
        # Defer ONLY on a CONFIRMED pending rewrite. An "unknown" result
        # (unreadable/unparseable pyproject, no pin found) must fail OPEN:
        # PyYAML is loaded in every CLI process, so treating unknown as
        # at-risk would re-create the exact always-firing loop this guard's
        # first version caused (#86735). The downside of a missed deferral
        # is the pre-existing failure mode — a mid-sync os error 5 that the
        # marker recovery already handles — which is strictly less harmful
        # than an update that can never run.
        if _m()._dependency_sync_would_rewrite(dist) is not True:
            continue
        found.append(display)
    return sorted(set(found))


def _abort_dependency_sync_if_self_locked(gateway_resume=None) -> None:
    """Defer (exit 2) when THIS process holds a native module the sync must replace.

    Runs at the last moment before the venv rewrite — after the code swap —
    so the on-disk pyproject reflects the update target and a deferral
    leaves the user on NEW code with only the dependency install pending
    (completed by the next launch's marker recovery).  No-op when nothing
    at-risk is loaded.
    """
    locked = _m()._detect_self_loaded_native_modules()
    if not locked:
        return
    _m()._defer_update_for_self_lock(locked)
    if gateway_resume is not None:
        _m()._resume_windows_gateways_after_update(gateway_resume)
    sys.exit(2)


def _defer_update_for_self_lock(loaded: list[str]) -> None:
    """Bail out before the dependency sync when the updater holds a lock.

    The install cannot win this race from inside the locked process — even
    killing threads would not unmap the image — so defer it: drop the
    update-incomplete marker (next launch's fresh process completes the
    install before importing anything heavy), explain, and exit 2 like the
    other preflight refusals.
    """
    print("✗ This updater process has already loaded native venv modules that")
    print("  the dependency sync must replace:")
    for name in loaded:
        print(f"    {name}")
    print()
    print("  On Windows a mapped extension cannot be replaced by the process")
    print("  holding it. The code update has been applied; only the dependency")
    print("  sync has been deferred: the next `hermes` launch will complete it")
    print("  in a fresh process before anything imports these modules.")
    _m()._write_update_incomplete_marker()


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them."""
    lines = [
        "✗ Other Hermes processes are running from this install's venv:",
    ]
    for pid, name, cmdline in matches[:6]:
        hint = ""
        low = cmdline.lower()
        if "serve" in low or "dashboard" in low:
            hint = "  ← Hermes Desktop backend (close the desktop app)"
        elif "gateway" in low:
            hint = "  ← gateway"
        lines.append(f"  PID {pid}  {name}  {cmdline[:120]}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append("")
    lines.append(
        "  On Windows these keep native extension files (.pyd) locked, so the"
    )
    lines.append(
        "  dependency update would fail partway and leave a broken install."
    )
    lines.append(
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:"
    )
    lines.append("    hermes update")
    lines.append("  (or use `hermes update --force-venv` to proceed anyway at your own risk)")
    return "\n".join(lines)

def _venv_launcher_ancestors(pids: list[int]) -> list[int]:
    """Return venv-interpreter ancestors of *pids* that hold the install open.

    On Windows a gateway started through the venv shim is a **two-process
    chain**: ``venv\\Scripts\\python.exe`` (the launcher, which keeps native
    ``.pyd`` files from the venv mapped) spawns the actual interpreter from
    uv's managed CPython directory (``AppData\\Roaming\\uv\\python\\...``).
    The gateway writes its PID file from the *child*, so
    ``find_gateway_pids()`` — and therefore this module's pause set — only
    ever sees the uv-side worker.

    ``_detect_venv_python_processes()`` matches on the venv path prefix, so
    the guard downstream of the pause sees the *launcher* instead. The two
    sets are disjoint, which meant a paused gateway still tripped the
    venv-holder guard and aborted the update every time (the Desktop
    "venv-blocked: N process(es) hold the install" dead-end, where the
    reported holder is a gateway the updater believes it already stopped).

    Walking one hop up from each mapped gateway PID and keeping ancestors
    that live under the project venv closes the gap. Only the venv-side
    parent is returned — unrelated ancestors (the Scheduled Task's
    ``cmd.exe``, an operator's shell) are ignored so we never widen the
    blast radius beyond the gateway's own launcher. Never raises.
    """
    if not _m()._is_windows() or not pids:
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep

    # Never return ourselves or our own ancestry: a CLI ``hermes update``
    # runs from the venv python and would otherwise nominate itself.
    skip: set[int] = {os.getpid()}
    try:
        for anc in psutil.Process().parents():
            skip.add(int(anc.pid))
    except Exception:
        pass

    found: list[int] = []
    for pid in pids:
        try:
            parent = psutil.Process(int(pid)).parent()
        except Exception:
            continue
        if parent is None:
            continue
        ppid = int(parent.pid)
        if ppid in skip or ppid in found or ppid in set(pids):
            continue
        try:
            exe = (parent.exe() or "").lower()
        except Exception:
            continue
        if exe.startswith(venv_prefix):
            found.append(ppid)
    return found


def _leftover_pausable_gateway_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when every remaining venv holder is a pausable gateway.

    ``_pause_windows_gateways_for_update()`` stops every gateway its discovery
    finds, but the venv-holder guard downstream sees the process table as it
    is *now*: a gateway respawned by its supervisor (Scheduled Task, login
    watchdog) inside the pause→guard window, or one started through a spawn
    path the discovery does not map, still holds venv ``.pyd`` files and
    would dead-end the update — an abort pointed at exactly the kind of
    process the pause machinery exists to stop.

    Holders are classified with the same matcher the Desktop preflight uses
    to exempt them (``_is_pausable_gateway``), so the preflight's exemption
    and this guard's tolerance cannot drift apart — matcher drift between
    two views of the same process table is what produced the launcher/worker
    dead-end fixed above. The scan captures only a 120-char cmdline prefix,
    so the live argv is re-read where psutil allows; an unreadable argv
    falls back to the captured prefix.

    Returns ``None`` when any holder is not a pausable gateway — an operator
    REPL, a stray script, or the Desktop backend has no pause machinery
    downstream, and the guard must keep refusing exactly as before.
    """
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    pids: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        if psutil is not None:
            try:
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
            except Exception:
                pass
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _orphaned_desktop_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when every remaining holder is an ORPHANED backend.

    The venv-holder guard refuses on the Desktop app's ``serve`` backend by
    design: while the Desktop is open, killing its backend is futile (the app
    supervises and respawns it within seconds), so the user must close the
    app. But in the GUI-updater handoff path the Desktop has *already
    exited* — by contract it tree-kills its backends and waits for the venv
    shim before spawning hermes-setup, and the update-in-progress marker
    parks any relaunched Desktop from spawning a fresh backend (#50238). A
    ``serve`` backend still holding the venv at that point is a straggler
    whose supervisor is gone: SIGTERM raced its spawn, or it belongs to a
    crashed window. Nothing will respawn it, and refusing on it dead-ends
    the update with "Hermes is still running" while the user stares at zero
    open windows (ryanc's 2026-08-09 01:59/02:17 failures).

    A holder qualifies only when BOTH hold:

    - its cmdline is a Hermes backend (``hermes_cli.main`` + ``serve`` /
      ``dashboard``), and
    - its supervising parent is demonstrably gone: the parent PID no longer
      exists, or the PID was reused (parent created *after* the child).

    Tree-aware: the scanner can return an orphaned backend AND one of its
    managed-runtime descendants (the ``.hermes-runtime`` interpreter child)
    in the same holder set. That descendant has a live parent — the orphaned
    backend itself — and isn't a ``serve`` cmdline, so per-process rules
    would refuse a set that is entirely safe to reap. Holders that sit
    inside an accepted orphan root's tree are therefore folded into that
    root (only roots are returned; ``taskkill /T`` reaps the descendants).

    Any other live-parent backend (the Desktop is still open), non-backend
    holder outside an orphan tree, or unprovable case disqualifies the whole
    set — the guard must keep refusing exactly as before. Returns ``None``
    in that case, or when psutil is unavailable (can't prove orphanhood →
    refuse). Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[int] = []
    remaining: list[tuple[int, str]] = []  # (pid, argv_low) still to justify
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Holder exited between scan and classification — nothing to
            # reap, nothing blocking. Skip it.
            continue
        except Exception:
            pass
        low = argv.lower()
        if not _is_backend(low):
            remaining.append((int(pid), low))
            continue
        try:
            proc = psutil.Process(int(pid))
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if parent is not None and parent.is_running():
                # PID-reuse check: a "parent" created after its child is a
                # recycled PID, not the real (dead) supervisor.
                if parent.create_time() <= proc.create_time():
                    # Live parent — NOT a root. But it may still be a
                    # descendant of an orphan root: the venv python.exe is
                    # a trampoline that re-execs the uv-managed interpreter
                    # with the SAME backend argv, so the worker half of the
                    # two-process chain lands here. Defer to pass 2 instead
                    # of refusing outright.
                    remaining.append((int(pid), low))
                    continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append(int(pid))

    # Pass 2: every non-backend holder must be a descendant of an accepted
    # orphan root — then it dies with the root's tree reap. Anything else
    # (operator REPL, stray script) keeps the refusal.
    root_set = set(roots)
    for pid, _low in remaining:
        if not root_set:
            return None
        try:
            ancestors = {int(a.pid) for a in psutil.Process(pid).parents()}
        except psutil.NoSuchProcess:
            continue  # exited already
        except Exception:
            return None
        if not (ancestors & root_set):
            return None
    return roots


def _stop_process_trees(pids: list[int]) -> None:
    """Force-stop each PID with its full child tree (Windows).

    ``taskkill /T /F`` mirrors the Desktop's ``forceKillProcessTree`` and
    install.ps1's venv sweep: stopping only the parent can leave a managed
    ``.hermes-runtime`` interpreter child alive and holding the install open
    (#70026). Best effort; never raises.
    """
    for pid in pids:
        try:
            subprocess.run(
                ["taskkill", "/PID", str(int(pid)), "/T", "/F"],
                check=False,
                capture_output=True,
            )
        except Exception as exc:
            logger.debug("Could not stop process tree %s: %s", pid, exc)


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Windows scheduled/startup gateways run through pythonw.exe, so the generic
    hermes.exe concurrent-instance guard does not see them. They still import
    from the checkout and can keep files locked while ``git`` or ``uv`` updates
    the install. Stop only PIDs that the gateway discovery code identifies.
    """
    if not _m()._is_windows():
        return None

    try:
        from gateway.status import terminate_pid
        from hermes_cli.gateway import (
            _capture_gateway_argv,
            _get_restart_drain_timeout,
            find_gateway_pids,
            find_profile_gateway_processes,
        )
    except Exception as exc:
        logger.debug("Could not prepare Windows gateway pause for update: %s", exc)
        return None

    try:
        running_pids = list(dict.fromkeys(find_gateway_pids(all_profiles=True)))
    except Exception as exc:
        logger.debug("Could not discover Windows gateway PIDs before update: %s", exc)
        return None
    if not running_pids:
        # No gateway is running right now, but the user may have installed an
        # autostart entry (Scheduled Task or Startup-folder login item) — that
        # is an explicit "I want a gateway" signal. A gateway that died between
        # updates (e.g. the spawning terminal/TUI closed, taking its child with
        # it) would otherwise never come back: the autostart entry only fires on
        # the next login, and the update flow's resume path only relaunched
        # gateways that were running when the update began. Cold-start one after
        # the update so an installed gateway is actually up post-update. Users
        # who run gateway-less (no autostart entry) get nothing forced on them.
        try:
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        except Exception as exc:
            logger.debug(
                "Could not check Windows gateway autostart state before update: %s",
                exc,
            )
        return None

    profile_processes = {}
    try:
        profile_processes = {
            proc.pid: proc for proc in find_profile_gateway_processes()
        }
    except Exception as exc:
        logger.debug("Could not map Windows gateway PIDs to profiles: %s", exc)

    profiles: dict[str, int] = {}
    mapped_pids = []
    for pid in running_pids:
        proc = profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))

    # Resolve each mapped worker's venv-side launcher BEFORE draining: the
    # drain stops tracking a PID exactly when it dies, so a gracefully
    # drained worker is gone by the time the wait returns — and a dead pid's
    # parent cannot be recovered (psutil raises NoSuchProcess). The snapshot
    # is stopped after the drain alongside the survivors.
    #
    # Why launchers matter: the drain targets the PID that wrote the PID
    # file (the uv-side worker). On Windows that worker's parent is usually
    # the venv-side ``python.exe`` launcher, which keeps venv ``.pyd`` files
    # mapped and is what ``_detect_venv_python_processes()`` reports
    # downstream. Left alive, it trips the venv-holder guard and aborts the
    # update even though the gateway itself is stopped.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    survivors = _m()._wait_for_windows_update_gateway_exit(
        mapped_pids,
        timeout=drain_timeout,
    )
    unmapped_pids = [pid for pid in running_pids if pid not in profile_processes]

    # Snapshot each unmapped gateway's command line *before* we force-kill it,
    # so ``_resume_windows_gateways_after_update`` can respawn it by replaying
    # its own argv. Unmapped gateways are ones with no profile→PID-file mapping
    # — e.g. a Windows Scheduled Task running ``pythonw.exe -m hermes_cli.main
    # gateway run``. Without this snapshot they were force-killed and never
    # restarted (the "Restart manually after update" dead-end from #50090).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    # Stop drain survivors, unmapped gateways, and the pre-drain launcher
    # snapshot. ``terminate_pid(force=True)`` is a tree kill, so a launcher
    # that outlived its worker takes any stragglers with it; a launcher that
    # already exited with its drained worker raises ProcessLookupError below
    # and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        try:
            terminate_pid(int(pid), force=True)
            force_killed.append(int(pid))
        except (ProcessLookupError, PermissionError, OSError):
            pass

    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")

    if unmapped_pids:
        respawnable = sum(1 for u in unmapped if u.get("argv"))
        print(
            f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping"
        )
        if respawnable < len(unmapped_pids):
            # Some had no recoverable command line (psutil missing, access
            # denied, already gone): those still need a manual restart.
            print("    Restart manually after update: hermes gateway run")

    return {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }

def _cold_start_windows_gateway_after_update() -> None:
    """Start a fresh detached gateway after update when one is installed but down.

    Invoked from ``_resume_windows_gateways_after_update`` for the
    ``cold_start_if_installed`` case: no gateway was running when the update
    began, but an autostart entry (Scheduled Task / Startup-folder login item)
    is installed, signalling the user wants a gateway. Unlike the relaunch
    paths — which watch an old PID and respawn once it exits — this is a direct
    fresh spawn via the same hidden-console + breakaway path that
    ``hermes gateway start`` uses (``gateway_windows._spawn_detached``).

    Best-effort and idempotent: re-checks that nothing is running first so a
    concurrent start (e.g. the autostart entry firing) can't produce a
    duplicate gateway.

    A successful ``Popen`` only proves the process was created, not that it
    survived (e.g. a Windows job object denying breakaway kills it before it
    logs anything — #84185). So the success line is gated on the same
    post-spawn liveness poll every other ``_spawn_detached`` caller uses
    (``gateway_windows._report_gateway_start``), instead of being printed
    unconditionally from the returned PID.
    """
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    except Exception as exc:
        logger.debug("Could not load Windows gateway cold-start helpers: %s", exc)
        return

    # Re-check liveness right before spawning — between pause and resume the
    # autostart entry may have already brought a gateway up, or a leftover
    # process may have re-registered. Don't double-start.
    try:
        if list(find_gateway_pids(all_profiles=True)):
            return
    except Exception as exc:
        logger.debug("Could not re-check gateway liveness before cold-start: %s", exc)
        return

    try:
        pid = gateway_windows._spawn_detached()
    except Exception as exc:
        logger.debug("Could not cold-start Windows gateway after update: %s", exc)
        return

    if pid:
        print()
        gateway_windows._report_gateway_start(f"cold-start after update (PID {pid})")

def _for_each_systemd_gateway_unit(
    list_units_stdout: str,
    *,
    process_unit,
    on_unit_timeout,
) -> None:
    """Process each ``hermes-gateway*.service``/``hermes-serve*.service`` unit
    from ``systemctl list-units``.

    ``subprocess.TimeoutExpired`` raised by ``process_unit`` is isolated to
    that unit via ``on_unit_timeout`` so one wedged systemctl call cannot
    abort the rest of the fleet (#68523).
    """
    for line in (list_units_stdout or "").strip().splitlines():
        parts = line.split()
        if not parts:
            continue
        unit = parts[0]
        if not unit.endswith(".service"):
            continue
        # list-units is already pattern-filtered, but keep the name gate so a
        # stray non-gateway/serve line cannot enter the restart path.
        # ``unit.startswith("hermes-serve")`` alone would also accept the
        # unrelated ``hermes-server.service`` — require the exact base unit
        # or the hyphenated profile family instead (review on #83595).
        if not (
            unit == "hermes-gateway.service"
            or unit.startswith("hermes-gateway-")
            or unit == "hermes-serve.service"
            or unit.startswith("hermes-serve-")
        ):
            continue
        svc_name = unit.removesuffix(".service")
        try:
            process_unit(svc_name)
        except subprocess.TimeoutExpired as exc:
            on_unit_timeout(svc_name, exc)

def _service_unit_supports_graceful_sigusr1_restart(svc_name: str) -> bool:
    """Whether *svc_name* wires SIGUSR1 to a graceful drain-then-restart.

    Only ``hermes-gateway*`` units run ``gateway/run.py``, which installs the
    SIGUSR1 handler. ``hermes-serve*`` units (#83438) don't, so sending them
    SIGUSR1 would just invoke the default terminate action and burn the full
    drain budget waiting for an exit that was never graceful — go straight to
    the blunt ``systemctl restart`` path for those instead.

    Uses the same strict exact/hyphenated shape as the unit-name gate in
    ``_for_each_systemd_gateway_unit`` so a hypothetical near-prefix unit
    (``hermes-gateway-helper`` is fine — profile units are
    ``hermes-gateway-<profile>`` — but ``hermes-gatewayd``-style names are
    not) can't be sent a SIGUSR1 it doesn't handle.
    """
    return svc_name == "hermes-gateway" or svc_name.startswith("hermes-gateway-")


def _warn_incomplete_gateway_fleet_restart(failed_units: list) -> None:
    """Print an explicit incomplete-update warning for unrestarted units."""
    if not failed_units:
        return
    # Preserve discovery order while de-duplicating.
    seen = set()
    ordered = []
    for name in failed_units:
        if name in seen:
            continue
        seen.add(name)
        ordered.append(name)
    print()
    print("⚠ Update incomplete — some units were not restarted:")
    for name in ordered:
        print(f"    - {name}")
    print("  Skipped units may still be running pre-update code (mixed")
    print("  sys.modules). Restart them manually, then verify:")
    print("    hermes gateway status")
    print("    systemctl --user restart <unit>   # user-scope")
    print("    sudo systemctl restart <unit>     # system-scope")

def _surviving_gateway_pids_after_failed_restart():
    """Best-effort PIDs of gateways still running after the restart phase died.

    Returns ``None`` when the answer cannot be determined — most importantly
    when ``hermes_cli.gateway`` itself no longer imports, which is one of the
    ways the restart phase aborts in the first place (the update replaced the
    checkout under a process that already loaded the old modules). ``None`` and
    a non-empty list are both treated as "assume stale" by the caller; only a
    positive empty result is proof that nothing needs restarting.
    """
    try:
        from hermes_cli.gateway import find_gateway_pids

        return list(find_gateway_pids(all_profiles=True))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Could not probe for surviving gateways after update: %s", exc)
        return None

def _warn_gateway_restart_phase_aborted(exc: BaseException, pids) -> None:
    """Print a recovery warning when the whole restart phase raised.

    Issue #78574: the gateway auto-restart phase was wrapped in a blanket
    ``except Exception`` that only logged at debug level, so an early failure
    (e.g. importing ``hermes_cli.gateway`` from the freshly pulled checkout)
    erased every drain/restart line from the update output. The update still
    printed "Update complete!" and exited 0 while the running gateway kept
    serving pre-update modules against replaced source files — the next turn
    died with an ImportError.
    """
    print()
    print(f"⚠ Update incomplete — gateway auto-restart failed: {exc}")
    if pids:
        listed = ", ".join(str(pid) for pid in pids)
        print(f"  Gateway process(es) still running pre-update code: {listed}")
    else:
        print("  Any gateway still running is serving pre-update code")
        print("  (mixed sys.modules) against the updated checkout.")
    print("  Restart it manually, then verify:")
    print("    hermes gateway restart")
    print("    hermes gateway status")

def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update.

    The Scheduled Task / Startup-folder launchers (``gateway.cmd`` +
    ``gateway.vbs``) are persistence artifacts written once at install time —
    ``hermes update`` never touched them, so installs created before the
    hidden-console rework (aa2ae36c3f) kept launching the gateway through
    ``pythonw.exe`` forever: every descendant spawn flashed a conhost
    (#54220/#56747) and, since #70344, the console-less gateway died at
    startup with ``RuntimeError: sys.stderr is None`` (#71671).

    The task's /TR points at a stable script path, so rewriting the files in
    place retargets the task without any schtasks call (no UAC needed).
    ``_write_task_script`` is idempotent and renders from current code, so
    this is a no-op for modern installs. Best-effort: a failed refresh must
    never fail the update.
    """
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows

        if not gateway_windows.is_installed():
            return
        gateway_windows._write_task_script()
        print("  ✓ Refreshed Windows gateway launcher scripts")
    except Exception as exc:
        logger.debug("Could not refresh Windows gateway launchers after update: %s", exc)

def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Sync the installer's bootstrap-cache scripts from the fresh checkout.

    The Desktop GUI updater (``hermes-setup.exe``) executes
    ``$HERMES_HOME/bootstrap-cache/install-<ref>.ps1`` (or ``.sh``) for its
    repair/bootstrap stages. Installer binaries built before the #67193
    cache-refresh fix (June 2026 and earlier) NEVER re-download a cached
    branch-ref script — ``install-main.ps1`` cached at install time is
    reused forever, executing months-stale code with long-fixed bugs (the
    2026-08-09 incident: a June 4 cached script's venv stage lacked the
    #81327 process-tree sweep and died on ``Access denied``). The binary
    has no self-update path, so the poisoned cache outlives every
    ``hermes update``.

    Overwriting the cached script for *branch* with the freshly pulled
    ``scripts/install.ps1`` / ``scripts/install.sh`` on every update turns
    the stale binary's unconditional reuse into a feature: it "reuses" a
    file this function keeps permanently current. Post-#67193 installers
    re-download on each run anyway, so for them this is a harmless
    pre-seed of the same bytes.

    Scope guards, mirroring ``install_script.rs``:

    - Only the cache key for the update-target *branch* is rewritten
      (``sanitize_ref``: non ``[A-Za-z0-9._-]`` chars become ``_``, so
      ``bb/gui`` → ``install-bb_gui.ps1``). Sibling mutable refs cache
      DIFFERENT branches' scripts — updating main must not clobber
      ``install-bb_gui.ps1`` with main's script.
    - Commit-SHA pins are immutable by design and never touched. The
      installer's ``is_valid_commit()`` accepts **7–40** hex chars, so an
      abbreviated pin like ``install-4ce1994.ps1`` is just as immutable as
      a full 40-hex one; the sanitized *branch* is additionally required
      to not itself look like a commit pin (defense in depth against a
      caller passing a SHA as the branch).

    The .ps1 copy gets a UTF-8 BOM to match the installer's cache format
    (#67193 encoding fix). Best-effort: a failed refresh must never fail
    the update.
    """
    try:
        import re as _re

        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        # Mirror install_script.rs::sanitize_ref().
        safe_ref = _re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))
        # Mirror install_script.rs::is_valid_commit(): 7-40 hex chars is an
        # immutable commit pin — abbreviated SHAs included. Never rewrite.
        if _re.fullmatch(r"[0-9a-fA-F]{7,40}", safe_ref):
            return
        refreshed = []
        for kind, src_name in (("ps1", "install.ps1"), ("sh", "install.sh")):
            src = _m().PROJECT_ROOT / "scripts" / src_name
            if not src.is_file():
                continue
            cached = cache_dir / f"install-{safe_ref}.{kind}"
            if not cached.is_file():
                continue  # this ref was never bootstrap-cached — nothing to heal
            data = src.read_bytes()
            if kind == "ps1" and not data.startswith(b"\xef\xbb\xbf"):
                # Match the installer's cache format: PowerShell needs the
                # UTF-8 BOM or localized/em-dash text mis-decodes (#67193).
                data = b"\xef\xbb\xbf" + data
            if cached.read_bytes() == data:
                continue  # already current
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
            refreshed.append(cached.name)
        if refreshed:
            print(
                "  ✓ Refreshed installer bootstrap-cache script(s): "
                + ", ".join(sorted(refreshed))
            )
    except Exception as exc:
        logger.debug("Could not refresh bootstrap-cache scripts after update: %s", exc)

def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    if not token or not token.get("resume_needed"):
        return
    token["resume_needed"] = False
    if not _m()._is_windows():
        return

    # Regenerate the persisted launcher scripts before respawning anything,
    # so a legacy pythonw-era Scheduled Task / Startup entry comes back on
    # the current hidden-console design at the next login too.
    _m()._refresh_windows_gateway_launchers()

    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    cold_start = bool(token.get("cold_start_if_installed"))
    if not profiles and not any(u.get("argv") for u in unmapped):
        if cold_start:
            _m()._cold_start_windows_gateway_after_update()
        return

    try:
        from hermes_cli.gateway import (
            launch_detached_gateway_restart_by_cmdline,
            launch_detached_profile_gateway_restart,
        )
    except Exception as exc:
        logger.debug("Could not load Windows gateway restart helper: %s", exc)
        return

    relaunched = []
    for profile, old_pid in sorted(profiles.items()):
        try:
            if launch_detached_profile_gateway_restart(str(profile), int(old_pid)):
                relaunched.append(str(profile))
        except Exception as exc:
            logger.debug(
                "Could not restart Windows gateway profile %s after update: %s",
                profile,
                exc,
            )

    # Respawn unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled
    # Task) by replaying the argv we snapshotted before force-killing them.
    unmapped_relaunched = 0
    for entry in unmapped:
        argv = entry.get("argv")
        old_pid = entry.get("pid")
        if not argv or not old_pid:
            continue
        try:
            if launch_detached_gateway_restart_by_cmdline(int(old_pid), list(argv)):
                unmapped_relaunched += 1
        except Exception as exc:
            logger.debug(
                "Could not restart unmapped Windows gateway (pid %s) after update: %s",
                old_pid,
                exc,
            )

    if relaunched:
        print()
        print(f"  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        if not relaunched:
            print()
        print(
            f"  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)"
        )

def _discard_lockfile_churn(git_cmd, repo_root):
    """Compatibility hook that deliberately never discards user lockfile edits.

    ``package-lock.json`` is ordinary tracked user data.  Older versions used
    ``git checkout --`` here before the stash probe, which silently destroyed an
    intentional edit.  The update transaction now stashes the complete dirty
    tree first; npm may still normalize the lockfile later, but no pre-stash
    cleanup is safe to perform.
    """
    return None

def _normalize_managed_eol(git_cmd, repo_root):
    """Compatibility no-op: never rewrite user bytes before autostash.

    A CRLF-only difference can be intentional user data.  The update transaction
    must preserve it through the complete stash boundary rather than guessing
    that it is machine-generated checkout churn.  The helper remains as a call
    seam for older callers, but all normalization belongs after user state has
    been safely captured (if it is needed at all).
    """
    return None


def _desktop_app_present(desktop_dir: Path) -> bool:
    """Return whether a packaged or source Desktop build exists."""
    return (
        _m()._desktop_packaged_executable(desktop_dir) is not None
        or _m()._desktop_dist_exists(desktop_dir)
    )


def _rebuild_desktop_after_update(
    desktop_dir: Path, *, had_desktop_app_before_update: bool
) -> None:
    """Rebuild an installed Desktop app when its source or artifact changed."""
    # The release tree is ignored by git and can disappear during an update.
    # Its pre-update presence is enough to restore it; do not make people who
    # have never used Desktop pay for an Electron build.
    has_desktop_app = had_desktop_app_before_update or _desktop_app_present(desktop_dir)
    if not (
        (desktop_dir / "package.json").exists()
        and _m()._resolve_node_runtime_npm()
        and has_desktop_app
    ):
        return

    print("→ Checking if desktop app needs rebuilding...")
    # Consult the content-hash stamp IN-PROCESS first. The spawned
    # `hermes desktop --build-only` subprocess re-imports the whole CLI stack
    # (~1-3 s) just to reach the same _m()._desktop_build_needed check; when
    # the stamp already says "up to date" we can skip the spawn entirely. The
    # update path never passes --source, so the subprocess would run with
    # source_mode=False — mirror that here. Any error in the pre-check falls
    # through to the subprocess.
    skip_desktop_build = False
    try:
        skip_desktop_build = not _m()._desktop_build_needed(
            desktop_dir, _m().PROJECT_ROOT, source_mode=False
        )
    except Exception:
        skip_desktop_build = False
    if skip_desktop_build:
        print("  ✓ Desktop app up to date")
        return

    desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
    # Capture the (very loud) Electron/vite build output into update.log
    # instead of streaming it to the terminal. On the rare nonzero exit,
    # retry once after waiting again for the venv — this covers a
    # still-settling rebuild window the first wait didn't fully catch — then
    # surface the captured tail so the failure is debuggable.
    #
    # Start the build subprocess with the Hermes-managed Node on PATH: when
    # `hermes update` runs inside the desktop updater chain (Desktop →
    # hermes-setup → hermes update), the shell PATH customizations are lost,
    # so a bare-PATH child would fail with `node: not found` before cmd_gui can
    # self-heal.
    from hermes_constants import with_hermes_node_path

    build_env = with_hermes_node_path()
    build_result = _m()._run_logged_subprocess(
        desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
    )
    if build_result.returncode != 0:
        build_result = _m()._run_logged_subprocess(
            desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
        )
    if build_result.returncode != 0:
        print("  ⚠ Desktop build failed (non-fatal; run `hermes desktop` to retry)")
        tail = "\n".join((build_result.stdout or "").strip().splitlines()[-15:])
        if tail:
            print(tail)
        from hermes_constants import display_hermes_home as _dhh

        print(f"  Full build log: {_dhh()}/logs/update.log")
    else:
        print("  ✓ Desktop app up to date")


def _cmd_update_impl(args, gateway_mode: bool):
    """Body of ``cmd_update`` — kept separate so the wrapper can always
    restore stdio even on ``sys.exit``."""
    # A managed-runtime refresh can replace site-packages before the normal
    # ``.[all]`` install runs. Snapshot while the old environment can still
    # prove which optional backends the user had activated.
    active_lazy_features = _m()._capture_active_lazy_features()
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    # Snapshot the pre-update version before any code is pulled so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()
    # In gateway mode, use file-based IPC for prompts instead of stdin
    gw_input_fn = (
        (lambda prompt, default="": _gateway_prompt(prompt, default))
        if gateway_mode
        else None
    )
    assume_yes = bool(getattr(args, "yes", False))
    release_tag = getattr(args, "release_tag", None)
    release_repo_lock = None
    release_target = None
    release_transaction_result = None
    release_upgrade_context = None
    release_success_banner_pending = False
    release_completion_banner_pending = False
    update_succeeded = False

    # Whether this update is running without a human at the keyboard.
    # Interactive terminal updates always stash-and-ask (unchanged behavior);
    # only non-interactive updates (desktop/chat app, gateway, `--yes`) consult
    # the `updates.non_interactive_local_changes` config setting to decide
    # whether to auto-restore stashed local source changes or throw them away.
    _non_interactive_update = (
        gateway_mode
        or assume_yes
        or not (sys.stdin.isatty() and sys.stdout.isatty())
    )
    discard_local_changes = False
    if _non_interactive_update:
        try:
            from hermes_cli.config import load_config

            _update_cfg = (load_config() or {}).get("updates", {})
            if isinstance(_update_cfg, dict):
                _mode = str(_update_cfg.get("non_interactive_local_changes", "stash")).lower()
                discard_local_changes = _mode == "discard"
        except Exception as exc:
            # Never let a config read failure change the safe default.
            logger.debug("Could not read updates.non_interactive_local_changes: %s", exc)
            discard_local_changes = False

    print("⚕ Updating Hermes Agent...")
    print()

    # On Windows, abort early if another hermes.exe is holding the venv shim
    # open. Continuing would result in a string of WinError 32 warnings and
    # then either a deferred-rename leftover or a failed git-pull fast path
    # that silently falls back to the slower ZIP route. See issue #26670.
    if _m()._is_windows() and not getattr(args, "force", False):
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            concurrent = _m()._detect_concurrent_hermes_instances(scripts_dir)
            if concurrent:
                print(_format_concurrent_instances_message(concurrent, scripts_dir))
                sys.exit(2)

    # Pre-update backup — runs before any git/file mutation so users can
    # always roll back to the exact state they had before this update.
    # Returns the quick-snapshot id (or None when disabled/failed); the
    # post-update cron-jobs safety net uses it to detect job loss.
    pre_update_snapshot_id = _m()._run_pre_update_backup(args)

    _windows_gateway_resume = _m()._pause_windows_gateways_for_update()
    if _windows_gateway_resume:
        import atexit as _atexit

        _atexit.register(
            _m()._resume_windows_gateways_after_update,
            _windows_gateway_resume,
        )

    # With gateways paused, anything still running from the venv interpreter
    # (most commonly the Desktop app's `hermes serve` backend) will keep .pyd
    # files locked and corrupt the dependency sync below. Refuse rather than
    # race: killing the desktop backend is futile (the app supervises and
    # respawns it), so the user must close the app. Deliberately NOT bypassed
    # by plain --force: the desktop bootstrap updater passes --force to skip
    # the hermes.exe shim guard above, but its lock probe only checks the shim
    # and app.asar — a non-desktop venv python holding a .pyd would sail
    # through and corrupt the sync (the exact failure this guard exists for).
    # --force-venv is the explicit escape hatch.
    if _m()._is_windows() and not getattr(args, "force_venv", False):
        _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            _gateway_holders = _m()._leftover_pausable_gateway_pids(_venv_holders)
            if _gateway_holders is not None:
                # Every remaining holder is a gateway the pause machinery
                # already owns — respawned by its supervisor inside the
                # pause→guard window, or up through a spawn path discovery
                # does not map. Stop them and re-check instead of
                # dead-ending; the post-update resume (and the supervisor
                # that respawned them) brings gateways back afterwards.
                from gateway.status import terminate_pid

                print(
                    f"  ⚠ {len(_gateway_holders)} gateway process(es) still "
                    "hold the venv after the pause; stopping them"
                )
                for _pid in _gateway_holders:
                    try:
                        terminate_pid(int(_pid), force=True)
                    except Exception as exc:
                        logger.debug(
                            "Could not stop leftover gateway %s: %s", _pid, exc
                        )
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            _orphan_backends = _m()._orphaned_desktop_backend_pids(_venv_holders)
            if _orphan_backends:
                # Every remaining holder is a Desktop `serve` backend whose
                # supervising app is GONE — the GUI-updater handoff race:
                # Electron's teardown lost the SIGTERM race, exited, and left
                # its backend (and any .hermes-runtime child) holding the
                # venv. Nothing will respawn an orphan, so reap the tree and
                # re-check instead of dead-ending with "Hermes is still
                # running" while no window is open. Backends whose Desktop
                # is still alive never reach here (_orphaned_desktop_
                # backend_pids returns None for them) — that path keeps the
                # refusal, because the app would just respawn what we kill.
                print(
                    f"  ⚠ {len(_orphan_backends)} orphaned Desktop backend "
                    "process(es) still hold the venv; stopping their trees"
                )
                _m()._stop_process_trees(_orphan_backends)
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
        if _venv_holders:
            print(_format_venv_python_holders_message(_venv_holders))
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(2)

    # Self-lock deferral moved: the venv-holder sweep above excludes this
    # process by design (a CLI `hermes update` IS the venv python), and an
    # updater that has imported a native venv extension cannot rewrite its
    # own mapped .pyd (#83569). That check used to run HERE — before the
    # fetch — but firing pre-fetch meant a deferral stranded the user on the
    # OLD checkout, and any startup path that eagerly loaded cryptography
    # turned every Windows update into an exit-2 loop (#86735/#86780/#86781).
    # It now runs via _abort_dependency_sync_if_self_locked() after the code
    # swap, immediately before the dependency sync — the only phase the lock
    # can actually break — and only when the sync would truly rewrite the
    # loaded distribution.

    # Capture this after every fail-closed venv guard, but before either
    # update path can remove the ignored release tree.
    desktop_dir = _m().PROJECT_ROOT / "apps" / "desktop"
    had_desktop_app_before_update = _desktop_app_present(desktop_dir)

    # Try git-based update first, fall back to ZIP download on Windows
    # when git file I/O is broken (antivirus, NTFS filter drivers, etc.)
    use_zip_update = False
    git_dir = _m().PROJECT_ROOT / ".git"

    if not git_dir.exists():
        if sys.platform == "win32":
            use_zip_update = True
        else:
            print("✗ Not a git repository. Please reinstall:")
            print(
                "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"
            )
            sys.exit(1)

    # On Windows, git can fail with "unable to write loose object file: Invalid argument"
    # due to filesystem atomicity issues. Set the recommended workaround.
    if sys.platform == "win32" and git_dir.exists() and not release_tag:
        subprocess.run(
            [
                "git",
                "-c",
                "windows.appendAtomically=false",
                "config",
                "windows.appendAtomically",
                "false",
            ],
            cwd=_m().PROJECT_ROOT,
            check=False,
            capture_output=True,
        )

    # Build git command once — reused for fork detection and the update itself.
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # Discard npm lockfile churn before any stash/branch logic. npm rewrites
    # tracked package-lock.json files non-deterministically at install/build
    # time (platform-specific optional deps, ideallyInert annotations, etc.),
    # which is never an intentional edit on a managed install but leaves the
    # tree dirty — forcing an autostash on every update and making branch
    # switches fragile. Restoring them first lets the common case (only
    # lockfile churn) update with a clean tree.
    if not release_tag:
        _discard_lockfile_churn(git_cmd, _m().PROJECT_ROOT)
    # Keep the compatibility seam, but never rewrite line endings before the
    # stash boundary: a CRLF-only difference can be intentional user data.
    # The ordinary stash captures it byte-for-byte before branch/update work.
    if not release_tag:
        _normalize_managed_eol(git_cmd, _m().PROJECT_ROOT)

    # Detect if we're updating from a fork (before any branch logic)
    origin_url = _m()._get_origin_url(git_cmd, _m().PROJECT_ROOT)
    is_fork = _is_fork(origin_url)

    if is_fork:
        print("⚠ Updating from fork:")
        print(f"  {origin_url}")
        print()

    if use_zip_update:
        # ZIP-based update for Windows when git is broken
        try:
            _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
        finally:
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        return

    # Fetch and pull
    try:
        if release_tag:
            release_repo_lock = RepositoryUpdateLock(_m().PROJECT_ROOT, git_cmd)
            try:
                release_repo_lock.acquire()
            except RuntimeError as exc:
                print(f"✗ Could not acquire the repository update lock: {exc}")
                sys.exit(1)

        # Resolve the target ref up front so the fetch can be scoped to it.
        # A bare `git fetch origin` pulls every ref, and this repo carries
        # thousands of auto-generated branches — an unscoped fetch can stall for
        # minutes on a non-single-branch checkout. Fetch only what we update
        # against.
        branch = "hermes-release" if release_tag else _m()._resolve_update_branch(args)

        if not release_tag:
            # Self-heal abandoned git lock files (e.g. .git/shallow.lock left
            # by a crashed fetch) before the ordinary update fetch. Release
            # transactions use their repository lock and exact-target fetch.
            from hermes_cli.gitlock import clear_stale_git_locks

            cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
            if cleared:
                print("  (removed stale git lock(s): %s)" % ", ".join(cleared))

        if release_tag:
            print("→ Resolving exact release target...")
            try:
                release_target = _resolve_release_target(
                    git_cmd, _m().PROJECT_ROOT, release_tag
                )
            except RuntimeError as exc:
                print(f"✗ Could not resolve Release {release_tag}.")
                print(f"  {exc}")
                sys.exit(1)
            fetch_result = subprocess.CompletedProcess(
                git_cmd + ["fetch", "--no-tags", "origin"],
                returncode=0,
                stdout="",
                stderr="",
            )
        else:
            print("→ Fetching updates...")
            fetch_result = subprocess.run(
                git_cmd + ["fetch", "origin", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
        if fetch_result.returncode != 0:
            stderr = fetch_result.stderr.strip()
            if "Could not resolve host" in stderr or "unable to access" in stderr:
                print("✗ Network error — cannot reach the remote repository.")
                print(f"  {stderr.splitlines()[0]}" if stderr else "")
            elif (
                "Authentication failed" in stderr or "could not read Username" in stderr
            ):
                print(
                    "✗ Authentication failed — check your git credentials or SSH key."
                )
            else:
                print("✗ Failed to fetch updates from origin.")
                if stderr:
                    print(f"  {stderr.splitlines()[0]}")
            sys.exit(1)

        # Get current branch (returns literal "HEAD" when detached)
        result = subprocess.run(
            git_cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            check=True,
        )
        current_branch = result.stdout.strip()

        parked_branch_switched = False
        if release_tag:
            print(f"→ Target Release: {release_tag}")
            auto_stash_ref = None
        # If user is on a different branch than the update target, switch
        # to the target. When the target is "main" this is the historical
        # "always update against main" behavior; for any other target it's
        # the same thing — get HEAD onto the requested branch first, then
        # fast-forward.
        #
        # Parked-branch guard (2026-08-17 live incident): the checkout can be
        # left parked on a stale feature branch by earlier tooling. Blindly
        # stash-switch-pull-switch-back "updates" main while the running code
        # stays days behind, then prints "✓ Code updated!". Only auto-switch
        # when the parked branch is clean AND fully merged into the target;
        # otherwise warn loudly, mark the code update SKIPPED, and stop
        # before the post-update steps reinforce the stale tree.
        elif current_branch != branch:
            if current_branch != "HEAD":
                switch_safe, switch_block_reason = _m()._assess_parked_branch_switch(
                    git_cmd, _m().PROJECT_ROOT, current_branch, branch
                )
                if not switch_safe:
                    _m()._print_parked_branch_skip_warning(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        current_branch,
                        branch,
                        switch_block_reason,
                    )
                    print()
                    print(
                        "⚠ Update finished — code update SKIPPED"
                        f"{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}"
                    )
                    _m()._resume_windows_gateways_after_update(
                        _windows_gateway_resume
                    )
                    sys.exit(1)
                parked_branch_switched = True
                print(
                    f"  ⚠ Checkout was parked on '{current_branch}' "
                    f"(fully merged) — switching back to {branch}..."
                )
            else:
                print(
                    f"  ⚠ Currently on detached HEAD — switching to {branch} "
                    "for update..."
                )
            # Stash before checkout so uncommitted work isn't lost
            auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)
            checkout_result = subprocess.run(
                git_cmd + ["checkout", branch],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if checkout_result.returncode != 0:
                # Local checkout doesn't have this branch yet. Try to set
                # it up as a tracking branch of origin/<branch>. This is
                # the common case when the requested branch exists upstream
                # but was never checked out locally.
                track_result = subprocess.run(
                    git_cmd + ["checkout", "-B", branch, f"origin/{branch}"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
                if track_result.returncode != 0:
                    # Restore the user's prior branch + stash before bailing
                    # so we don't leave them stranded in a weird state.
                    if auto_stash_ref is not None:
                        _m()._restore_stashed_changes(
                            git_cmd,
                            _m().PROJECT_ROOT,
                            auto_stash_ref,
                            prompt_user=False,
                            input_fn=gw_input_fn,
                        )
                    print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                    if track_result.stderr.strip():
                        print(f"  {track_result.stderr.strip().splitlines()[0]}")
                    sys.exit(1)
        else:
            auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)

        prompt_for_restore = (
            auto_stash_ref is not None
            and not assume_yes
            and (gateway_mode or (sys.stdin.isatty() and sys.stdout.isatty()))
        )

        # Release upgrades compare the exact fetched tag with the maintenance
        # branch. Ordinary updates compare the fetched remote branch and keep
        # the shallow-checkout correction below.
        if release_tag:
            maintenance_sha = _git_resolve_commit(
                git_cmd, _m().PROJECT_ROOT, f"refs/heads/{branch}"
            )
            if maintenance_sha is None:
                print(
                    "✗ Local maintenance branch 'hermes-release' is missing; "
                    "create it from an official release and initialize "
                    "local-patches/.release_base before upgrading."
                )
                sys.exit(1)
            release_merged = subprocess.run(
                git_cmd
                + [
                    "merge-base",
                    "--is-ancestor",
                    release_target.target_sha,
                    f"refs/heads/{branch}",
                ],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            )
            if release_merged.returncode == 0:
                commit_count = 0
            elif release_merged.returncode == 1:
                commit_count = 1
            else:
                print(f"✗ Failed to compare local branch with Release {release_tag}.")
                if release_merged.stderr.strip():
                    print(f"  {release_merged.stderr.strip().splitlines()[0]}")
                sys.exit(1)
        else:
            # On shallow checkouts `rev-list --count` walks the truncated graph
            # and can report the entire remote ancestry. The zero/nonzero gate
            # is still sound, while the correction below recovers the count
            # from the GitHub compare API when possible.
            result = subprocess.run(
                git_cmd + ["rev-list", f"HEAD..origin/{branch}", "--count"],
                cwd=_m().PROJECT_ROOT,
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                check=True,
            )
            commit_count = int(result.stdout.strip())

        if release_tag:
            # Release targets are compared against the exact fetched tag and
            # never use the ordinary remote-branch shallow-count fallback.
            apply_is_shallow = False
        else:
            apply_is_shallow = (
                subprocess.run(
                    git_cmd + ["rev-parse", "--is-shallow-repository"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                ).stdout.strip()
                == "true"
            )
        if not release_tag and commit_count > 0 and apply_is_shallow:
            from hermes_cli.banner import _github_compare_behind

            head_sha = subprocess.run(
                git_cmd + ["rev-parse", "HEAD"],
                cwd=_m().PROJECT_ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            target_sha = subprocess.run(
                git_cmd + ["rev-parse", f"origin/{branch}"],
                cwd=_m().PROJECT_ROOT, capture_output=True,
                text=True, encoding="utf-8", errors="replace",
            ).stdout.strip()
            counted = _github_compare_behind(head_sha, target_sha)
            # counted == 0 means local-ahead (remote tip reachable from HEAD):
            # not behind, fall through to the up-to-date path.
            commit_count = counted if counted is not None else -1

        if commit_count == 0:
            _invalidate_update_cache()

            # Even if origin is up to date, the fork may be behind upstream
            if is_fork and branch == "main":
                _m()._sync_with_upstream_if_needed(git_cmd, _m().PROJECT_ROOT)

            # Restore stash and switch back to original branch if we moved.
            # EXCEPTION: a parked feature branch we verified clean + fully
            # merged stays on the target — re-parking the checkout on the
            # stale branch is the 2026-08-17 incident all over again.
            if auto_stash_ref is not None:
                _m()._restore_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
            if parked_branch_switched:
                print(
                    f"  ✓ Checkout was parked on '{current_branch}' (fully "
                    f"merged) — switched back to {branch}."
                )
            elif current_branch not in {branch, "HEAD"}:
                subprocess.run(
                    git_cmd + ["checkout", current_branch],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                    check=False,
                )

            # "No new commits" does not mean the managed interpreter is safe.
            # uv can retain the same CPython patch while python-build-standalone
            # refreshes the embedded SQLite underneath it. Keep the existing
            # update-boundary hook active on this retry path too.
            from hermes_cli.managed_uv import ensure_uv, update_managed_uv

            runtime_repairs = []
            update_managed_uv(repair_observer=runtime_repairs.append)
            ensure_uv(repair_observer=runtime_repairs.append)
            runtime_repaired = next(
                (result for result in runtime_repairs if result.repaired),
                None,
            )

            # A current checkout does NOT imply a healthy install: a previous
            # dependency sync may have failed partway (classic on Windows,
            # where a running gateway/desktop backend keeps .pyd files locked
            # and uv/pip dies with access-denied, stranding the venv between
            # versions). Probe the venv's core imports and repair if broken —
            # otherwise "Already up to date!" gaslights the user while their
            # install stays bricked.
            healthy, detail = _venv_core_imports_healthy()
            if not healthy:
                print("⚠ Checkout is current, but the venv is unhealthy:")
                print(f"  {detail}")
                print("→ Repairing Python dependencies...")
                # Self-lock deferral (#86735): the repair rewrites the venv
                # too — same mapped-extension hazard as the update sync.
                _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
                _write_update_incomplete_marker()
                from hermes_cli.managed_uv import ensure_uv

                repair_uv = ensure_uv()
                # A managed install whose venv is gone entirely (interrupted
                # repair after the old venv was moved aside) needs the venv
                # recreated before dependencies can be installed into it.
                venv_python_missing = not (
                    venv_python_path(
                        _m().PROJECT_ROOT / "venv", windows=_m()._is_windows()
                    )
                ).exists()
                if venv_python_missing and repair_uv:
                    print("→ Recreating virtual environment...")
                    subprocess.run(
                        [repair_uv, "venv", "venv"],
                        cwd=_m().PROJECT_ROOT,
                        check=False,
                    )
                if repair_uv:
                    repair_env = {**os.environ, "VIRTUAL_ENV": str(_m().PROJECT_ROOT / "venv")}
                    _m()._install_python_dependencies_with_optional_fallback(
                        [repair_uv, "pip"], env=repair_env, group="all"
                    )
                    _m()._refresh_active_lazy_features(
                        [repair_uv, "pip"],
                        env=repair_env,
                        features=active_lazy_features,
                    )
                    _m()._restore_active_tool_dependencies(
                        active_tool_dependencies,
                        [repair_uv, "pip"],
                        env=repair_env,
                    )
                else:
                    _m()._install_python_dependencies_with_optional_fallback(
                        [sys.executable, "-m", "pip"], group="all"
                    )
                    _m()._refresh_active_lazy_features(
                        [sys.executable, "-m", "pip"],
                        features=active_lazy_features,
                    )
                    _m()._restore_active_tool_dependencies(
                        active_tool_dependencies,
                        [sys.executable, "-m", "pip"],
                    )
                _m()._clear_update_incomplete_marker()
                healthy_after, detail_after = _venv_core_imports_healthy()
                if healthy_after:
                    print("✓ Dependencies repaired!")
                    _print_update_completion("✓ Update complete!")
                else:
                    print(f"⚠ Venv still unhealthy after repair: {detail_after}")
                    print("  Close all Hermes windows/gateways and re-run: hermes update")
            else:
                _repair_node_deps_on_current_checkout(_print_update_completion)
            if runtime_repaired is not None and not _m()._is_windows():
                print()
                print(
                    "⚠ Restart required to finish the managed Python runtime repair."
                )
                print(
                    "  Any running Hermes gateways, Desktop backends, or other "
                    "long-lived processes still use the previous runtime."
                )
                print("  Restart each of them to pick up the repaired runtime.")
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            return

        if release_tag:
            print(f"→ Release upgrade available: {release_tag}")
        elif commit_count > 0:
            print(f"→ Found {commit_count} new commit(s)")
        else:
            # Shallow checkout, exact count unrecoverable (offline/rate-limited
            # compare API) — the tips differ, so there IS an update.
            print("→ Updates available (commit count unknown on this shallow checkout)")

        print(
            "→ Promoting isolated release candidate..."
            if release_tag
            else "→ Pulling updates..."
        )
        update_succeeded = False
        # Capture the pre-pull SHA so we can auto-roll-back if the new code
        # has a syntax error in a critical-path file (PR #28452 incident:
        # orphan merge-conflict markers in hermes_cli/config.py bricked
        # every user who ran ``hermes update`` for the 7 minutes between
        # the bad commit and the fix landing).
        pre_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        try:
            if release_tag:
                try:
                    release_transaction_result = _prepare_and_promote_release(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        release_tag,
                        release_target.target_sha,
                        input_fn=gw_input_fn,
                    )
                    release_upgrade_context = getattr(
                        release_transaction_result, "context", None
                    )
                except RuntimeError as exc:
                    print(f"✗ Could not upgrade to Release {release_tag}.")
                    print(f"  {exc}")
                    sys.exit(1)
                pull_result = subprocess.CompletedProcess(
                    ["hermes-release-transaction", release_tag],
                    returncode=0,
                    stdout="",
                    stderr="",
                )
            else:
                # Merge the ref we already fetched above (→ Fetching updates...)
                # instead of `git pull`, which performs a SECOND network fetch of
                # the same branch (~0.5-1.5 s of redundant round-trip per update).
                # `merge --ff-only origin/<branch>` is byte-identical in effect to
                # `pull --ff-only origin <branch>` given the fresh tracking ref;
                # the divergence fallback below is unchanged.
                pull_result = subprocess.run(
                    git_cmd + ["merge", "--ff-only", f"origin/{branch}"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
            if pull_result.returncode != 0 and not release_tag:
                # ff-only failed — local and remote have diverged (e.g. upstream
                # force-pushed or rebase).  Since local changes are already
                # stashed, reset to match the remote exactly.
                print(
                    "  ⚠ Fast-forward not possible (history diverged), resetting to match remote..."
                )
                reset_result = subprocess.run(
                    git_cmd + ["reset", "--hard", f"origin/{branch}"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    text=True, encoding="utf-8", errors="replace",
                )
                if reset_result.returncode != 0:
                    print(f"✗ Failed to reset to origin/{branch}.")
                    if reset_result.stderr.strip():
                        print(f"  {reset_result.stderr.strip()}")
                    print(
                        f"  Try manually: git fetch origin && git reset --hard origin/{branch}"
                    )
                    sys.exit(1)

            # Post-pull syntax guard: validate critical-path files actually
            # parse before declaring the update successful. If a bad commit
            # made it through CI (e.g. admin-merge bypass of a failing
            # ruff check), this catches it on the user side and rolls back
            # so the CLI stays bootable. The user can then retry ``hermes
            # update`` later once a fix lands upstream.
            if release_tag:
                # The isolated candidate was syntax/import validated before
                # promotion. Do not run the ordinary post-pull rollback path
                # against the already-promoted live checkout.
                syntax_ok, failing_path, syntax_error = True, None, None
            else:
                syntax_ok, failing_path, syntax_error = _validate_post_pull_critical_files_syntax(
                    _m().PROJECT_ROOT
                )
            if not syntax_ok:
                print()
                print("✗ Pulled code has a syntax error in a critical file:")
                print(f"  {failing_path}")
                if syntax_error:
                    # py_compile errors can be multi-line; show the first
                    # ~6 lines so the user sees the actual SyntaxError text.
                    for line in str(syntax_error).splitlines()[:6]:
                        print(f"    {line}")
                if pre_pull_sha:
                    print()
                    print(f"→ Rolling back to {pre_pull_sha[:10]}...")
                    rollback_result = subprocess.run(
                        git_cmd + ["reset", "--hard", pre_pull_sha],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                    )
                    if rollback_result.returncode == 0:
                        print("  ✓ Rollback complete — your install is unchanged.")
                        print("  Try ``hermes update`` again later once a fix lands.")
                    else:
                        print("  ✗ Rollback failed. Recover manually with:")
                        print(f"    cd {_m().PROJECT_ROOT} && git reset --hard {pre_pull_sha}")
                        if rollback_result.stderr.strip():
                            print(f"    ({rollback_result.stderr.strip().splitlines()[0]})")
                else:
                    print()
                    print("  Could not capture pre-pull SHA — recover manually with:")
                    print(f"    cd {_m().PROJECT_ROOT} && git reflog && git reset --hard <prev-sha>")
                sys.exit(1)

            update_succeeded = True
            if release_tag:
                # Do not announce success until the durable user-state
                # finalizer has returned True.  A retained recovery journal is
                # a failed command, not a successful upgrade with a warning.
                release_success_banner_pending = True
        finally:
            if auto_stash_ref is not None:
                # Don't attempt stash restore if the code update itself failed —
                # working tree is in an unknown state.
                if not update_succeeded:
                    print(
                        f"  ℹ️  Local changes preserved in stash (ref: {auto_stash_ref})"
                    )
                    print("  Restore manually with: git stash apply")
                elif discard_local_changes:
                    # Non-interactive update + user opted into discarding local
                    # source edits (updates.non_interactive_local_changes:
                    # discard). Throw the stash away instead of re-applying it.
                    _m()._discard_stashed_changes(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        auto_stash_ref,
                    )
                else:
                    _m()._restore_stashed_changes(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        auto_stash_ref,
                        prompt_user=prompt_for_restore,
                        input_fn=gw_input_fn,
                    )

        _invalidate_update_cache()

        # Verify HEAD actually moved (issue #79678). ``merge --ff-only``
        # succeeding only means the merge completed, not that the update
        # applied: a checkout that is pinned to a raw SHA (detached HEAD) can
        # report "N new commit(s)" against origin yet still sit on the old
        # commit afterward (the branch-switch step re-detaches to the SHA).
        # Before this guard, ``hermes update`` printed "✓ Code updated!" and
        # reinstalled deps + rebuilt the desktop app against the stale tree —
        # no error, no warning, ``hermes doctor`` healthy. Compare pre-pull
        # and post-pull HEAD; if they match, surface the no-op instead of
        # claiming success.
        post_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        if not release_tag and pre_pull_sha and post_pull_sha == pre_pull_sha:
            print()
            print("✗ Code did not move — update was a no-op.")
            print(
                f"  HEAD is pinned to {pre_pull_sha[:10]} (detached checkout); "
                f"origin/{branch} advanced but the working tree stayed put."
            )
            print(
                "  Reattach to the branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # And verify HEAD actually sits on the target branch. The parked-
        # branch guard above should make this unreachable, but if any path
        # leaves the checkout attached elsewhere, "✓ Code updated!" would be
        # a lie — refuse to claim success (2026-08-17 incident class).
        post_pull_branch = subprocess.run(
            git_cmd + ["rev-parse", "--abbrev-ref", "HEAD"],
            cwd=_m().PROJECT_ROOT,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
        ).stdout.strip()
        if post_pull_branch and post_pull_branch not in {branch, "HEAD"}:
            print()
            print(
                f"✗ Update pulled origin/{branch}, but the checkout is on "
                f"'{post_pull_branch}' — not claiming success."
            )
            print(
                "  Switch to the target branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # Clear stale .pyc bytecode cache — prevents ImportError on gateway
        # restart when updated source references names that didn't exist in
        # the old bytecode (e.g. get_hermes_home added to hermes_constants).
        removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
        if removed:
            print(
                f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
            )
        _m()._record_bytecode_fingerprint()
        _m()._refresh_bootstrap_cache_scripts(branch)

        # Fork upstream sync logic (only for main branch on forks)
        if is_fork and branch == "main":
            _m()._sync_with_upstream_if_needed(git_cmd, _m().PROJECT_ROOT)

        # Reinstall Python dependencies. Prefer .[all], but if one optional extra
        # breaks on this machine, keep base deps and reinstall the remaining extras
        # individually so update does not silently strip working capabilities.
        #
        # Self-lock deferral (relocated preflight — #86735): if THIS process
        # holds a native extension the sync must rewrite, defer NOW — after
        # the code swap, so only the dependency install is pending and the
        # next fresh launch completes it via the marker.
        _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
        #
        # Drop the core-install breadcrumb BEFORE touching the venv. If the
        # install is killed mid-flight (Ctrl-C, terminal close, WSL OOM), the
        # marker survives and the next ``hermes`` launch finishes the install
        # via ``_recover_from_interrupted_install``. Cleared after the core
        # ``.[all]`` install completes — lazy refresh uses a separate marker.
        _write_update_incomplete_marker()
        print("→ Updating Python dependencies...")
        from hermes_cli.managed_uv import ensure_uv, update_managed_uv

        # Keep managed uv current — runs `uv self update` if we already have one.
        update_managed_uv()

        uv_bin = ensure_uv()

        pip_cmd = [sys.executable, "-m", "pip"]
        if not uv_bin:
            uv_bin = _ensure_uv_for_termux(pip_cmd)
        install_group = "all"

        if uv_bin:
            uv_env = {**os.environ, "VIRTUAL_ENV": str(_m().PROJECT_ROOT / "venv")}
            if _m()._is_termux_env(uv_env):
                uv_env.pop("PYTHONPATH", None)
                uv_env.pop("PYTHONHOME", None)
                install_group = "termux-all"
                print("  → Termux detected: using uv + curated termux-all optional profile...")
            if _m()._is_termux_env(uv_env) and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat([uv_bin, "pip"], env=uv_env)
            _m()._install_python_dependencies_with_optional_fallback(
                [uv_bin, "pip"], env=uv_env, group=install_group
            )
        else:
            # Use sys.executable to explicitly call the venv's pip module,
            # avoiding PEP 668 'externally-managed-environment' errors on Debian/Ubuntu.
            # Some environments lose pip inside the venv; bootstrap it back with
            # ensurepip before trying the editable install.
            pip_cmd = [sys.executable, "-m", "pip"]
            try:
                subprocess.run(
                    pip_cmd + ["--version"],
                    cwd=_m().PROJECT_ROOT,
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError:
                subprocess.run(
                    [sys.executable, "-m", "ensurepip", "--upgrade", "--default-pip"],
                    cwd=_m().PROJECT_ROOT,
                    check=True,
                )
            if _m()._is_termux_env():
                install_group = "termux-all"
                print("  → Termux detected: using curated termux-all optional profile...")
            if _m()._is_termux_env() and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat(pip_cmd)
            _m()._install_python_dependencies_with_optional_fallback(pip_cmd, group=install_group)

        install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
        lazy_env = uv_env if uv_bin else None

        # Core ``.[all]`` install finished. Clear the generic core breadcrumb
        # before the lazy-refresh phase — that phase uses its own marker so a
        # later lazy failure cannot be "healed" by clearing the core marker
        # based on a narrow 7-package import probe (#58004 review).
        _m()._clear_update_incomplete_marker()

        # The update process is still the old Python interpreter process. Run
        # one final cache/module refresh immediately before lazy backend
        # refresh, which imports newly-pulled modules that may depend on fresh
        # symbols in hermes_constants or lazy_deps. The dependency install
        # above may also have regenerated bytecode from build-cache copies —
        # this second sweep catches those stragglers (#60242, #65240).
        removed = _m()._clear_bytecode_cache(_m().PROJECT_ROOT)
        if removed:
            print(
                f"  ✓ Cleared {removed} stale __pycache__ director{'y' if removed == 1 else 'ies'}"
            )
        _m()._record_bytecode_fingerprint()
        _m()._refresh_bootstrap_cache_scripts(branch)
        _m()._reload_updated_runtime_modules()

        # Upgrade pip before lazy refreshes — stale pip can fail source builds
        # and leave partially-written packages (#57828).
        _write_lazy_refresh_incomplete_marker()
        _m()._upgrade_pip_before_lazy_refresh(install_prefix, env=lazy_env)

        # Lazy refresh can corrupt the venv when a backend install fails.
        # Clear the lazy marker only when refresh/repair is confirmed healthy.
        lazy_ok = _m()._refresh_active_lazy_features(
            install_prefix,
            env=lazy_env,
            features=active_lazy_features,
        )
        if lazy_ok:
            _m()._clear_lazy_refresh_incomplete_marker()
        else:
            print(
                "  ⚠ Lazy-refresh recovery incomplete — run `hermes` again "
                "to finish import-based venv repair."
            )

        _m()._restore_active_tool_dependencies(
            active_tool_dependencies,
            install_prefix,
            env=lazy_env,
        )

        # Heal the active memory provider's bridge packages last — the core
        # reinstall + lazy refresh above may have stripped or downgraded
        # plugin.yaml-declared deps that aren't in extras (#53272, #70636).
        _m()._refresh_active_memory_provider_dependencies()

        # Everything that can legitimately produce a transient ImportError has
        # now run (bytecode sweep, dependency reinstall, lazy refresh), so a
        # module that still won't import is real breakage. Warn only — never
        # roll back here: `cannot import name X` is also the signature of the
        # stale-bytecode class (#6207, #60242), and the launch-time sweep in
        # _sweep_stale_bytecode_if_checkout_changed() self-heals that on the
        # next run. A destructive reset would undo a good update over a state
        # that fixes itself.
        import_ok, failing_module, import_error = _validate_critical_modules_import(
            _m().PROJECT_ROOT
        )
        if not import_ok:
            print()
            print(f"  ⚠ {failing_module} still fails to import after updating:")
            print(f"      {import_error}")
            print("    Run `hermes update` again — if it persists, reinstall:")
            print("    https://hermes-agent.nousresearch.com")

        node_failures = _update_node_dependencies()
        _m()._build_web_ui(_m().PROJECT_ROOT / "web")

        _rebuild_desktop_after_update(
            desktop_dir,
            had_desktop_app_before_update=had_desktop_app_before_update,
        )

        if not release_tag:
            print()
            print(f"✓ Code updated!{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}")

        # ── Post-update state.db integrity guard (#68474) ─────────────────
        # Verify that state.db survived the update intact.  If the live file
        # is now corrupted (zeroed, missing header, integrity failure),
        # automatically restore from the pre-update snapshot rather than
        # letting the user discover silently that their sessions are gone.
        try:
            from hermes_cli.backup import _quick_snapshot_root, verify_sqlite_integrity

            _state_path = get_hermes_home() / "state.db"
            if _state_path.exists():
                _state_ok = verify_sqlite_integrity(
                    _state_path,
                    check_header=True,
                    run_pragma=True,
                )
                if _state_ok.get("valid"):
                    logger.debug(
                        "Post-update state.db integrity check: %s",
                        _state_ok.get("message"),
                    )
                else:
                    print()
                    print(
                        "⚠ state.db is corrupted after update: "
                        + _state_ok.get("message", "unknown error")
                    )
                    _pre_snap_id = pre_update_snapshot_id
                    if _pre_snap_id:
                        _snap_state = (
                            _quick_snapshot_root(get_hermes_home())
                            / _pre_snap_id
                            / "state.db"
                        )
                        if _snap_state.exists():
                            _snap_ok = verify_sqlite_integrity(
                                _snap_state, check_header=True, run_pragma=True
                            )
                            if _snap_ok.get("valid"):
                                try:
                                    import shutil as _shutil

                                    _shutil.copy2(_snap_state, _state_path)
                                    _restored_ok = verify_sqlite_integrity(
                                        _state_path,
                                        check_header=True,
                                        run_pragma=True,
                                    )
                                    if _restored_ok.get("valid"):
                                        print(
                                            "  ✓ Auto-restored from pre-update "
                                            f"snapshot ({_pre_snap_id})"
                                        )
                                    else:
                                        print(
                                            "  ✗ Auto-restore FAILED — restored "
                                            "copy also failed integrity"
                                        )
                                except OSError as _exc:
                                    print(
                                        f"  ✗ Auto-restore file copy failed: {_exc}"
                                    )
                            else:
                                print(
                                    "  ✗ Pre-update snapshot also failed integrity"
                                )
                        else:
                            print(
                                "  ⚠ Pre-update snapshot does not contain state.db"
                            )
                    else:
                        print("  ⚠ No pre-update snapshot was taken")
                    print()
        except Exception as exc:
            logger.debug("Post-update state.db integrity check failed: %s", exc)

        # Seed the model-catalog disk cache from the freshly-pulled checkout.
        # The repo ships the canonical catalog at
        # website/static/api/model-catalog.json, and `git pull` just made it
        # current — so copy it straight over ~/.hermes/cache/model_catalog.json
        # instead of waiting on a network fetch (which can be bot-gated or hit a
        # Portal hiccup). Keeps the model picker's curated/free lists in sync
        # with the version the user just installed. Non-fatal on failure: the
        # normal network refresh still applies on the next picker open.
        try:
            from hermes_cli.model_catalog import seed_cache_from_checkout

            if seed_cache_from_checkout(_m().PROJECT_ROOT):
                print("  ✓ Model catalog cache refreshed from checkout")
        except Exception as e:
            logger.debug("Model catalog seed during update failed: %s", e)

        # Sync bundled skills (copies new, updates changed, respects user deletions)
        try:
            from tools.skills_sync import sync_skills

            print()
            print("→ Syncing bundled skills...")
            result = sync_skills(quiet=True)
            if result["copied"]:
                print(f"  + {len(result['copied'])} new: {', '.join(result['copied'])}")
            if result.get("updated"):
                print(
                    f"  ↑ {len(result['updated'])} updated: {', '.join(result['updated'])}"
                )
            if result.get("user_modified"):
                print(f"  ~ {len(result['user_modified'])} user-modified (kept)")
                print(
                    "    → see them: hermes skills list-modified  "
                    "(diff/reset to resume updates)"
                )
            if result.get("cleaned"):
                print(f"  − {len(result['cleaned'])} removed from manifest")
            if result.get("relocated"):
                print(
                    f"  → {len(result['relocated'])} moved to new upstream paths: "
                    f"{', '.join(result['relocated'])}"
                )
            if not result["copied"] and not result.get("updated"):
                print("  ✓ Skills are up to date")
        except Exception as e:
            logger.debug("Skills sync during update failed: %s", e)

        # Sync bundled skills to all profiles (including the active one).
        # seed_profile_skills() uses subprocess with an explicit HERMES_HOME so
        # it is not affected by sync_skills()'s module-level HERMES_HOME cache,
        # which means the active profile is reliably synced regardless of whether
        # the caller's HERMES_HOME env var points at the default or a named profile.
        try:
            from hermes_cli.profiles import (
                list_profiles,
                seed_profile_skills,
            )

            all_profiles = list_profiles()
            if all_profiles:
                print()
                print("→ Syncing bundled skills to all profiles...")
                for p in all_profiles:
                    try:
                        r = seed_profile_skills(p.path, quiet=True)
                        if r and r.get("skipped_opt_out"):
                            status = "opted out (--no-skills)"
                        elif r:
                            copied = len(r.get("copied", []))
                            updated = len(r.get("updated", []))
                            modified = len(r.get("user_modified", []))
                            parts = []
                            if copied:
                                parts.append(f"+{copied} new")
                            if updated:
                                parts.append(f"↑{updated} updated")
                            if modified:
                                parts.append(f"~{modified} user-modified")
                            status = ", ".join(parts) if parts else "up to date"
                        else:
                            status = "sync failed"
                        print(f"  {p.name}: {status}")
                    except Exception as pe:
                        print(f"  {p.name}: error ({pe})")
        except Exception:
            pass  # profiles module not available or no profiles

        # Backfill per-profile .env files for profiles created before the
        # .env-seeding fix (#44792). Copies the default install's .env so
        # those profiles keep the credentials they were effectively using.
        try:
            from hermes_cli.profiles import backfill_profile_envs

            backfilled = backfill_profile_envs(quiet=True)
            if backfilled:
                print()
                print(
                    f"→ Seeded .env for {len(backfilled)} profile(s) "
                    f"(copied from default): {', '.join(backfilled)}"
                )
        except Exception:
            pass  # profiles module not available or no profiles

        # Sync Honcho host blocks to all profiles
        try:
            from plugins.memory.honcho.cli import sync_honcho_profiles_quiet

            synced = sync_honcho_profiles_quiet()
            if synced:
                print(f"\n-> Honcho: synced {synced} profile(s)")
        except Exception:
            pass  # honcho plugin not installed or not configured

        # Check for config migrations.
        #
        # CRITICAL: check_config_version and migrate_config must use
        # freshly-reloaded modules, not the sys.modules cache. The
        # ``hermes update`` process is the PRE-pull Python process — its
        # ``sys.modules`` cache holds the OLD ``hermes_cli.config`` and
        # ``hermes_cli.config_migrations`` from before ``git pull`` updated
        # the source files. A function-level ``from hermes_cli.config import
        # check_config_version`` returns the cached module, so
        # ``DEFAULT_CONFIG["_config_version"]`` is the OLD value and
        # ``check_config_version()`` reports ``(33, 33)`` — "up to date" —
        # even though the freshly-pulled code has v34 with a migration to
        # run. The personality reset migration (#81946) was silently skipped
        # this way, leaving ``display.personality: kawaii`` active after
        # updates that should have reset it.
        print()
        print("→ Checking configuration for new options...")

        # Reload config modules BEFORE any config reads so get_missing_*,
        # check_config_version, and migrate_config all use the updated code.
        _reload_config_modules()

        from hermes_cli.config import (
            get_missing_env_vars,
            get_missing_config_fields,
        )

        missing_env = get_missing_env_vars(required_only=True)
        missing_config = get_missing_config_fields()
        current_ver, latest_ver = _run_config_check_fresh()

        has_new_options = bool(missing_env or missing_config)
        version_bump_only = (
            not has_new_options and current_ver < latest_ver
        )
        needs_migration = has_new_options or current_ver < latest_ver

        if version_bump_only:
            # Nothing for the user to fill in — only the config format version
            # changed (new defaults already merge in transparently). Asking
            # "configure new options now?" here is misleading: saying yes just
            # bumps the version and looks like a no-op (issue: ScottFive /
            # Tt2021). Apply it silently and say what actually happened.
            print()
            print(
                f"  ℹ Updating config format (v{current_ver} → v{latest_ver})…"
            )
            try:
                _mig_results = _run_migrate_config_fresh(
                    interactive=False, quiet=True
                )
                print("  ✓ Config format updated (no new settings to configure)")
                # quiet=True also mutes migration steps that RESET or REMOVE an
                # existing setting (e.g. the v33→v34 personality reset from
                # #81946, which records its note only in the results dict).
                # Re-surface those notes so an unattended update never silently
                # changes user configuration (#86656). In this branch
                # missing_config is empty, so config_added can only contain
                # migration-step mutations, not missing-key listings.
                for _note in _mig_results.get("config_added") or []:
                    print(f"  ℹ {_note}")
                for _warn in _mig_results.get("warnings") or []:
                    print(f"  ⚠️  {_warn}")
            except Exception as _mig_err:
                print(f"  ⚠️  Config format update failed: {_mig_err}")
                print("     Run 'hermes config migrate' to retry.")
        elif needs_migration:
            print()
            # Show WHAT changed, not just a count, so the user can make an
            # informed yes/no decision (previously the prompt named nothing).
            def _print_items(items, label, key, fallback_key=None):
                if not items:
                    return
                print(f"  {label}:")
                shown = items[:8]
                for it in shown:
                    if isinstance(it, dict):
                        name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
                        desc = (it.get("description") or "").strip()
                    else:
                        # Defensive: some callers/mocks pass bare name strings.
                        name = str(it)
                        desc = ""
                    if desc:
                        print(f"      • {name} — {desc}")
                    else:
                        print(f"      • {name}")
                extra = len(items) - len(shown)
                if extra > 0:
                    print(f"      … and {extra} more")

            if missing_env:
                print(
                    f"  ⚠️  {len(missing_env)} new required setting(s) need configuration"
                )
                _print_items(missing_env, "New settings", "name")
            if missing_config:
                print(f"  ℹ️  {len(missing_config)} new config option(s) available")
                _print_items(missing_config, "New options", "key")

            print()
            if assume_yes:
                print(
                    "  ℹ --yes: auto-applying config migration (skipping API-key prompts)."
                )
                response = "y"
            elif gateway_mode:
                response = (
                    _gateway_prompt(
                        "Would you like to configure new options now? [Y/n]", "n"
                    )
                    .strip()
                    .lower()
                )
            elif not (sys.stdin.isatty() and sys.stdout.isatty()):
                print("  ℹ Non-interactive session — applying safe config migrations.")
                response = "auto"
            else:
                try:
                    response = (
                        input("Would you like to configure them now? [Y/n]: ")
                        .strip()
                        .lower()
                    )
                except EOFError:
                    response = "n"
                except UnicodeDecodeError:
                    # input() can raise this when the terminal encoding can't
                    # decode the byte sequence (e.g. a non-UTF-8 locale, or an
                    # embedded terminal). Without this, the exception escapes
                    # here and crashes the update at this prompt.
                    print(
                        "  ⚠ Could not read input (encoding issue). Skipping. "
                        "Run 'hermes config migrate' manually to configure."
                    )
                    response = "n"

            if response in {"", "y", "yes", "auto"}:
                print()
                # Gateway mode, --yes, and non-interactive update contexts
                # (dashboard / web server actions) cannot prompt for API keys.
                # Still run the non-interactive migration pass before restarting
                # so new default config fields and version bumps are written
                # before the freshly updated gateway validates config at startup.
                interactive_migration = not (
                    gateway_mode or assume_yes or response == "auto"
                )
                results = _run_migrate_config_fresh(interactive=interactive_migration, quiet=False)

                if results["env_added"] or results["config_added"]:
                    print()
                    print("✓ Configuration updated!")
                if (gateway_mode or assume_yes or response == "auto") and missing_env:
                    print("  ℹ API keys require manual entry: hermes config migrate")
            else:
                print()
                print("Skipped. Run 'hermes config migrate' later to configure.")
        else:
            print("  ✓ Configuration is up to date")

        # Safety net: config-version migrations have been observed to leave
        # cron/jobs.json valid-but-empty, silently dropping every scheduled
        # job (issue #34600). The desktop scheduler can also overwrite with
        # its own small set, causing partial loss (issue #52144). If the
        # live file now has fewer jobs than the pre-update snapshot, restore
        # it and warn loudly.
        try:
            from hermes_cli.backup import restore_cron_jobs_if_emptied

            cron_restore = restore_cron_jobs_if_emptied(pre_update_snapshot_id)
            if cron_restore:
                print()
                print(
                    "  ⚠️  cron/jobs.json lost jobs during this update — "
                    f"restored {cron_restore['job_count']} job(s) from "
                    f"pre-update snapshot {cron_restore['snapshot_id']}."
                )
        except Exception as exc:
            # Never let the cron safety net break an otherwise-good update.
            logger.debug("Cron jobs auto-restore check failed: %s", exc)

        print()
        if node_failures:
            print(
                "⚠ Update partially complete — Node.js dependencies for "
                f"{', '.join(node_failures)} did not refresh."
            )
            print("  Code and Python deps are updated, but the dashboard/TUI may")
            print("  be in a mixed state until the Node deps are rebuilt.")
        else:
            if release_tag:
                release_completion_banner_pending = True
            else:
                _print_update_completion(_update_complete_message(pre_update_version))

        # Search-index optimization notice (v23). Existing installs keep their
        # working search index untouched on update; the compact v23 layout —
        # which reclaims a large fraction of state.db on heavy users — is
        # opt-in. Surface it here (the moment the user is already thinking
        # about their install) with the exact command and the concrete size
        # win. Show-once-ish: only when a legacy index is actually present.
        try:
            _print_fts_optimize_available_notice()
        except Exception as e:
            logger.debug("FTS optimize notice failed: %s", e)

        # Curator first-run heads-up. Only prints when curator is enabled AND
        # has never run — i.e. the window where the ticker would otherwise
        # have fired against a fresh skill library. Kept silent on steady
        # state so we don't nag.
        try:
            _print_curator_first_run_notice()
        except Exception as e:
            logger.debug("Curator first-run notice failed: %s", e)

        # Most-recent curator run notice — show-once per run. Surfaces the
        # rename map (`old-name → umbrella`) on the high-attention update
        # surface so users learn about consolidations without having to
        # check `hermes curator status`. Self-stamps after printing so it
        # never repeats for the same run.
        try:
            _print_curator_recent_run_notice()
        except Exception as e:
            logger.debug("Curator recent-run notice failed: %s", e)

        # Repair RHEL-family root installs where /usr/local/bin isn't on PATH
        # for non-login interactive shells.  No-op on every other platform.
        try:
            _ensure_fhs_path_guard()
        except Exception as e:
            logger.debug("FHS PATH guard check failed: %s", e)

        # Self-heal the hermes-acp launcher for installs that predate it, so
        # ACP hosts (Zed, JetBrains, Buzz) can resolve Hermes on PATH without
        # a reinstall.  No-op on Windows and when already present.
        try:
            _ensure_acp_launcher()
        except Exception as e:
            logger.debug("hermes-acp launcher self-heal failed: %s", e)

        # Refresh the cua-driver binary used by the Computer Use toolset.
        # The upstream installer is gated on supported platforms and on the
        # binary already being on PATH, so this is a no-op for users who
        # don't have it. Tying the refresh to ``hermes update`` gives users a
        # predictable cadence (matches when they pull new agent code) without
        # adding startup latency or a per-launch GitHub API call.
        try:
            refresh_cua_driver = True
            try:
                from hermes_cli.config import load_config

                _update_cfg = (load_config() or {}).get("updates", {})
                if isinstance(_update_cfg, dict):
                    refresh_cua_driver = bool(
                        _update_cfg.get("refresh_cua_driver", True)
                    )
            except Exception as cfg_exc:
                logger.debug("Could not read updates.refresh_cua_driver: %s", cfg_exc)

            if (
                refresh_cua_driver
                and sys.platform in ("darwin", "win32", "linux")
                and shutil.which("cua-driver")
            ):
                from hermes_cli.tools_config import install_cua_driver

                print()
                print("→ Refreshing cua-driver (Computer Use)...")
                # require_confirmed_update: only run the (multi-minute,
                # silent) upstream installer when the driver's native
                # check-update verb positively reports a newer release.
                # An indeterminate check (offline, rate-limited, old
                # driver) keeps the installed version — `hermes update`
                # must stay fast; `hermes computer-use install --upgrade`
                # remains the force path.
                install_cua_driver(
                    upgrade=True,
                    require_confirmed_update=True,
                    show_installer_progress=False,
                )
        except Exception as e:
            logger.debug("cua-driver refresh failed: %s", e)

        # Write exit code *before* the gateway restart attempt.
        # When running as ``hermes update --gateway`` (spawned by the gateway's
        # /update command), this process lives inside the gateway's systemd
        # cgroup.  A graceful SIGUSR1 restart keeps the drain loop alive long
        # enough for the exit-code marker to be written below, but the
        # fallback ``systemctl restart`` path (see below) kills everything in
        # the cgroup (KillMode=mixed → SIGKILL to remaining processes),
        # including us and the wrapping bash shell.  The shell never reaches
        # its ``printf $status > .update_exit_code`` epilogue, so the
        # exit-code marker file would never be created.  The new gateway's
        # update watcher would then poll for 30 minutes and send a spurious
        # timeout message.
        #
        # Writing the marker here — after git pull + pip install succeed but
        # before we attempt the restart — ensures the new gateway sees it
        # regardless of how we die.
        if gateway_mode:
            _exit_code_path = get_hermes_home() / ".update_exit_code"
            try:
                _exit_code_path.write_text("0", encoding="utf-8")
            except OSError:
                pass

        gateway_fleet_restart_incomplete = False
        # Snapshot of gateways running before we touch anything. Stays empty
        # until we successfully import the probe and are about to stop/drain —
        # so an exception raised before we touch any gateway keeps this empty
        # (nothing to fail closed on), while a failure after we have stopped a
        # discovered gateway lets the handler fail closed on an empty survivor
        # probe rather than reporting a clean update (#78574).
        _pre_restart_gateway_pids: list | None = []
        # Declared outside the restart try/except below (and never reset
        # to None) so it's always safe to read afterwards even if that
        # block raises before reaching its own restart bookkeeping —
        # needed to forward already-restarted units to
        # ``_finish_dashboard_update_cleanup`` (review on #83595).
        restarted_services: list = []

        # Auto-restart ALL gateways after update.
        # The code update (git pull) is shared across all profiles, so every
        # running gateway needs restarting to pick up the new code.
        try:
            from hermes_cli.gateway import (
                is_macos,
                supports_systemd_services,
                _ensure_user_systemd_env,
                find_gateway_pids,
                find_profile_gateway_processes,
                _prepare_profile_gateway_update_restart,
                _get_service_pids,
                _graceful_restart_via_sigusr1,
                _wait_for_gateway_exit,
            )
            import signal as _signal

            def _wait_for_service_active(
                scope_cmd_: list,
                svc_name_: str,
                timeout: float = 10.0,
            ) -> bool:
                """Poll ``systemctl is-active`` until the unit reports active.

                systemd's Stopped -> Started transition after a graceful exit
                (or a hard restart) is not instantaneous; a one-shot check
                races that window and falsely reports the unit as down.
                Poll every 0.5s up to ``timeout`` seconds before giving up.
                """
                deadline = _time.monotonic() + max(timeout, 0.5)
                while True:
                    try:
                        _verify = subprocess.run(
                            scope_cmd_ + ["is-active", svc_name_],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=5,
                        )
                        if _verify.stdout.strip() == "active":
                            return True
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        pass
                    if _time.monotonic() >= deadline:
                        return False
                    _time.sleep(0.5)

            def _service_restart_sec(
                scope_cmd_: list,
                svc_name_: str,
                default: float = 0.0,
            ) -> float:
                """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

                After a graceful exit-75, systemd waits ``RestartSec`` before
                respawning the unit.  Callers that poll for ``is-active``
                must use a timeout >= ``RestartSec`` + transition slack, or
                they'll give up *during* the cooldown window and wrongly
                conclude the unit didn't relaunch.
                """
                try:
                    _show = subprocess.run(
                        scope_cmd_
                        + [
                            "show",
                            svc_name_,
                            "--property=RestartUSec",
                            "--value",
                        ],
                        capture_output=True,
                        text=True, encoding="utf-8", errors="replace",
                        timeout=5,
                    )
                except (FileNotFoundError, subprocess.TimeoutExpired):
                    return default
                raw = (_show.stdout or "").strip()
                # systemd emits values like "30s", "100ms", "1min 30s", or
                # "infinity".  Parse conservatively; on any miss return default.
                if not raw or raw == "infinity":
                    return default
                total = 0.0
                matched = False
                for part in raw.split():
                    for _suf, _mult in (
                        ("ms", 0.001),
                        ("us", 0.000001),
                        ("min", 60.0),
                        ("s", 1.0),
                    ):
                        if part.endswith(_suf):
                            try:
                                total += float(part[: -len(_suf)]) * _mult
                                matched = True
                            except ValueError:
                                pass
                            break
                return total if matched else default

            _manage_cmd_cache: dict = {}

            def _resolve_manage_cmd(scope_: str, scope_cmd_: list, svc_name_: str):
                """Resolve the command prefix for manage-units operations.

                Read-only systemctl calls (``is-active``, ``show``,
                ``list-units``) work unprivileged, but manage-units verbs
                (``reset-failed``, ``start``, ``restart``) on a *system*
                service trigger a polkit ``org.freedesktop.systemd1.manage-units``
                authentication prompt when run as a non-root user.  That
                interactive prompt runs inside our captured subprocess with a
                10-15s timeout — the user sees the prompt flash and "exit
                directly" before they can answer, and the resulting
                TimeoutExpired used to be swallowed silently.

                Strategy: if root, plain systemctl.  If not root, try
                non-interactive sudo (``sudo -n``) — first a blanket probe,
                then a targeted ``systemctl reset-failed`` probe so a
                least-privilege sudoers entry scoped to
                ``systemctl ... hermes-gateway*`` also qualifies
                (``reset-failed`` is an idempotent no-op we run before every
                privileged restart anyway).  If neither works, return None —
                the caller must SKIP the restart (without draining the
                gateway first!) and tell the user how to restart manually.
                ``--no-ask-password`` guarantees polkit can never hang a
                captured subprocess on this path.
                """
                if scope_ in _manage_cmd_cache:
                    return _manage_cmd_cache[scope_]
                cmd = scope_cmd_ + ["--no-ask-password"]
                if (
                    scope_ == "system"
                    and hasattr(os, "geteuid")
                    and os.geteuid() != 0  # windows-footgun: ok — systemd path, Linux-only
                ):
                    sudo_cmd = ["sudo", "-n"] + scope_cmd_ + ["--no-ask-password"]
                    sudo_ok = False
                    try:
                        _probe = subprocess.run(
                            ["sudo", "-n", "true"],
                            capture_output=True,
                            timeout=5,
                        )
                        sudo_ok = _probe.returncode == 0
                        if not sudo_ok:
                            # Blanket sudo refused — a targeted sudoers entry
                            # (NOPASSWD for systemctl ... hermes-gateway*)
                            # may still allow the exact commands we need.
                            _probe = subprocess.run(
                                sudo_cmd + ["reset-failed", svc_name_],
                                capture_output=True,
                                timeout=5,
                            )
                            sudo_ok = _probe.returncode == 0
                    except (FileNotFoundError, subprocess.TimeoutExpired):
                        sudo_ok = False
                    cmd = sudo_cmd if sudo_ok else None
                _manage_cmd_cache[scope_] = cmd
                return cmd

            # Wait budget for graceful SIGUSR1 restarts.  In-band restart
            # may defer stop() until active turns finish
            # (``restart_after_turn_timeout``, #77184) and then spend up to
            # ``restart_drain_timeout`` inside stop(). Cover both phases so
            # we don't fall back to a hard kill while the gateway is still
            # patiently waiting for the requesting turn. On older systemd
            # units without SIGUSR1 wiring this wait just times out and we
            # fall back to ``systemctl restart`` (the old behaviour).
            try:
                from hermes_cli.gateway import _get_restart_exit_wait_budget

                _drain_budget = max(float(_get_restart_exit_wait_budget()), 45.0)
            except Exception:
                _drain_budget = 45.0

            failed_or_stale_units = []
            killed_pids = set()
            relaunched_profiles = []
            externally_supervised_profiles = []

            # Record which gateways are running before any stop/drain, so a
            # later failure that leaves the survivor probe empty can still be
            # recognised as "a running gateway was stopped and did not come
            # back" rather than "nothing was running" (#78574). Best-effort:
            # if the probe itself raises, leave the snapshot as-is (the
            # survivor probe's own None result already fails closed).
            try:
                _pre_restart_gateway_pids = list(find_gateway_pids(all_profiles=True))
            except Exception:
                _pre_restart_gateway_pids = None

            # --- Systemd services (Linux) ---
            # Discover all hermes-gateway* units (default + profiles) plus
            # hermes-serve* units (the Desktop app's backend, #83438).
            if supports_systemd_services():
                try:
                    _ensure_user_systemd_env()
                except Exception:
                    pass

                for scope, scope_cmd in [
                    ("user", ["systemctl", "--user"]),
                    ("system", ["systemctl"]),
                ]:
                    try:
                        result = subprocess.run(
                            scope_cmd
                            + [
                                "list-units",
                                "hermes-gateway*",
                                "hermes-serve*",
                                "--plain",
                                "--no-legend",
                                "--no-pager",
                            ],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=10,
                        )
                    except FileNotFoundError:
                        continue
                    except subprocess.TimeoutExpired as exc:
                        # Discovery timeout — skip this scope, keep the other.
                        print(
                            f"  ⚠ systemctl timed out listing {scope}-scope "
                            f"gateway units ({exc.cmd if exc.cmd else 'unknown command'}). "
                            f"Check the gateway with: hermes gateway status"
                        )
                        continue

                    def _restart_one_systemd_gateway_unit(svc_name: str) -> None:
                        # Check if active
                        check = subprocess.run(
                            scope_cmd + ["is-active", svc_name],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=5,
                        )
                        if check.stdout.strip() != "active":
                            return

                        # Resolve how we may run manage-units verbs
                        # (reset-failed/start/restart) for this scope.
                        # None ⇒ no non-interactive privilege path; we
                        # must avoid those verbs entirely or polkit will
                        # throw an interactive auth prompt inside our
                        # captured 10-15s subprocess (the user sees it
                        # flash and "exit directly" — reported June 2026).
                        _manage_cmd = _resolve_manage_cmd(
                            scope, scope_cmd, svc_name
                        )

                        # Prefer a graceful SIGUSR1 restart so in-flight
                        # agent runs drain instead of being SIGKILLed.
                        # The gateway's SIGUSR1 handler calls
                        # request_restart(via_service=True) → drain →
                        # exit; systemd's Restart=always respawns the unit.
                        # hermes-serve has no such handler (it isn't
                        # gateway/run.py), so skip straight to the blunt
                        # restart below rather than sending it an unhandled
                        # signal and waiting out the drain budget for
                        # nothing.
                        _main_pid = 0
                        if _service_unit_supports_graceful_sigusr1_restart(svc_name):
                            try:
                                _show = subprocess.run(
                                    scope_cmd
                                    + [
                                        "show",
                                        svc_name,
                                        "--property=MainPID",
                                        "--value",
                                    ],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=5,
                                )
                                _main_pid = int((_show.stdout or "").strip() or 0)
                            except (
                                ValueError,
                                subprocess.TimeoutExpired,
                                FileNotFoundError,
                            ):
                                _main_pid = 0

                        _graceful_ok = False
                        if _main_pid > 0:
                            from hermes_cli.gateway import (
                                GATEWAY_LOOP_WEDGED,
                                _escalate_wedged_gateway,
                                probe_gateway_loop_liveness,
                            )

                            if (
                                probe_gateway_loop_liveness(_main_pid)
                                == GATEWAY_LOOP_WEDGED
                            ):
                                # Loop-liveness probe says the gateway's event
                                # loop is provably dead (#81642): SIGUSR1 can
                                # never drain it, so waiting the full budget
                                # (180s default) only wedges the update too.
                                # Bounded escalation (SIGTERM grace → SIGKILL,
                                # ~10s) then restart the unit. A busy gateway
                                # keeps a fresh heartbeat and never takes this
                                # path — its drain (incl. the #86684 cron
                                # floor) is untouched.
                                print(
                                    f"  ⚠ {svc_name}: gateway event loop is "
                                    "unresponsive — skipping drain, forcing "
                                    "a bounded stop..."
                                )
                                _escalate_wedged_gateway(_main_pid)
                                _graceful_ok = True
                            else:
                                print(
                                    f"  → {svc_name}: draining (up to {int(_drain_budget)}s)..."
                                )
                                _graceful_ok = _graceful_restart_via_sigusr1(
                                    _main_pid,
                                    drain_timeout=_drain_budget,
                                )

                        if _graceful_ok:
                            # Gateway exited after a planned restart.
                            # ``Restart=always`` means systemd WILL respawn
                            # the unit — but only after
                            # ``RestartSec`` (default 60s on our unit
                            # file). That 60s wait is a crash-loop guard,
                            # and is the right default when the gateway
                            # dies unexpectedly. For a voluntary restart
                            # on update, it's dead time the user watches.
                            #
                            # Shortcut it: ``reset-failed`` + ``start``
                            # skips RestartSec entirely (we're manually
                            # initiating the unit, not waiting for
                            # systemd's auto-restart logic). Takes about
                            # as long as the process takes to come up
                            # (~1-3s on a warm box).
                            #
                            # If the unit is already active because
                            # RestartSec elapsed while we were draining,
                            # ``start`` is a no-op and we fall through to
                            # the poll below. Either way we collapse the
                            # 60s+ delay to a ~5s one.
                            #
                            # The shortcut needs manage-units privileges.
                            # Without them (system service, non-root, no
                            # passwordless sudo) skip it — systemd's own
                            # auto-restart still relaunches the unit after
                            # RestartSec, no privileges required.
                            if _manage_cmd is not None:
                                subprocess.run(
                                    _manage_cmd + ["reset-failed", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=10,
                                )
                                subprocess.run(
                                    _manage_cmd + ["start", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=15,
                                )
                                # Short poll: the gateway should be up
                                # within a few seconds now that we
                                # bypassed RestartSec.
                                if _wait_for_service_active(
                                    scope_cmd,
                                    svc_name,
                                    timeout=10.0,
                                ):
                                    restarted_services.append(svc_name)
                                    return
                            # Passive poll: systemd's auto-restart fires
                            # after RestartSec regardless of privileges.
                            # This is the primary path when _manage_cmd is
                            # None, and the fallback when the explicit
                            # start didn't take.
                            _restart_sec = _service_restart_sec(
                                scope_cmd,
                                svc_name,
                                default=0.0,
                            )
                            _post_drain_timeout = max(
                                10.0,
                                _restart_sec + 10.0,
                            )
                            if _manage_cmd is None and _restart_sec > 5.0:
                                print(
                                    f"  → {svc_name}: waiting for systemd "
                                    f"auto-restart (~{int(_restart_sec)}s; "
                                    "no root for an immediate restart)..."
                                )
                            if _wait_for_service_active(
                                scope_cmd,
                                svc_name,
                                timeout=_post_drain_timeout,
                            ):
                                restarted_services.append(svc_name)
                                return
                            # Process exited but wasn't respawned (older
                            # unit without Restart=on-failure or
                            # RestartForceExitStatus=75).  Fall through
                            # to systemctl start/restart.
                            print(
                                f"  ⚠ {svc_name} drained but didn't relaunch — forcing restart"
                            )

                        # Forcing a restart requires manage-units
                        # privileges.  Without a non-interactive path,
                        # running systemctl here would spawn a polkit
                        # auth prompt inside a captured 10-15s subprocess
                        # — it flashes and dies before the user can
                        # answer.  Skip with clear instructions instead.
                        if _manage_cmd is None:
                            failed_or_stale_units.append(svc_name)
                            print(
                                f"  ⚠ {svc_name} is a system service and restarting it needs root.\n"
                                f"    Restart it manually to load the new version:\n"
                                f"      sudo systemctl restart {svc_name}\n"
                                f"    To let `hermes update` restart it automatically, allow\n"
                                f"    passwordless sudo for systemctl, or run updates with sudo."
                            )
                            return

                        # Fallback: blunt systemctl restart.  This is
                        # what the old code always did; we get here only
                        # when the graceful path failed (unit missing
                        # SIGUSR1 wiring, drain exceeded the budget,
                        # restart-policy mismatch).
                        #
                        # Always `reset-failed` first.  If systemd's own
                        # auto-restart attempts already parked the unit
                        # in a failed state (transient CHDIR / OOM /
                        # filesystem race after our drain + exit-75),
                        # a plain `systemctl restart` can wedge against
                        # the RestartSec backoff and leave the unit
                        # dead.  Clearing the failed state first makes
                        # the restart idempotent.  Mirrors the recovery
                        # path in `hermes gateway restart`
                        # (`systemd_restart()`) as of PR #20949.
                        subprocess.run(
                            _manage_cmd + ["reset-failed", svc_name],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=10,
                        )
                        restart = subprocess.run(
                            _manage_cmd + ["restart", svc_name],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=15,
                        )
                        if restart.returncode == 0:
                            # Verify the service actually survived the
                            # restart.  systemctl restart returns 0 even
                            # if the new process crashes immediately.
                            if _wait_for_service_active(
                                scope_cmd,
                                svc_name,
                                timeout=10.0,
                            ):
                                restarted_services.append(svc_name)
                            else:
                                # Retry once — transient startup failures
                                # (stale module cache, import race) often
                                # resolve on the second attempt.  Again
                                # clear any failed state first so the
                                # retry isn't blocked by the previous
                                # crash.
                                print(
                                    f"  ⚠ {svc_name} died after restart, retrying..."
                                )
                                subprocess.run(
                                    _manage_cmd + ["reset-failed", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=10,
                                )
                                subprocess.run(
                                    _manage_cmd + ["restart", svc_name],
                                    capture_output=True,
                                    text=True, encoding="utf-8", errors="replace",
                                    timeout=15,
                                )
                                if _wait_for_service_active(
                                    scope_cmd,
                                    svc_name,
                                    timeout=10.0,
                                ):
                                    restarted_services.append(svc_name)
                                    print(f"  ✓ {svc_name} recovered on retry")
                                else:
                                    failed_or_stale_units.append(svc_name)
                                    _scope_flag = "--user " if scope == "user" else ""
                                    _sudo_hint = "sudo " if scope == "system" else ""
                                    print(
                                        f"  ✗ {svc_name} failed to stay running after restart.\n"
                                        f"    Check logs: {_sudo_hint}journalctl {_scope_flag}-u {svc_name} --since '2 min ago'\n"
                                        f"    Recover manually:\n"
                                        f"      {_sudo_hint}systemctl {_scope_flag}reset-failed {svc_name}\n"
                                        f"      {_sudo_hint}systemctl {_scope_flag}restart {svc_name}"
                                    )
                        else:
                            failed_or_stale_units.append(svc_name)
                            print(
                                f"  ⚠ Failed to restart {svc_name}: {restart.stderr.strip()}"
                            )

                    def _on_unit_timeout(svc_name: str, exc: subprocess.TimeoutExpired) -> None:
                        # Isolate the timeout to this unit and keep going
                        # (#68523). A scope-wide handler used to abort every
                        # later gateway and leave the fleet on mixed code.
                        failed_or_stale_units.append(svc_name)
                        print(
                            f"  ⚠ systemctl timed out restarting {svc_name} "
                            f"({exc.cmd if exc.cmd else 'unknown command'}); "
                            f"continuing with remaining gateways"
                        )

                    _for_each_systemd_gateway_unit(
                        result.stdout,
                        process_unit=_restart_one_systemd_gateway_unit,
                        on_unit_timeout=_on_unit_timeout,
                    )

            # --- Launchd services (macOS) ---
            if is_macos():
                try:
                    from hermes_cli.gateway import (
                        launchd_restart,
                        get_launchd_label,
                        get_launchd_plist_path,
                    )

                    plist_path = get_launchd_plist_path()
                    if plist_path.exists():
                        check = subprocess.run(
                            ["launchctl", "list", get_launchd_label()],
                            capture_output=True,
                            text=True, encoding="utf-8", errors="replace",
                            timeout=5,
                        )
                        if check.returncode == 0:
                            try:
                                launchd_restart()
                                restarted_services.append(get_launchd_label())
                            except subprocess.CalledProcessError as e:
                                stderr = (getattr(e, "stderr", "") or "").strip()
                                print(f"  ⚠ Gateway restart failed: {stderr}")
                except (FileNotFoundError, subprocess.TimeoutExpired, ImportError):
                    pass

            # --- Manual (non-service) gateways ---
            # Kill any remaining gateway processes not managed by a service.
            # Exclude PIDs that belong to just-restarted services so we don't
            # immediately kill the process that systemd/launchd just spawned.
            service_pids = _get_service_pids()
            manual_pids = find_gateway_pids(
                exclude_pids=service_pids, all_profiles=True
            )
            profile_processes = {
                proc.pid: proc
                for proc in find_profile_gateway_processes(exclude_pids=service_pids)
                if proc.pid in manual_pids
            }
            for pid, proc in profile_processes.items():
                restart_mode = _prepare_profile_gateway_update_restart(
                    proc.profile, pid
                )
                if restart_mode is None:
                    continue
                # Prefer a graceful SIGUSR1 drain so in-flight agent runs
                # finish before the watcher respawns the gateway.  If the
                # gateway doesn't support SIGUSR1 or doesn't exit within
                # the drain budget, fall back to SIGTERM — the watcher
                # still sees the exit and relaunches either way.
                # Announce the drain first: this wait can hold for the full
                # budget per gateway with no other output, and on surfaces
                # that stream update progress (the desktop updater most of
                # all) the silence reads as a hung update (#44515).
                print(
                    f"  → {proc.profile}: draining gateway PID {pid} "
                    f"(up to {int(_drain_budget)}s)..."
                )
                from hermes_cli.gateway import (
                    GATEWAY_LOOP_WEDGED,
                    _escalate_wedged_gateway,
                    probe_gateway_loop_liveness,
                )

                if probe_gateway_loop_liveness(pid) == GATEWAY_LOOP_WEDGED:
                    # Loop-liveness probe: this gateway's event loop is
                    # provably dead (#81642) — SIGUSR1/SIGTERM shutdown can
                    # never run, so the drain wait would burn the full budget
                    # and stall the update. Bounded stop instead (SIGTERM
                    # grace → SIGKILL, ~10s). A busy-but-alive gateway keeps
                    # a fresh heartbeat and never takes this branch, so live
                    # drains (incl. the #86684 cron floor) are unaffected.
                    print(
                        f"  ⚠ {proc.profile}: gateway event loop is "
                        "unresponsive — skipping drain, forcing a bounded stop..."
                    )
                    _escalate_wedged_gateway(pid)
                    drained = True
                else:
                    drained = _graceful_restart_via_sigusr1(
                        pid,
                        drain_timeout=_drain_budget,
                    )
                if not drained:
                    try:
                        os.kill(pid, _signal.SIGTERM)
                    except (ProcessLookupError, PermissionError):
                        pass
                # Wait for the old process to fully exit before the watcher
                # spawns the new gateway.  Telegram holds the previous
                # getUpdates long-poll session open on its servers for up to
                # ~30s after the client disconnects.  If the new gateway
                # connects before that window expires it receives a 409
                # Conflict, which _handle_polling_conflict() recovers from
                # via back-off retries — but a brief wait here reduces the
                # chance of hitting that path at all, especially on fast
                # machines where the watcher loop restarts in < 1s.
                # We wait up to 5s for the process to exit (the OS-level
                # close, not the Telegram server-side expiry), then let the
                # watcher take over.  The Telegram adapter's retry logic
                # handles any remaining 409s if the server session is still
                # live when the new gateway polls.
                _wait_for_gateway_exit(timeout=5.0, force_after=None)
                killed_pids.add(pid)
                if restart_mode == "external-supervisor":
                    externally_supervised_profiles.append(proc.profile)
                else:
                    relaunched_profiles.append(proc.profile)

            for pid in manual_pids:
                if pid in profile_processes:
                    continue
                try:
                    os.kill(pid, _signal.SIGTERM)
                    killed_pids.add(pid)
                except (ProcessLookupError, PermissionError):
                    pass

            if restarted_services or killed_pids:
                print()
                for svc in restarted_services:
                    print(f"  ✓ Restarted {svc}")
                if relaunched_profiles:
                    names = ", ".join(relaunched_profiles)
                    print(f"  ✓ Restarting manual gateway profile(s): {names}")
                if externally_supervised_profiles:
                    names = ", ".join(externally_supervised_profiles)
                    print(
                        "  ✓ Handed gateway profile(s) back to their external "
                        f"supervisor: {names}"
                    )
                unmapped_count = (
                    len(killed_pids)
                    - len(relaunched_profiles)
                    - len(externally_supervised_profiles)
                )
                if unmapped_count:
                    print(f"  → Stopped {unmapped_count} manual gateway process(es)")
                    print("    Restart manually: hermes gateway run")
                    if unmapped_count > 1:
                        print(
                            "    (or: hermes -p <profile> gateway run  for each profile)"
                        )

            if failed_or_stale_units:
                gateway_fleet_restart_incomplete = True
                if gateway_mode:
                    _exit_code_path = get_hermes_home() / ".update_exit_code"
                    try:
                        _exit_code_path.write_text("1", encoding="utf-8")
                    except OSError:
                        pass
            _warn_incomplete_gateway_fleet_restart(failed_or_stale_units)

            if not restarted_services and not killed_pids:
                # No gateways were running — nothing to do
                pass

            # --- Post-restart survivor sweep -----------------------------
            # Issue #17648: some gateways ignore SIGTERM (stuck drain,
            # blocked I/O, PID dead but zombie).  The detached profile
            # watchers wait 120s for the old PID to exit — if it never
            # does, no respawn happens and the user keeps hitting
            # ImportError against a stale sys.modules.  Give the
            # graceful paths a brief window to complete, then SIGKILL
            # any remaining pre-update PIDs so the watcher / service
            # manager can relaunch with fresh code.
            try:
                _time.sleep(3.0)
                _service_pids_after = _get_service_pids()
                _surviving = find_gateway_pids(
                    exclude_pids=_service_pids_after,
                    all_profiles=True,
                )
                # Scope to PIDs we already tried to kill during this
                # update (killed_pids).  Anything new is a gateway that
                # started AFTER our restart attempt — respecting user
                # intent, we don't kill those.
                _stuck = [pid for pid in _surviving if pid in killed_pids]
                if _stuck:
                    print()
                    print(
                        f"  ⚠ {len(_stuck)} gateway process(es) ignored SIGTERM — force-killing"
                    )
                    from gateway.status import terminate_pid as _terminate_pid
                    for pid in _stuck:
                        try:
                            # Routes through taskkill /T /F on Windows,
                            # SIGKILL on POSIX — _signal.SIGKILL doesn't
                            # exist on Windows so the old raw os.kill call
                            # used to crash the entire update path.
                            _terminate_pid(pid, force=True)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    # Give the OS a beat to reap the processes so the
                    # watchers see them exit and respawn.
                    _time.sleep(1.5)
            except Exception as _sweep_exc:
                logger.debug("Post-restart survivor sweep failed: %s", _sweep_exc)

        except Exception as e:
            logger.debug("Gateway restart during update failed: %s", e)
            # An exception escaping the whole phase means the drain/restart
            # output the user relies on never printed. Don't let that pass for
            # a clean update: surface it and treat the fleet as stale unless we
            # can positively prove no gateway is running (#78574).
            #
            # A positive-empty ``_surviving`` is only proof-of-safety when
            # nothing was running before we touched anything. If a gateway was
            # discovered pre-restart and none survive now, it was stopped and
            # its replacement was never verified — the same fail-open contract
            # this fix closes — so we must still fail closed on ``[]``.
            _surviving = _surviving_gateway_pids_after_failed_restart()
            if _restart_phase_failure_is_incomplete(
                _surviving, _pre_restart_gateway_pids
            ):
                gateway_fleet_restart_incomplete = True
                _warn_gateway_restart_phase_aborted(e, _surviving)
                if gateway_mode:
                    _exit_code_path = get_hermes_home() / ".update_exit_code"
                    try:
                        _exit_code_path.write_text("1", encoding="utf-8")
                    except OSError:
                        pass

        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)

        # Warn if legacy Hermes gateway unit files are still installed.
        # When both hermes.service (from a pre-rename install) and the
        # current hermes-gateway.service are enabled, they SIGTERM-fight
        # for the same bot token (see PR #11909). Flagging here means
        # every `hermes update` surfaces the issue until the user migrates.
        try:
            from hermes_cli.gateway import (
                has_legacy_hermes_units,
                _find_legacy_hermes_units,
                supports_systemd_services,
            )

            if supports_systemd_services() and has_legacy_hermes_units():
                print()
                print("⚠ Legacy Hermes gateway unit(s) detected:")
                for name, path, is_sys in _find_legacy_hermes_units():
                    scope = "system" if is_sys else "user"
                    print(f"    {path}  ({scope} scope)")
                print()
                print("  These pre-rename units (hermes.service) fight the current")
                print("  hermes-gateway.service for the bot token and cause SIGTERM")
                print("  flap loops. Remove them with:")
                print()
                print("    hermes gateway migrate-legacy")
                print()
                print("  (add `sudo` if any are in system scope)")
        except Exception as e:
            logger.debug("Legacy unit check during update failed: %s", e)

        # Restart a managed dashboard through systemd, or stop stale manual
        # dashboard processes. Raw-killing a systemd-owned dashboard PID makes
        # systemd treat it as a clean stop, leaving the Cloudflare origin dead.
        # Preserve the safety rule above: a failed Node refresh leaves the
        # currently running dashboard untouched.
        #
        # Forward the systemd units restarted above (includes hermes-serve*,
        # #83438) so a Serve-only install's freshly restarted process isn't
        # found and restarted again below (review on #83595).
        _finish_dashboard_update_cleanup(
            node_failures, already_restarted_units=set(restarted_services)
        )

        print()
        print("Tip: You can now select a provider and model:")
        print("  hermes model              # Select provider and model")

        if gateway_fleet_restart_incomplete:
            # Code update itself succeeded, but at least one gateway still
            # runs pre-update modules — surface that as a failed update so
            # automation / operators do not treat the fleet as healthy.
            sys.exit(1)

    except subprocess.CalledProcessError as e:
        if release_tag:
            if (
                release_transaction_result is not None
                or release_upgrade_context is not None
                or update_succeeded
            ):
                print("✗ Release transaction was promoted, but a post-promotion step failed.")
                print(f"  {e}")
                print(
                    "  Recovery/finalization will run before the repository lock is released; "
                    "the ZIP fallback is intentionally disabled for release upgrades."
                )
            else:
                print(f"✗ Release update failed before promotion: {e}")
                print("  The ZIP fallback is intentionally disabled for release upgrades.")
            sys.exit(1)
        if _m()._is_windows() and not update_succeeded:
            print(f"⚠ Git update failed: {e}")
            print("→ Falling back to ZIP download...")
            print()
            _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
        else:
            print(f"✗ Update failed: {e}")
            sys.exit(1)
    finally:
        # Finalization is the release transaction's sole top-level success
        # owner.  Capture the active exception before entering it so a failed
        # cleanup can never replace the operation failure already in flight.
        primary_exc_info = sys.exc_info()
        try:
            release_finalization_verified = False
            if release_upgrade_context is not None:
                release_finalization_verified = _finalize_release_upgrade_for_orchestration(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    release_upgrade_context,
                    input_fn=gw_input_fn,
                    primary_exc_info=primary_exc_info,
                )
            if (
                release_finalization_verified
                and primary_exc_info[0] is None
            ):
                if release_success_banner_pending:
                    print()
                    print("✓ Code updated!")
                if release_completion_banner_pending:
                    print("✓ Update complete!")
        finally:
            # Keep lock release outside the finalizer's failure path, exactly
            # once, while the release context is still protected above.
            if release_repo_lock is not None:
                release_repo_lock.release()

# --- Hoisted from the body of _cmd_update_impl (self-contained, no closure state) ---

def _restart_phase_failure_is_incomplete(surviving, pre_restart_pids) -> bool:
    """Whether an escaped gateway-restart-phase exception must fail the update.

    Fail closed unless we can positively prove the fleet is safe:

    * ``surviving is None`` — the survivor probe could not determine state
      (typically the freshly-pulled ``hermes_cli.gateway`` no longer imports,
      one of the ways the phase aborts). Assume stale.
    * ``surviving`` non-empty — a gateway is still running pre-update code.
    * ``surviving == []`` — nothing is running now. That is proof-of-safety
      ONLY when nothing was running before we touched anything. If a gateway
      was discovered pre-restart (``pre_restart_pids`` non-empty, or ``None``
      meaning the pre-state could not be read), it was stopped without a
      verified replacement, so we still fail closed (#78574).
    """
    if surviving is None or surviving:
        return True
    # surviving == []: safe only if we know nothing was running beforehand.
    return pre_restart_pids is None or bool(pre_restart_pids)


def _print_items(items, label, key, fallback_key=None):
    if not items:
        return
    print(f"  {label}:")
    shown = items[:8]
    for it in shown:
        if isinstance(it, dict):
            name = it.get(key) or (fallback_key and it.get(fallback_key)) or "?"
            desc = (it.get("description") or "").strip()
        else:
            # Defensive: some callers/mocks pass bare name strings.
            name = str(it)
            desc = ""
        if desc:
            print(f"      • {name} — {desc}")
        else:
            print(f"      • {name}")
    extra = len(items) - len(shown)
    if extra > 0:
        print(f"      … and {extra} more")

def _wait_for_service_active(
    scope_cmd_: list,
    svc_name_: str,
    timeout: float = 10.0,
) -> bool:
    """Poll ``systemctl is-active`` until the unit reports active.

    systemd's Stopped -> Started transition after a graceful exit
    (or a hard restart) is not instantaneous; a one-shot check
    races that window and falsely reports the unit as down.
    Poll every 0.5s up to ``timeout`` seconds before giving up.
    """
    deadline = _time.monotonic() + max(timeout, 0.5)
    while True:
        try:
            _verify = subprocess.run(
                scope_cmd_ + ["is-active", svc_name_],
                capture_output=True,
                text=True, encoding="utf-8", errors="replace",
                timeout=5,
            )
            if _verify.stdout.strip() == "active":
                return True
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
        if _time.monotonic() >= deadline:
            return False
        _time.sleep(0.5)

def _service_restart_sec(
    scope_cmd_: list,
    svc_name_: str,
    default: float = 0.0,
) -> float:
    """Read the unit's ``RestartUSec`` (RestartSec) in seconds.

    After a graceful exit-75, systemd waits ``RestartSec`` before
    respawning the unit.  Callers that poll for ``is-active``
    must use a timeout >= ``RestartSec`` + transition slack, or
    they'll give up *during* the cooldown window and wrongly
    conclude the unit didn't relaunch.
    """
    try:
        _show = subprocess.run(
            scope_cmd_
            + [
                "show",
                svc_name_,
                "--property=RestartUSec",
                "--value",
            ],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return default
    raw = (_show.stdout or "").strip()
    # systemd emits values like "30s", "100ms", "1min 30s", or
    # "infinity".  Parse conservatively; on any miss return default.
    if not raw or raw == "infinity":
        return default
    total = 0.0
    matched = False
    for part in raw.split():
        for _suf, _mult in (
            ("ms", 0.001),
            ("us", 0.000001),
            ("min", 60.0),
            ("s", 1.0),
        ):
            if part.endswith(_suf):
                try:
                    total += float(part[: -len(_suf)]) * _mult
                    matched = True
                except ValueError:
                    pass
                break
    return total if matched else default
