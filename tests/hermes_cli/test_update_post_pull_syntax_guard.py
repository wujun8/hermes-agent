"""Tests for the ordinary post-pull syntax guard in ``hermes update``.

Release candidates use a strict validator: every critical path must exist and
be a regular, contained file.  Ordinary updates retain compatibility with
older checkout layouts: they compile every critical file that exists, while
still rejecting unsafe paths and syntax errors in those files.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from hermes_cli import main as hermes_main
from hermes_cli import update_cmd


# ---------------------------------------------------------------------------
# _capture_head_sha
# ---------------------------------------------------------------------------


def test_capture_head_sha_returns_stripped_sha(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        assert cmd[-2:] == ["rev-parse", "HEAD"]
        return SimpleNamespace(stdout="deadbeefcafe\n", returncode=0)

    monkeypatch.setattr(hermes_main.subprocess, "run", fake_run)

    assert hermes_main._capture_head_sha(["git"], tmp_path) == "deadbeefcafe"


# ---------------------------------------------------------------------------
# Ordinary post-pull compatibility validator
# ---------------------------------------------------------------------------


def _populate_critical_tree(root: Path, *, broken_file: str | None = None) -> None:
    """Create stub files for every entry in the production critical manifest."""
    broken_payload = (
        "x = {\n"
        '    "a": 1,\n'
        "<<<<<<< HEAD\n"
        '    "b": 2,\n'
        "=======\n"
        '    "c": 0b6d673e7,\n'
        ">>>>>>> 0b6d673e7\n"
        "}\n"
    )
    for relpath in update_cmd._UPDATE_CRITICAL_FILES:
        path = root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        if relpath == broken_file:
            path.write_text(broken_payload)
        else:
            path.write_text("# stub\n")


def test_validate_post_pull_critical_files_syntax_tolerates_missing_files(tmp_path):
    """Older layouts may omit a critical file; ordinary updates tolerate it."""
    for relpath in update_cmd._UPDATE_CRITICAL_FILES:
        if relpath == "hermes_constants.py":
            continue
        path = tmp_path / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# stub\n")

    ok, failing_path, error = update_cmd._validate_post_pull_critical_files_syntax(tmp_path)

    assert (ok, failing_path, error) == (True, None, None)


def test_validate_post_pull_critical_files_syntax_rejects_existing_syntax_error(tmp_path):
    _populate_critical_tree(tmp_path, broken_file="hermes_cli/config.py")

    ok, failing_path, error = update_cmd._validate_post_pull_critical_files_syntax(tmp_path)

    assert ok is False
    assert failing_path == str(tmp_path / "hermes_cli/config.py")
    assert error and "syntax" in error.lower()


def test_validate_post_pull_critical_files_syntax_rejects_existing_symlink(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(update_cmd, "_UPDATE_CRITICAL_FILES", ("required.py",))
    outside = tmp_path.parent / "post-pull-outside.py"
    outside.write_text("VALUE = 1\n")
    try:
        (tmp_path / "required.py").symlink_to(outside)
        ok, failing_path, error = update_cmd._validate_post_pull_critical_files_syntax(tmp_path)
    finally:
        outside.unlink(missing_ok=True)

    assert ok is False
    assert failing_path == str(tmp_path / "required.py")
    assert error and "symlink" in error.lower()
