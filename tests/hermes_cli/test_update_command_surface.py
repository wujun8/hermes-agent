"""Command-level regression coverage for the public update surface."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


pytestmark = pytest.mark.live_system_guard_bypass

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run_cli(tmp_path: Path, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env["HERMES_HOME"] = str(tmp_path / ".hermes")
    return subprocess.run(
        [sys.executable, "-m", "hermes_cli.main", *args],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )


def test_upgrade_help_is_rejected_as_an_unknown_command(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "upgrade", "--help")

    assert result.returncode != 0
    assert "invalid choice" in result.stderr


def test_update_help_keeps_the_official_update_options(tmp_path: Path) -> None:
    result = _run_cli(tmp_path, "update", "--help")

    assert result.returncode == 0
    for expected in ("--branch", "default (main)", "--backup", "--no-backup"):
        assert expected in result.stdout
