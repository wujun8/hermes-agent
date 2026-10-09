"""A desktop check must not reinterpret unreadable channel config as main."""
from pathlib import Path
import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize("content", ["update: [", "- not-a-mapping", b"\xff"])
def test_invalid_config_refuses_desktop_channel_probe(tmp_path, content):
    home = tmp_path / "home"
    home.mkdir()
    config = home / "config.yaml"
    config.write_bytes(content.encode() if isinstance(content, str) else content)
    source_root = Path(__file__).resolve().parents[2]
    root = tmp_path / "install-root"
    root.mkdir()
    (root / "README.md").write_text("Source check fixture\n", encoding="utf-8")
    git_env = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    for args in (
        ["init"],
        ["add", "README.md"],
        ["-c", "user.name=Fixture", "-c", "user.email=fixture@example.invalid",
         "-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", "commit", "-m", "Fixture"],
    ):
        subprocess.run(["git", *args], cwd=root, env=git_env, check=True,
                       capture_output=True, text=True, timeout=10)
    result = subprocess.run(
        [sys.executable, "-m", "hermes_cli.source_check", "--install-root", str(root), "--home", str(home)],
        cwd=source_root, env={**os.environ, "HERMES_HOME": str(home), "HERMES_IGNORE_USER_CONFIG": "0"},
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert '"channel": "main"' not in result.stdout
    assert "config" in result.stderr
    assert config.read_bytes() == (content.encode() if isinstance(content, str) else content)
