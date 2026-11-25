"""Shared fixtures. Adapted from lag-ladder tests/conftest.py: the git helpers are verbatim; the seal
fixtures are dropped with the seal, which this repo does not carry yet."""
import subprocess
from pathlib import Path

import pytest

GIT_ID = ["-c", "user.name=test", "-c", "user.email=test@example.com"]


def git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", *GIT_ID, *args], cwd=repo, check=True,
                          capture_output=True, text=True).stdout


@pytest.fixture
def repo(tmp_path) -> Path:
    """A fresh git repo with one commit and an empty ledger directory."""
    r = tmp_path / "repo"
    (r / "ledger").mkdir(parents=True)
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    git(r, "add", "README.md")
    git(r, "commit", "-q", "-m", "init")
    return r


def commit_all(repo: Path, msg: str = "ledger") -> None:
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", msg)
