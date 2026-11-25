"""Adapted from lag-ladder tests/test_imports.py: this repo's shas; the instrument and results cases are
dropped because no instrument is pinned and no results tree is tracked."""
import re

import passing_check


def test_repo_root_is_the_repo():
    assert (passing_check.REPO_ROOT / "pyproject.toml").exists()


def test_every_sha_in_upstream_md_is_declared_and_vice_versa():
    text = (passing_check.REPO_ROOT / "UPSTREAM.md").read_text(encoding="utf-8")
    shas = set(re.findall(r"\b[0-9a-f]{40}\b", text))
    assert shas == {passing_check.CHASSIS_SHA, passing_check.POLICY_RECORD_OBSERVED_SHA,
                    passing_check.HANDOFF_RECORD_OBSERVED_SHA}, shas


def test_nothing_under_src_imports_the_two_records():
    """The two records are cited and, later, invoked at a pinned commit. They are never imported."""
    for p in (passing_check.REPO_ROOT / "src").rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not re.search(r"^\s*(from|import)\s+(lag_ladder|linear_ceiling|kvt)\b", text, re.M), p.name
