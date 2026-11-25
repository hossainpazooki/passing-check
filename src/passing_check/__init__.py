"""passing-check: the ledger and its gates for one question -- what a passing cache reuse check establishes.

Nothing here fits, dumps or scores a model. The chassis is copied from lag-ladder with provenance (UPSTREAM.md).
"""
from pathlib import Path

__version__ = "0.0.1"

# src/passing_check/__init__.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]

# Single source of truth for every sha; UPSTREAM.md repeats them for humans and tests/test_imports.py
# asserts the two agree (every 40-hex sha in UPSTREAM.md is one of these, and each of these is there).
CHASSIS_REPO = "https://github.com/hossainpazooki/lag-ladder"
CHASSIS_SHA = "87956359ef88e85b2c8e7a6fa61325614c8dcb4c"      # the commit the copied modules were read at (2026-11-25)

# The two records this one continues. OBSERVED, NOT PINNED: nothing here invokes them yet.
POLICY_RECORD_REPO = CHASSIS_REPO
POLICY_RECORD_OBSERVED_SHA = CHASSIS_SHA
HANDOFF_RECORD_REPO = "https://github.com/hossainpazooki/linear-ceiling"
HANDOFF_RECORD_OBSERVED_SHA = "a82ff6e176abe8e6b2dd4e912b086569ad4f6e2b"   # main on 2026-11-25
