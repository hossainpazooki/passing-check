"""passing-check: the ledger and its gates for one question -- what a passing cache reuse check establishes.

The package stands alone: it imports the standard library, itself, and numpy, torch and transformers, and
tests/test_imports.py refuses anything else until the allowed set is widened on purpose.
"""
from pathlib import Path

__version__ = "0.0.1"

# src/passing_check/__init__.py -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[2]
