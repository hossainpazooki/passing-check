# Provenance of copied code

Nothing here imports from or invokes another repository. This file says only where copied code came
from: the ledger tooling (first table) and parts of the handoff instruments (second table).

- Source: https://github.com/hossainpazooki/lag-ladder, the committed files at
  `87956359ef88e85b2c8e7a6fa61325614c8dcb4c` (read 2026-11-25).
- lag-ladder adapted these modules from linear-ceiling; its own `UPSTREAM.md` carries that step.
- Copied, not shared, so a change there cannot move this repo's gates.

| what | path at that commit | here | change |
|---|---|---|---|
| ledger lint: structure, chain, block diff, verdict-cell provenance | `src/lag_ladder/ledger_check.py` | `src/passing_check/ledger_check.py` | package name, `prog` string, docstring |
| scope-sentence lint | `src/lag_ladder/lint_scope.py` | `src/passing_check/lint_scope.py` | `SCOPE_SENTENCE` is this repo's; `EXEMPT` is empty; docstring |
| ledger lint tests | `tests/test_ledger_check.py` | `tests/test_ledger_check.py` | package name, docstring |
| scope lint tests | `tests/test_lint_scope.py` | `tests/test_lint_scope.py` | same cases, re-anchored on this repo's sentence |
| git fixtures | `tests/conftest.py` | `tests/conftest.py` | git helpers verbatim; the seal fixtures dropped |
| CI workflow | `.github/workflows/ci.yml` | `.github/workflows/ci.yml` | package name; the seal step dropped |
| license text | `LICENSE` | `LICENSE` | verbatim |

## Handoff instruments

- Source: https://github.com/hossainpazooki/linear-ceiling, the committed files at
  `a82ff6e176abe8e6b2dd4e912b086569ad4f6e2b` (read 2026-11-28).
- Copied functions are trimmed to what Run A calls. The driver `run_a.py`, `names.py` and the tests are
  written here and copy nothing.

| what | path at that commit | here | change |
|---|---|---|---|
| trajectory loader for the composio_swekit files | `src/linear_ceiling/e7_swe.py`, `e7_traces.py` | `src/passing_check/traces.py` | token counting and other suites dropped; a message's role is read from its type or class name as a whole word (the source matched substrings, so every `langchain` id read as "ai"); the loader also returns which messages are recorded responses |
| handoffs and the token alignment | `src/linear_ceiling/e9_align.py` | `src/passing_check/traces.py` | a handoff must land on a recorded response; the recorded response is kept; exclusions moved to the driver |
| cache assembly, random candidate, logit comparison | `tools/cache_behavior/core.py` | `src/passing_check/cache.py` | the recompute budget and the attention observer dropped; `key_deviation` is the source's `token_deviation`; `fork` and `greedy` are new |
| quantile convention, seeded generator | `src/linear_ceiling/e7_stats.py`, `rng.py` | `src/passing_check/stats.py` | merged into one file |

On the public traces the loader here finds the same 64 sender and receiver texts as the source's loader
(compared by digest on 2026-11-28).
