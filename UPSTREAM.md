# Upstreams and chassis provenance

`src/passing_check/__init__.py` is the single source of truth for every sha below;
`tests/test_imports.py` asserts this file names exactly those shas.

## 1. Chassis: copied from `lag-ladder` with provenance

- Repo: https://github.com/hossainpazooki/lag-ladder, read at
  `87956359ef88e85b2c8e7a6fa61325614c8dcb4c` (the committed files at that commit, 2026-11-25).
- lag-ladder adapted these modules from linear-ceiling; its own `UPSTREAM.md` carries that step.
- Copied, not shared, so a repo under active amendment cannot move this one's gates.

| what | filePath at that commit | here | change |
|---|---|---|---|
| ledger lint: structure, chain, block diff, verdict-cell provenance | `src/lag_ladder/ledger_check.py` | `src/passing_check/ledger_check.py` | package name, `prog` string, the docstring's provenance paragraph |
| scope-sentence lint | `src/lag_ladder/lint_scope.py` | `src/passing_check/lint_scope.py` | `SCOPE_SENTENCE` is this repo's; `EXEMPT` is empty; the docstring's provenance paragraph |
| ledger lint tests | `tests/test_ledger_check.py` | `tests/test_ledger_check.py` | package name, docstring |
| scope lint tests | `tests/test_lint_scope.py` | `tests/test_lint_scope.py` | same cases, re-anchored on this repo's sentence |
| git fixtures | `tests/conftest.py` | `tests/conftest.py` | git helpers verbatim; the seal fixtures dropped |
| CI workflow | `.github/workflows/ci.yml` | `.github/workflows/ci.yml` | package name; the seal step dropped |
| license text | `LICENSE` | `LICENSE` | verbatim |

Not copied yet: `hashing`, `rng`, `seal`, `upstream_gate` and `config`. They come, with their tests and
a row here, in the change that first registers a sealed prediction.

## 2. The two records this one continues: observed, not pinned

- Policy-update record: https://github.com/hossainpazooki/lag-ladder, `main` observed at the chassis
  commit above.
- Handoff record: https://github.com/hossainpazooki/linear-ceiling, `main` observed at
  `a82ff6e176abe8e6b2dd4e912b086569ad4f6e2b` on 2026-11-25.
- Rule: nothing in this repo imports from either or writes into either. Their definitions are cited by
  ledger entry number at the commits above. When an experiment here invokes a tool of theirs, a
  numbered entry pins the commit and the invoked paths first, and this section gains the pin.
