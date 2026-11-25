# passing-check: agent contract

A pre-registered experiment record on what a passing KV cache reuse check establishes. `README.md` says
what the record is; `ledger/ledger.md` is the source of truth for hypotheses, rules, results and
verdicts. This file holds only the stable rules, for any agent. `CLAUDE.md` adds the operator's Claude
Code notes on top of it and restates nothing from it.

## Invariants
- `ledger/ledger.md` is append-only by numbered entry and hash-chained in file order;
  `ledger_check --against <rev>` makes the trailing entry immutable too. A verdict cell changes only
  through an entry with a `verdict:` line.
- Never write a number into the ledger or a doc that was not recomputed from raw output by a
  summarizer. A fact from outside the repo carries a primary source and a retrieval date, or the word
  "unverified".
- Never edit a hypothesis after its experiment starts. Never edit a committed entry. Append.
- `../linear-ceiling` and `../lag-ladder` are read-only here. Nothing under `src/` imports from them.
  A definition of theirs is cited by entry number and commit; a tool of theirs is invoked only at a
  commit that a numbered entry here pins (`UPSTREAM.md`).
- The chassis is copied from lag-ladder, not shared. A change there is not a change here until an
  entry says so.
- Raw output (`results/`, `data/`, `traces/`) never enters git history.
- The scope sentence is held verbatim in `README.md`; `lint_scope` refuses a paraphrase anywhere else.
- Ledger entries are short: what was registered, ruled or measured, its sources, its status.
- Build what the next step calls. A module nothing imports does not belong here yet.
- Working documents (handoffs, learnings, designs in progress, paper drafts) stay out of the
  repository.

## Where to read current state
Not restated here. Read `ledger/ledger.md`: the hypothesis table, then the entries its cells cite.

## Before changing anything
- Run `git status`: parallel sessions leave uncommitted work in a tree.
- Leave to the operator, or do only on the operator's explicit instruction: appending an entry, taking
  or reinterpreting a ruling, requesting or releasing a GPU, pinning an upstream commit, editing a
  registered config.

## Checks: the definition of done
`$PY` is the repo venv: `.venv/Scripts/python.exe` (Windows) or `.venv/bin/python` (Linux, macOS).
```
$PY -m pytest -q                       # synthetic, offline
$PY -m passing_check.lint_scope
$PY -m passing_check.ledger_check      # --against <rev> in CI
```
A green suite proves the tools work, not a scientific claim. Green means the CI run
(`.github/workflows/ci.yml`, Linux).

## Layout
- `src/passing_check/`: `ledger_check` (structure, chain, block diff, verdict-cell provenance) and
  `lint_scope` (the scope sentence). `tests/` mirrors it.
- `ledger/ledger.md`: the record.
- `UPSTREAM.md`: the provenance of everything copied, and the commits of the two records this one
  continues. `src/passing_check/__init__.py` is the single source of truth for every sha in it.
