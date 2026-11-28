# passing-check: agent contract

A pre-registered experiment record on what a passing KV cache reuse check establishes. `README.md` says
what the record is; `ledger/ledger.md` is the source of truth for hypotheses, rules, results and
verdicts. This file holds only the stable rules, for any agent. `CLAUDE.md` adds the operator's Claude
Code notes on top of it and restates nothing from it.

## Invariants
- The record stands alone. No code here imports from or invokes another repository, and no rule here
  is defined by reference to another record: a statistic, tolerance, threshold or control is stated in
  full by the entry that registers it. `tests/test_imports.py` holds the import half of this: `src/`
  imports the standard library, this package, and numpy, torch and transformers.
- `ledger/ledger.md` is append-only by numbered entry and hash-chained in file order;
  `ledger_check --against <rev>` makes the trailing entry immutable too. A verdict cell changes only
  through an entry with a `verdict:` line.
- Never write a number into the ledger or a doc that was not recomputed from this repository's raw
  output by a summarizer. A fact from outside the repo carries a primary source and a retrieval date,
  or the word "unverified".
- Never edit a hypothesis after its experiment starts. Never edit a committed entry. Append.
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
  or reinterpreting a ruling, requesting or releasing a GPU, adding a dependency, editing a registered
  config.

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
  `lint_scope` (the scope sentence); `traces` (handoffs in the public trajectories, token alignment),
  `cache` (the reused and the random cache, the key deviation, logits, greedy decode), `names` (what a
  continuation names), `stats`, and `run_a` (prepare, run, summarize). `tests/` covers each on CPU.
- `config/run-a.toml`: Run A's settings. Proposed until a ledger entry names the file by digest.
- `config/run-a-scaled.toml`: the noise scales of Run A's reduced-size pass; the same rule.
- `ledger/ledger.md`: the record.
- `UPSTREAM.md`: where the ledger tooling was copied from. Provenance only; nothing is imported or invoked.
