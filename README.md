# passing-check

When a cache reuse check passes, what has it established?

A language model's KV cache stores what the model computed for a prompt. It can be reused in place of
recomputation after the text before it has changed, as at an agent handoff, or after the weights have
changed, as at a policy update. Reuse is accepted on a check: the reused keys agree with fresh ones
within a tolerance, or the reader's predictions stay under a threshold on a validation text. This
repository is the pre-registered record of what such a pass does and does not say about what the
reader outputs next.

The record stands alone. Every statistic, tolerance, threshold and control it uses is defined in full
by the ledger entry that registers it, and its code imports from and invokes no other repository.

## Scope

> This record reads, on one cache at a time, whether a cache reuse check passes and whether the reader's
> predictions, answers and task outcome under that cache differ from a fresh read; it does not propose a
> check, set an acceptance threshold, or measure savings or latency.

## The four levels

| level | the question at that level |
|---|---|
| keys | do the reused keys agree with fresh ones? |
| predictions | does the next-token distribution move? |
| answers | does the generated answer change? |
| task | does the graded result change? |

A statistic for a level exists here once an entry registers it; the ledger says which are.

## Status

`ledger/ledger.md` is the status: the hypothesis table first, then the entries its cells cite. This
file restates none of it.

## How the record stays auditable

- A hypothesis is registered before its run, and its verdict is stated against the rule as written.
- Entries are numbered, dated and immutable. Each one hashes the entries above it, and CI refuses an
  edit to any entry already committed. A correction is a new entry.
- A number enters the ledger only from a summarizer that recomputes it from raw output.
- Failed predictions stay on the record.

## Setup

Python 3.12.

```
python -m venv .venv
.venv/Scripts/python.exe -m pip install -e ".[dev]"    # .venv/bin/python on macOS and Linux
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m passing_check.lint_scope
.venv/Scripts/python.exe -m passing_check.ledger_check
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`. `UPSTREAM.md` says where the ledger tooling came from.
