# passing-check

When a cache reuse check passes, what has it established?

A language model's KV cache stores what the model computed for a prompt. It can be reused in place of
recomputation after the text before it has changed, as at an agent handoff, or after the weights have
changed, as at a policy update. Reuse is accepted on a check: the reused keys agree with fresh ones
within a tolerance, or the reader's predictions stay under a threshold on a validation text. This
repository is the pre-registered record of what such a pass does and does not say about what the
reader outputs next.

It continues two records. [`linear-ceiling`](https://github.com/hossainpazooki/linear-ceiling) reads a
cache carried across a change of context, and [`lag-ladder`](https://github.com/hossainpazooki/lag-ladder)
reads one carried across a change of weights.

## Scope

> This record reads, on one cache at a time, whether a cache reuse check passes and whether the reader's
> predictions, answers and task outcome under that cache differ from a fresh read; it does not propose a
> check, set an acceptance threshold, or measure savings or latency.

## What is read, and where it is defined

| level | statistic | defined in |
|---|---|---|
| keys | f\*(τ): the smallest fraction of matched tokens an oracle removes before the mean key deviation of the rest is within a tolerance τ | linear-ceiling ledger 0023, as 0045 and 0058 correct it |
| predictions | the mean per-token KL between the next-token distributions with and without reuse | lag-ladder ledger 0011 and 0012; linear-ceiling ledger 0047 |
| answers | the share of prompts whose greedy final answer changes | lag-ladder ledger 0025 |
| task | not defined yet | a registering entry in this ledger |

The definitions are cited, not recomputed. A rule or a threshold applies here only once an entry in
this ledger adopts it.

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

Apache-2.0. See `LICENSE` and `NOTICE`.
