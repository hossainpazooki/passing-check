# Ledger — passing-check

House style, borrowed with provenance {sourceRepo: lag-ladder, filePath: ledger/ledger.md,
commitSha: 87956359ef88e85b2c8e7a6fa61325614c8dcb4c} (which borrowed it from linear-ceiling):
hypotheses are pre-registered before any run; verdicts are stated against the rule as written, before
considering which outcome is more interesting to report; entries are numbered, dated and immutable —
an amendment is a new entry, never an edit; status tags are `[VALIDATED]` (ran, and survived an
independent attempt to refute it), `[BASELINE]` (ran; numbers here), `[STRETCH]` (designed, not run),
`[FUTURE]` (not designed), `[SUPERSEDED]`.

From entry 0002 on, every entry records `prior-entries-sha256:` over the entries section above it.
A verdict cell changes only with a machine-readable line `verdict: H-XX = <VERDICT>` in a numbered
entry. Both are recomputed by `ledger_check` in CI, which also refuses any edit to an entry block
already committed at the base revision (the trailing entry included).

## Hypotheses (pre-registered; verdict column is the only cell that ever changes, and only via a numbered entry)

| id | statement | decided by | verdict |
|---|---|---|---|

No hypothesis is registered yet.

## Entries

### 0001 — 2026-11-25 — Repo created; chassis, sources and scope on record

**What this repo is.** The record for one question: when a cache reuse check passes, what does that
establish about the reader's output? It continues two records, linear-ceiling (a cache carried across a
change of context) and lag-ladder (a cache carried across a change of weights), and reads the check and
the outcome on the same cache.

**Separate repo, by ruling (2026-11-25).** The chassis is copied, not shared: `ledger_check` and
`lint_scope` with their tests, from lag-ladder at `87956359ef88e85b2c8e7a6fa61325614c8dcb4c`.
`UPSTREAM.md` carries `{sourceRepo, filePath, commitSha}` and what changed in each file. `hashing`,
`rng`, `seal`, `upstream_gate` and `config` are not copied yet; they come with the first registration
that seals a prediction.

**Definitions this record may cite, not recompute.** None is a rule here until a registering entry
adopts it.
- The key check: f\*(τ), the oracle removal fraction, with τ_K = 0.3186. linear-ceiling ledger 0023,
  read as 0045 and 0058 correct it, at commit `a82ff6e176abe8e6b2dd4e912b086569ad4f6e2b`.
- The prediction reading: the mean per-token KL between the reader's next-token distributions under the
  fresh and the reused cache, with the bands HOLDS_B at or under 1e-3 and DEGRADES_B at or over 1e-1
  nats per token. lag-ladder ledger 0011 and 0012, at the chassis commit. The same reading at handoffs,
  with no threshold: linear-ceiling ledger 0047.
- The answer reading: the share of prompts whose greedy final answer differs under the two caches.
  lag-ladder ledger 0025.

**Instruments.** None pinned. The two records are observed at the commits above, not invoked; a pin
comes by numbered entry with the first registration.

**Scope sentence** (held verbatim in README.md; `lint_scope` refuses a paraphrase anywhere else): This
record reads, on one cache at a time, whether a cache reuse check passes and whether the reader's
predictions, answers and task outcome under that cache differ from a fresh read; it does not propose a
check, set an acceptance threshold, or measure savings or latency.

**Status.** `[FUTURE]` Nothing is registered and nothing has run.
