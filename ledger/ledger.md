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

### 0002 — 2026-11-28 — Run A registered: keys, predictions and the continuation read on one cache at each handoff; every definition stated here

prior-entries-sha256: 41c75b51d8fdc0167aa3b3c121635216ff2ac686f6a0de2a2681e81d7b86f439

**Stands alone.** Every definition this run uses is in this entry. Entry 0001's paragraph "Definitions this
record may cite, not recompute" and its sentence on pinning instruments are withdrawn. In the scope sentence,
"task outcome" means here what the receiver hands forward, read as below; no resolved task is read.

**Registered by digest.** `config/run-a.toml` (`9f6044953691e11a825feab7ece07863975a4b7afd0a88491bf45cec74be4984`); the code is `src/passing_check/` at the commit
that carries this entry, and the run records each file's digest. Reader `Qwen/Qwen3-1.7B` at revision
`70d244cc86cc`, 32-bit floats. Nothing has run on a card.

**Corpus.** Sixty public trajectory files of one agent system on SWE-bench Verified: the first 30 instances
in key order of each of the submissions `20241016_composio_swekit` and `20241025_composio_swekit`, from
`s3://swe-bench-submissions/verified/<submission>/trajs/`. Not a random draw. The sha256 of their 60-line
`sha256sum` list (binary mode, paths from the submissions' parent folder, byte order) is
`2dd9019d8006a4afd634e1de0f98422870a21c3584b233ce8dafb72cada7577a`. That these are the first 30 of 498 and 499
listed is from linear-ceiling's ledger at `a82ff6e` (listing retrieved 2026-09-01); not re-listed here. A
handoff is a recorded response whose serving model differs from the model on the non-user message before it.
S is every message up to that one; R is the prompt of the receiving call; both joined by newlines. The 60
files hold 64. In each the receiver is asked for a
summary of the agent's actions. The matched set M is the tokens in the matching blocks of
`difflib.SequenceMatcher(None, S, R, autojunk=False)`.

**Cells, never pooled.** `native`: S, and R with its continuation, within 32,768 tokens. `long`: within 81,920
and not both within 32,768, read with the position schedule scaled by 2.5 (YaRN). A handoff outside a cell
is excluded and counted, never truncated. Prepared on 2026-11-28: `native` holds 25 handoffs and `long` 35; the
other 4 exceed 81,920 tokens.

**The prompt.** R as one user message in the reader's chat template, reply opened, thinking off. Template
tokens are never matched, so every arm computes the end of the prompt.

**Arms, each against FRESH** (the reader's own pass over the prompt):
- REUSE: the reader's keys and values for M from its pass over S, keys moved to their positions in R under the
  model's own rotary schedule; every other token computed against the cache so assembled.
- RANDOM: the fresh keys AND values of M, each plus isotropic noise whose norm, per token, layer and head, is
  REUSE's error on that key or that value. So both arms change keys and values by the same amounts; only the
  direction of the change differs. One noise draw per handoff, seeded from the config seed and the handoff's
  place in the run order.

**Readings.**
- Key deviation, per matched token: the squared distance between an arm's key and the fresh key, position
  removed, over that layer-head's mean squared distance of a fresh matched key from the matched-set mean;
  averaged over layers and heads. Per handoff: mean, p90, maximum. RANDOM equals REUSE on it by construction.
- Predictions: on the first 256 tokens of the recorded response, teacher-forced, KL(FRESH ‖ arm) per token and
  whether the top token agrees.
- Continuation: a greedy decode of 384 tokens from the bare prompt, read up to its first end-of-turn token
  (`<|im_end|>` or `<|endoftext|>`) and without template tokens. Against FRESH's: the position of the
  first differing token, agreement over the first 32, identical or not, and the common share: the longest
  common subsequence of the two decodes, as read, over the length of the longer. One changed first word moves the first two to their floor
  and leaves the common share readable, so the common share leads.
- Names: what a decode puts in backticks that is one file path or identifier (`names.named`: the span cut at
  its first "(", not a lowercase word under four letters, not a language constant). Prose is never searched.
  A name is a backticked span and nothing else: an arm that names the same file without backticks reads as
  naming nothing, so an arm's loss of formatting counts as a loss of names. Beside it, per arm, the number of
  decodes that reach an end of turn within the 384 tokens.
  Named-set overlap is the Jaccard index with FRESH's set. Grounding is the share of an arm's names found
  verbatim in R, without the template.
- Fallback, fixed now and applied by the summarizer: a FRESH decode is degenerate if it has fewer than 20
  distinct words (lowercased runs of letters, digits and underscores) or names nothing. If more than a quarter
  of a cell's handoffs are degenerate, the names readings are not reported for that cell and the continuation
  readings stand alone.

**Tolerances.** No threshold is proposed. The anchor is the median over handoffs of the key deviation between
unrelated matched tokens. The ladder is 1, 0.3, 0.1, 0.03 and 0.01 of it; REUSE passes a step when its mean
key deviation is within it. There is no lower anchor: two FRESH passes give identical states, which the first
control reads on the logits and the decode.

**Rules the code applies, stated so no definition lives only in the code.** A handoff enters a cell only if
R, wrapped, leaves room for 640 more tokens (256 scored and 384 decoded) under the cell's cap; a handoff
with an empty R, fewer than two tokens of recorded response or fewer than two matched tokens is excluded and
counted. The "unrelated" token for a matched token is the one half the matched set away. A name loses
its trailing full stops; a lowercase candidate under four characters is dropped; a backticked span is one
line of at most 200 characters. The position of the first differing token reads -1 for identical decodes and
is summarized over the changed handoffs only. Words are runs of ASCII letters,
digits and underscores. Two empty named sets read overlap 1; two empty decodes read common share 1, and one
empty decode reads 0. Agreement over the first 32 counts a position past the end of either decode as a
disagreement. A median over an even count is the mean of the middle two. A rank correlation is undefined
when either side is constant; such resamples are dropped and counted.

**Controls; a failure stops the run and is not relaxed afterwards.** On the first handoff of the run order
(the largest sender): a second FRESH pass with logit error exactly 0 and an identical decode; a copy of the
reader's own states for the first 128 prompt tokens, their keys moved to position 0 and back by the relocation
REUSE uses, with logit error at most 0.005 (it read 1.8e-4 on a CPU; a wrong assembly moves logits by order 1,
so the limit leaves room for a card's rounding and still catches an error). On every handoff, before any output is
read under an arm: RANDOM's mean key deviation within 0.1% of REUSE's; and REUSE's first-layer key deviation
at most 1e-6 on every matched token and head, the maximum being read (a first-layer key depends on its token
and position only, so a wrong position schedule shows there). A failed control is written to the report
before the run stops; a resume refuses while it is recorded. The report carries it; the summary carries it
when at least one handoff was scored. These limits were
set on a CPU; the first handoff on the card is their first test there.

**Readout, fixed now. Descriptive: no hypothesis row, no sealed prediction.** Per cell, per arm, median with
p10 and p90 over handoffs (lower nearest rank), handoffs weighted equally: each reading above; counts of
handoffs whose decode and whose named set changed; per ladder step, handoffs passing and, among them, those
changed. Between REUSE's mean key deviation and (a) its mean KL, (b) one minus its common share, (c) one minus its named-set overlap: the rank
correlation with a 95% percentile interval from 2,000 resamples of handoffs (the 2.5th and 97.5th
percentiles, lower nearest rank). It "orders" if the interval excludes
zero and the correlation is at least 0.5, "orders weakly" if the interval excludes zero and it is under 0.5,
"does not order" if the interval includes zero. A negative correlation whose interval excludes
zero is "orders in reverse". Beside it, the area under the curve of the key deviation as a
predictor of "the named set changed", reported only if each class holds at least five handoffs; otherwise it
is "undefined, k of N changed". A 384-token decode will differ on nearly every handoff, so each ladder step's
count of changed handoffs is reported beside the median common share and named-set overlap of the handoffs
passing it. With 25 and 35 handoffs these intervals are wide; that is stated with them.

**What is already known.** Keys and predictions on this corpus were read before, outside this record. A probe
of this code read every arm on one handoff, `20241025_composio_swekit/django__django-11066_traj#36`, and FRESH's
decode on two more, before this entry. On that handoff REUSE's mean key
deviation was 0.16, which is 0.077 of that handoff's anchor of 2.08 (it passes the ladder at 1, 0.3 and 0.1 and
fails at 0.03 and 0.01); the KL was 0.24 under REUSE and 7.6 under RANDOM. That probe ran on a CPU under
transformers 5.18.0; the run is pinned to 5.17.0. They stay in their cell. Nothing here is a
blind prediction.

**Expected, stated before the run.** In each cell the median KL under RANDOM is at least ten times the median
under REUSE, at equal key deviation. If so, the same key deviation is compatible with outputs that barely move
and outputs that collapse. If not, that is what is reported.

**Order and stop.** Largest sender first, then by handoff id. Stop on a failed control, a digest mismatch or
an out-of-memory error; keep the scored prefix; resume only with unchanged code, inputs, config and runtime.
Numbers enter the ledger from `run_a --summarize` over the pulled output, never from a log.

**Status.** `[STRETCH]` Designed, not run.
