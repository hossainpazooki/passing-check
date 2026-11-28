"""Run A: at each handoff, read the reader under a fresh cache, a reused cache and a random cache of the
same size on the key deviation, all on the same prompt.

    python -m passing_check.run_a --prepare   --config C --cell NAME --traces DIR --inputs DIR
    python -m passing_check.run_a --run       --config C --cell NAME --inputs DIR --output DIR [--limit N] [--resume]
    python -m passing_check.run_a --summarize --config C --cell NAME --inputs DIR --output DIR
    python -m passing_check.run_a --run       ... --output DIR2 --scaled S       (the reduced-size pass)
    python -m passing_check.run_a --summarize ... --output DIR2 --main DIR

`--prepare` is CPU only: it finds the handoffs, tokenizes them, aligns sender and receiver, and writes one
input file per handoff with a manifest of digests. `--run` reads nothing but those files. `--summarize`
recomputes every reported number from the files `--run` wrote; no number is carried from the run's own log.

Arms, per handoff, all against FRESH (the reader's own prefill of the receiver prompt R):
  REUSE   the reader's states for the matched tokens, taken from its prefill of the sender context S and
          moved to their positions in R; every other token of R computed against the cache so assembled.
  RANDOM  the fresh states of the matched tokens plus isotropic noise whose size per token, layer and head
          is REUSE's error. It reads the same as REUSE on the key deviation, by construction.
The reduced-size pass reads RANDOM again with its noise shrunk by each scale in S (the same draw, so the
same direction), against the same FRESH: what a small perturbation does to the same readings. It needs no
output of the two-arm run and changes none; its summary checks its fresh decode against that run's.
Controls, on the first handoff of the run order (the largest sender): a second FRESH pass, which must
reproduce the first exactly, and a copy of the reader's own states for a prefix of R, their keys moved to
position 0 and back by the same relocation REUSE uses, which must stay within `control_max_logit_error`.

The receiver prompt is R wrapped in the reader's own chat template as one user message, with the reply
opened and thinking switched off, so the reader answers R's request. The template's tokens are never
matched, so every arm computes the end of the prompt itself and decodes from the bare prompt.

Readings under each arm: the key deviation of the matched tokens; on the recorded response, teacher-forced,
the per-token KL from FRESH and whether the top token agrees; and a greedy decode of a fixed length, which
the summary cuts at the first end-of-turn token and reads without template tokens.
"""
import argparse
import gc
import hashlib
import json
import os
import subprocess
import tomllib
from pathlib import Path

import numpy as np
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

from passing_check import REPO_ROOT
from passing_check.cache import (compare_logits, construct, continuation_logits, greedy, key_deviation,
                                 matched, prefill, random_candidate, relocate, relocated_candidate,
                                 unrelated_candidate, validate_pairs)
from passing_check.names import (agreement, common_share, cut_at, distinct_words, first_difference,
                                 grounding, jaccard, named)
from passing_check.stats import make_rng, quantile, summary
from passing_check.traces import load_handoffs, matching_pairs

ARMS = ("reuse", "random")


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write(path, body):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(body, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def git_commit():
    """The commit of the tree the run was started from; None outside a git checkout (an unpacked copy)."""
    done = subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True)
    return done.stdout.strip() if done.returncode == 0 else None


def load_config(path, cell):
    cfg = tomllib.loads(Path(path).read_text(encoding="utf-8"))
    if cell not in cfg["cells"]:
        raise ValueError(f"no cell {cell!r} in {path}")
    return cfg, cfg["cells"][cell]


def stem_of(handoff_id):
    return handoff_id.replace("/", "__").replace("#", "_sw")


def prepare(args, cfg, cell):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    encode = lambda text: tokenizer.encode(text, add_special_tokens=False)      # noqa: E731
    mark = "\x00RECEIVER\x00"
    before, after = tokenizer.apply_chat_template([{"role": "user", "content": mark}], tokenize=False,
                                                  add_generation_prompt=True, enable_thinking=False).split(mark)
    head, tail = encode(before), encode(after)
    if not tail:
        raise ValueError("the chat template adds nothing after the user message; the prompt's end would be copied")
    submissions = sorted(p for p in Path(args.traces).iterdir() if p.is_dir() and cfg["submission_match"] in p.name)
    if not submissions:
        raise ValueError(f"no submission folder matching {cfg['submission_match']!r} under {args.traces}")
    args.inputs.mkdir(parents=True, exist_ok=True)
    if (args.inputs / "manifest.json").exists():
        raise ValueError("a prepared manifest already exists here; it is never overwritten")
    records = []
    room = cfg["continuation_tokens"] + cfg["decode_tokens"]
    for h in load_handoffs(submissions):
        sender, body = encode(h.sender_text), encode(h.receiver_text)
        receiver = head + body + tail
        continuation = encode(h.response_text)[:cfg["continuation_tokens"]]
        row = {"handoff": h.handoff_id, "sender_model": h.sender_model, "receiver_model": h.receiver_model,
               "sender_tokens": len(sender), "receiver_tokens": len(receiver),
               "continuation_tokens": len(continuation), "excluded": True, "reason": None, "file": None}
        if not body:
            row["reason"] = "the receiver prompt is empty in the trace"
        elif len(sender) > cell["context_cap"] or len(receiver) + room > cell["context_cap"]:
            row["reason"] = f"sender, or receiver plus continuation, exceeds the cell's cap {cell['context_cap']}"
        elif max(len(sender), len(receiver)) <= cell.get("context_floor", 0):
            row["reason"] = f"sender and receiver both within the cell's floor {cell['context_floor']}"
        elif len(continuation) < 2:
            row["reason"] = "fewer than two tokens of recorded response"
        else:
            pairs = matching_pairs(sender, body)
            pairs[:, 1] += len(head)          # positions in the wrapped prompt; the template is never matched
            if len(pairs) < 2:
                row["reason"] = "fewer than two matched tokens"
            else:
                sender, receiver = np.asarray(sender, np.int64), np.asarray(receiver, np.int64)
                validate_pairs(sender, receiver, pairs)
                path = args.inputs / f"{stem_of(h.handoff_id)}.npz"
                np.savez_compressed(path, sender=sender, receiver=receiver, pairs=pairs,
                                    continuation=np.asarray(continuation, np.int64))
                row.update(excluded=False, file=path.name, sha256=digest(path), matched_tokens=int(len(pairs)))
        records.append(row)
    stop_ids = sorted(int(i) for i in tokenizer.convert_tokens_to_ids(cfg["stop_tokens"]))
    manifest = {"config_sha256": digest(args.config), "cell": args.cell, "model": cfg["model"],
                "revision": cfg["revision"], "template_head": len(head), "template_tail": len(tail),
                "stop_ids": stop_ids, "observed": len(records),
                "included": sum(not r["excluded"] for r in records), "records": records}
    write(args.inputs / "manifest.json", manifest)
    print(f"prepared {manifest['included']} of {manifest['observed']} handoffs for cell {args.cell}")


def checked_manifest(args):
    manifest = json.loads((args.inputs / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != digest(args.config) or manifest["cell"] != args.cell:
        raise ValueError("the configuration or the cell changed after preparation")
    for row in manifest["records"]:
        if not row["excluded"] and digest(args.inputs / row["file"]) != row["sha256"]:
            raise ValueError(f"input file differs from the manifest: {row['file']}")
    return manifest


def run_order(manifest):
    # The largest sender first, as the memory probe and the home of the controls; then by handoff id.
    rows = sorted((r for r in manifest["records"] if not r["excluded"]), key=lambda r: r["handoff"])
    if not rows:
        raise ValueError("the prepared cell holds no included handoff")
    largest = max(rows, key=lambda r: r["sender_tokens"])
    return [largest] + [r for r in rows if r is not largest]


def load_model(cfg, cell, device):
    model_cfg = AutoConfig.from_pretrained(cfg["model"], revision=cfg["revision"])
    if "rope" in cell:
        model_cfg.rope_parameters = {**model_cfg.rope_parameters, **cell["rope"]}
        model_cfg.max_position_embeddings = cell["context_cap"]
    model = AutoModelForCausalLM.from_pretrained(cfg["model"], revision=cfg["revision"], config=model_cfg,
                                                 dtype=torch.float32, attn_implementation="sdpa")
    return model.eval().to(device)


def scaled_scales(args):
    """The noise scales of the reduced-size pass; () for the two-arm run."""
    if args.scaled is None:
        return ()
    scales = tuple(tomllib.loads(Path(args.scaled).read_text(encoding="utf-8"))["scales"])
    if not scales or any(not 0 < s < 1 for s in scales) or len(set(scales)) != len(scales):
        raise ValueError("scales must be distinct and lie strictly between 0 and 1")
    return scales


def bracket(value, references, scales):
    """Where REUSE's value falls against the same handoff's noise arms, given smallest noise first:
    "below 0.1", "0.1 to 0.3", "0.3 to 1" or "at or above 1" for scales (0.1, 0.3, 1). "not ordered" when
    an arm's own value is below a smaller arm's, so no bracket exists; equal arms count as ordered."""
    if any(later < earlier for earlier, later in zip(references, references[1:])):
        return "not ordered"
    place = sum(value >= reference for reference in references)
    if place == 0:
        return f"below {scales[0]:g}"
    if place == len(scales):
        return f"at or above {scales[-1]:g}"
    return f"{scales[place - 1]:g} to {scales[place]:g}"


def read_arm(model, cache, prompt_logits, continuation, cfg):
    """Teacher-forced logits for every token of the recorded response (the first from the logits after the
    prompt), then the cache cut back to the prompt and a greedy decode from the bare prompt."""
    logits = torch.cat((prompt_logits[None],
                        continuation_logits(model, cache, continuation, cfg["continuation_chunk"])))
    cache.crop(-(len(continuation) - 1))
    return logits, greedy(model, cache, prompt_logits, cfg["decode_tokens"])


def run(args, cfg, cell, manifest):
    device = torch.device(args.device)
    if device.type == "cuda":
        # Set before any CUDA tensor exists; deterministic cuBLAS refuses GEMMs otherwise.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.use_deterministic_algorithms(True)
    scales = scaled_scales(args)
    arms = tuple(f"random_{s:g}" for s in scales) if scales else ARMS
    model = load_model(cfg, cell, device)
    freq = model.model.rotary_emb.inv_freq.detach().cpu()
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    identity = {"config_sha256": digest(args.config), "manifest_sha256": digest(args.inputs / "manifest.json"),
                "code_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                "scaled_sha256": digest(args.scaled) if scales else None,
                "transformers": transformers.__version__, "torch": torch.__version__, "device": str(device),
                "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
    order = run_order(manifest)
    if report_path.exists():
        if not args.resume:
            raise ValueError("the output holds a run; pass --resume to verify and continue it")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if report["identity"] != identity:
            raise ValueError("resume refused: code, inputs, configuration, runtime or device differ")
        if report.get("failed"):
            raise ValueError(f"resume refused: a control failed on {report['failed']['handoff']} and the run "
                             "stopped there; a failed control is not passed over")
    else:
        report = {"identity": identity, "cell": args.cell, "arms": list(arms),
                  "git_commit": git_commit(),
                  "inv_freq_sha256": hashlib.sha256(freq.numpy().tobytes()).hexdigest(),
                  "run_order": [r["handoff"] for r in order], "scores": {}, "complete": False}
        write(report_path, report)
    for index, record in enumerate(order[:args.limit] if args.limit else order):
        hid = record["handoff"]
        if hid in report["scores"]:
            continue
        with np.load(args.inputs / record["file"], allow_pickle=False) as arrays:
            sender, receiver, pairs, continuation = (arrays[k] for k in ("sender", "receiver", "pairs", "continuation"))
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        cache, at_end = prefill(model, receiver, cfg["chunk_tokens"], last_logits=True)
        fresh_states = matched(cache, pairs[:, 1])
        fresh_logits, fresh_decode = read_arm(model, cache, at_end, continuation, cfg)
        del cache
        controls = {}

        def stop(message):
            # A failed control is written before the run stops; a resume refuses while it is recorded.
            report["failed"] = {"handoff": hid, "message": message, "controls": controls}
            write(report_path, report)
            raise ValueError(message)
        if index == 0:
            repeat_logits, repeat_decode = read_arm(
                model, *prefill(model, receiver, cfg["chunk_tokens"], last_logits=True), continuation, cfg)
            controls["fresh_repeat"] = {"max_logit_error": compare_logits(fresh_logits, repeat_logits)["max_logit_error"],
                                        "decode_identical": bool(np.array_equal(fresh_decode, repeat_decode))}
            prefix_n = min(cfg["prefix_control_tokens"], len(receiver) - 1)
            prefix_pairs = np.column_stack((np.arange(prefix_n), np.arange(prefix_n)))
            prefix_states = matched(prefill(model, receiver[:prefix_n], cfg["chunk_tokens"]), np.arange(prefix_n))
            # Out to position 0 and back: the copy then also passes through the relocation arithmetic.
            there, zero = np.arange(prefix_n), np.zeros(prefix_n, dtype=np.int64)
            prefix_states = [(relocate(relocate(k, there, zero, freq), zero, there, freq), v) for k, v in prefix_states]
            copied = construct(model, receiver, prefix_pairs, prefix_states, cfg["chunk_tokens"])
            copy_logits, _ = read_arm(model, *copied, continuation, cfg)
            controls["prefix_copy"] = {"max_logit_error": compare_logits(fresh_logits, copy_logits)["max_logit_error"],
                                       "tokens": int(prefix_n)}
            del copied, prefix_states
            if controls["fresh_repeat"]["max_logit_error"] != 0 or not controls["fresh_repeat"]["decode_identical"]:
                stop(f"the fresh repeat is not exact: {controls}")
            if controls["prefix_copy"]["max_logit_error"] > cfg["control_max_logit_error"]:
                stop(f"the prefix copy moved the output: {controls}")
        source = matched(prefill(model, sender, cfg["chunk_tokens"]), pairs[:, 0])
        reused = relocated_candidate(source, pairs, freq)
        del source
        if scales:
            # The two-arm run's own draw for this handoff, shrunk: each arm lies along RANDOM's direction.
            candidates = {arm: random_candidate(fresh_states, reused, make_rng(cfg["seed"] + index), s)
                          for arm, s in zip(arms, scales)}
        else:
            candidates = {"reuse": reused, "random": random_candidate(fresh_states, reused, make_rng(cfg["seed"] + index))}
        # The unrelated-token anchor: each matched token's fresh key against another matched token's.
        shifted = unrelated_candidate(fresh_states, pairs, freq)
        outputs = {"continuation": continuation, "fresh_decode": fresh_decode,
                   "unrelated_key": key_deviation(fresh_states, shifted, pairs, freq).mean((1, 2))}
        del shifted
        deviation = {arm: key_deviation(fresh_states, candidates[arm], pairs, freq) for arm in arms}
        reuse_deviation = deviation["reuse"] if "reuse" in deviation else key_deviation(fresh_states, reused, pairs, freq)
        # Two controls on every handoff, before any output is read under an arm. A first-layer key depends on
        # its token and its position only, so with position removed REUSE's must equal FRESH's; a wrong rotary
        # schedule shows here. And RANDOM must read as REUSE on the key deviation, or the contrast is void.
        controls["layer0_reuse_key"] = float(reuse_deviation[:, 0].max())
        if scales:
            # A scaled arm must read scale squared times REUSE on the key deviation: 1.0 here means it does.
            controls["scaled_key_ratio"] = {arm: float(deviation[arm].mean() / reuse_deviation.mean() / s ** 2)
                                            for arm, s in zip(arms, scales)}
            ratios = list(controls["scaled_key_ratio"].values())
        else:
            controls["random_key_ratio"] = float(deviation["random"].mean() / deviation["reuse"].mean())
            ratios = [controls["random_key_ratio"]]
        if controls["layer0_reuse_key"] > cfg["layer0_key_limit"]:
            stop(f"REUSE's first-layer keys differ from FRESH's after position is removed: {controls}")
        if any(abs(ratio - 1) > cfg["key_match_relative"] for ratio in ratios):
            stop(("the random arm does not match the reused cache on the key deviation" if not scales else
                  "a scaled arm is not its scale squared of the reused cache on the key deviation") + f": {controls}")
        if scales:
            outputs["reuse_key"] = reuse_deviation.mean((1, 2))
        for arm in arms:
            outputs[f"{arm}_key"] = deviation[arm].mean((1, 2))
            cache, at_end = construct(model, receiver, pairs, candidates[arm], cfg["chunk_tokens"])
            logits, decoded = read_arm(model, cache, at_end, continuation, cfg)
            compared = compare_logits(fresh_logits, logits)
            outputs.update({f"{arm}_kl": compared["kl"], f"{arm}_top1_agrees": compared["top1_agrees"],
                            f"{arm}_decode": decoded})
            del cache, logits
            gc.collect()
        destination = args.output / record["file"]
        np.savez_compressed(destination, **outputs)
        report["scores"][hid] = {"file": destination.name, "sha256": digest(destination), "controls": controls,
                                 "peak_allocated_GiB": torch.cuda.max_memory_allocated(device) / 2**30
                                 if device.type == "cuda" else None}
        report["complete"] = len(report["scores"]) == len(order)
        write(report_path, report)
        print(f"{len(report['scores'])}/{len(order)} {hid}", flush=True)
        del candidates, reused, fresh_states, fresh_logits


def ranks(values):
    """Average ranks, 1-based, ties sharing the mean of their positions."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="stable")
    out = np.empty(len(values))
    i = 0
    while i < len(values):
        j = i
        while j + 1 < len(values) and values[order[j + 1]] == values[order[i]]:
            j += 1
        out[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return out


def spearman(x, y):
    """Rank correlation; None when either side is constant."""
    rx, ry = ranks(x), ranks(y)
    if rx.std() == 0 or ry.std() == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def auc(score, positive):
    """Probability that a positive case outscores a negative one, ties counting half; None without both."""
    score, positive = np.asarray(score, dtype=np.float64), np.asarray(positive, dtype=bool)
    n_pos, n_neg = int(positive.sum()), int((~positive).sum())
    if not n_pos or not n_neg:
        return None
    return float((ranks(score)[positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def bootstrap(fn, columns, rng, reps):
    """Percentile interval of fn over resampled rows; resamples where fn is undefined are dropped and counted."""
    n = len(columns[0])
    values = []
    for _ in range(reps):
        rows = rng.integers(0, n, n)
        value = fn(*(np.asarray(c)[rows] for c in columns))
        if value is not None:
            values.append(value)
    if not values:
        return {"low": None, "high": None, "defined": 0, "reps": reps}
    return {"low": quantile(values, 0.025), "high": quantile(values, 0.975), "defined": len(values), "reps": reps}


def ordering(rho, interval):
    """The registered wording for a rank correlation and its percentile interval."""
    if rho is None or interval["low"] is None:
        return "undefined"
    if interval["low"] <= 0 <= interval["high"]:
        return "does not order"
    if rho < 0:
        return "orders in reverse"
    return "orders" if rho >= 0.5 else "orders weakly"


def summarize(args, cfg, manifest):
    report = json.loads((args.output / "report.json").read_text(encoding="utf-8"))
    if report["identity"]["manifest_sha256"] != digest(args.inputs / "manifest.json"):
        raise ValueError("the report was run on a different prepared manifest")
    order = [r["handoff"] for r in run_order(manifest)]
    scored = list(report["scores"])
    if report.get("failed") and not scored:
        # Nothing was scored, so there is no prefix to read; say why before refusing.
        raise ValueError(f"a control failed on {report['failed']['handoff']} before any handoff was scored: "
                         f"{report['failed']['message']}")
    if report["run_order"] != order or scored != order[:len(scored)] or not scored:
        raise ValueError("the scored handoffs must be a non-empty prefix of the prepared run order")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model"], revision=cfg["revision"])
    stop_ids = set(manifest["stop_ids"])
    arms = tuple(report.get("arms", ARMS))
    if "reuse" not in arms and args.main is None:
        raise ValueError("a reduced-size pass is summarized against the two-arm run's output: pass --main")
    rows = []
    for hid in scored:
        result = report["scores"][hid]
        if digest(args.output / result["file"]) != result["sha256"]:
            raise ValueError(f"output file differs from the report: {result['file']}")
        with np.load(args.output / result["file"], allow_pickle=False) as out, \
                np.load(args.inputs / result["file"], allow_pickle=False) as inp:
            # R itself, without the chat template around it: a name is grounded in the text summarised.
            body = inp["receiver"][manifest["template_head"]:len(inp["receiver"]) - manifest["template_tail"]]
            prompt = tokenizer.decode(body, skip_special_tokens=True)
            n = len(out["continuation"])
            if args.main is not None:
                # The reduced-size pass is read against the same fresh decode as the two-arm run, or not at all.
                if not (args.main / result["file"]).exists():
                    raise ValueError(f"the two-arm run's output for {hid} is missing under --main")
                with np.load(args.main / result["file"], allow_pickle=False) as main_out:
                    if not np.array_equal(main_out["fresh_decode"], out["fresh_decode"]):
                        raise ValueError(f"the fresh decode differs from the two-arm run's: {hid}")
                    main_read = {arm: (float(main_out[f"{arm}_kl"].mean()), cut_at(main_out[f"{arm}_decode"], stop_ids))
                                 for arm in ARMS}
            # Every decode is read up to its first end-of-turn token.
            fresh_ids = cut_at(out["fresh_decode"], stop_ids)
            fresh_text = tokenizer.decode(fresh_ids, skip_special_tokens=True)
            fresh_names = named(fresh_text)
            row = {"handoff": hid, "matched": int(len(inp["pairs"])), "scored_tokens": n,
                   "unrelated_key_mean": float(out["unrelated_key"].mean()),
                   "fresh": {"decode_tokens": len(fresh_ids), "ended": len(fresh_ids) < len(out["fresh_decode"]),
                             "distinct_words": distinct_words(fresh_text),
                             "names": len(fresh_names), "grounding": grounding(fresh_names, prompt)}}
            for arm in arms:
                kl, top1, key = out[f"{arm}_kl"], out[f"{arm}_top1_agrees"], out[f"{arm}_key"]
                if kl.shape != (n,) or top1.shape != (n,) or not np.isfinite(kl).all() or np.any(kl < 0):
                    raise ValueError(f"invalid per-token readings: {hid} {arm}")
                ids = cut_at(out[f"{arm}_decode"], stop_ids)
                names = named(tokenizer.decode(ids, skip_special_tokens=True))
                row[arm] = {"key_mean": float(key.mean()), "key_p90": quantile(key.tolist(), 0.9),
                            "key_max": float(key.max()), "mean_kl": float(kl.mean()),
                            "top1_agreement": float(top1.mean()),
                            "first_difference": first_difference(fresh_ids, ids),
                            "agreement_32": agreement(fresh_ids, ids, min(32, cfg["decode_tokens"])),
                            "common_share": common_share(fresh_ids, ids),
                            "identical": fresh_ids == ids, "decode_tokens": len(ids),
                            "ended": len(ids) < len(out[f"{arm}_decode"]), "names": len(names),
                            "overlap": jaccard(fresh_names, names), "set_changed": names != fresh_names,
                            "grounding": grounding(names, prompt)}
            # Checked by the run before any output was read; checked again here from the stored readings.
            if "reuse" in arms:
                if abs(row["random"]["key_mean"] - row["reuse"]["key_mean"]) > cfg["key_match_relative"] * row["reuse"]["key_mean"]:
                    raise ValueError(f"the random arm does not match the reused cache on the key deviation: {hid}")
            else:
                row["reuse_key_mean"] = float(out["reuse_key"].mean())
                for arm in arms:
                    expected = float(arm.removeprefix("random_")) ** 2 * row["reuse_key_mean"]
                    if abs(row[arm]["key_mean"] - expected) > cfg["key_match_relative"] * expected:
                        raise ValueError(f"a scaled arm is not its scale squared of REUSE on the key deviation: {hid} {arm}")
                if args.main is not None:
                    # REUSE against this handoff's own noise arms, smallest first, RANDOM (scale 1) last.
                    by_scale = sorted(arms, key=lambda a: float(a.removeprefix("random_")))
                    sizes = [float(a.removeprefix("random_")) for a in by_scale] + [1.0]
                    row["reuse"] = {"mean_kl": main_read["reuse"][0],
                                    "common_share": common_share(fresh_ids, main_read["reuse"][1])}
                    row["random"] = {"mean_kl": main_read["random"][0],
                                     "common_share": common_share(fresh_ids, main_read["random"][1])}
                    row["bracket"] = {
                        "mean_kl": bracket(row["reuse"]["mean_kl"],
                                           [row[a]["mean_kl"] for a in by_scale] + [row["random"]["mean_kl"]], sizes),
                        "one_minus_common_share": bracket(
                            1 - row["reuse"]["common_share"],
                            [1 - row[a]["common_share"] for a in by_scale] + [1 - row["random"]["common_share"]], sizes)}
            rows.append(row)
    # The fallback: where FRESH itself writes too little to name anything, the names readings are not reported.
    degenerate = [r["handoff"] for r in rows if r["fresh"]["distinct_words"] < cfg["fallback_min_distinct_words"]
                  or not r["fresh"]["names"]]
    names_reported = len(degenerate) * 4 <= len(rows)

    def column(arm, field):
        values = [r[arm][field] for r in rows if r[arm][field] is not None]
        return summary(values) if values else None
    table = {arm: {field: column(arm, field) for field in
                   ("key_mean", "key_p90", "key_max", "mean_kl", "top1_agreement", "agreement_32", "common_share",
                    "overlap", "grounding", "names")}
             | {"decode_changed": sum(not r[arm]["identical"] for r in rows),
                "ended_within_decode": sum(r[arm]["ended"] for r in rows),
                "named_set_changed": sum(r[arm]["set_changed"] for r in rows),
                "first_difference_where_changed": summary([r[arm]["first_difference"] for r in rows
                                                           if not r[arm]["identical"]])
                if any(not r[arm]["identical"] for r in rows) else None}
             for arm in arms}
    fresh = {"grounding": summary([r["fresh"]["grounding"] for r in rows if r["fresh"]["grounding"] is not None])
             if any(r["fresh"]["grounding"] is not None for r in rows) else None,
             "names": summary([r["fresh"]["names"] for r in rows]),
             "ended_within_decode": sum(r["fresh"]["ended"] for r in rows)}
    if "reuse" not in arms:
        # The reduced-size pass: the same readings under smaller noise, and nothing that needs REUSE's outputs.
        if not names_reported:
            for arm in arms:
                for field in ("overlap", "grounding", "names", "named_set_changed"):
                    table[arm][field] = None
                for r in rows:
                    for field in ("overlap", "grounding", "names", "set_changed"):
                        r[arm][field] = None
            fresh["names"] = fresh["grounding"] = None
            for r in rows:
                r["fresh"]["names"] = r["fresh"]["grounding"] = None
        body = {"cell": report["cell"], "pass": "scaled", "arms": list(arms), "complete": report["complete"],
                "failed": report.get("failed"),
                "scored": len(rows), "registered": len(order), "names_reported": names_reported,
                "fresh_degenerate": degenerate, "checked_against_main": args.main is not None,
                "controls": report["scores"][scored[0]]["controls"],
                "reuse_key_mean": summary([r["reuse_key_mean"] for r in rows]),
                # Per reading, the number of handoffs in each bracket; None without the two-arm run's output.
                "brackets": {reading: {label: sum(r["bracket"][reading] == label for r in rows)
                                       for label in sorted({r["bracket"][reading] for r in rows})}
                             for reading in ("mean_kl", "one_minus_common_share")} if args.main is not None else None,
                "fresh": fresh, "table": table, "handoffs": rows}
        write(args.output / "summary.json", body)
        return body
    anchor = float(np.median([r["unrelated_key_mean"] for r in rows]))

    def passing(f):
        return [r for r in rows if r["reuse"]["key_mean"] <= f * anchor]
    ladder = [{"fraction_of_anchor": f, "tolerance": f * anchor,
               "passing": sum(r["reuse"]["key_mean"] <= f * anchor for r in rows),
               "common_share_among_passing": summary([r["reuse"]["common_share"] for r in passing(f)])
               if passing(f) else None,
               "overlap_among_passing": summary([r["reuse"]["overlap"] for r in passing(f)]) if passing(f) else None,
               "passing_with_named_set_changed": sum(r["reuse"]["key_mean"] <= f * anchor and r["reuse"]["set_changed"]
                                                     for r in rows),
               "passing_with_decode_changed": sum(r["reuse"]["key_mean"] <= f * anchor and not r["reuse"]["identical"]
                                                  for r in rows)}
              for f in cfg["ladder_fractions"]]
    rng = make_rng(cfg["seed"])
    key = [r["reuse"]["key_mean"] for r in rows]
    figure = {}
    for name, outcome in (("mean_kl", [r["reuse"]["mean_kl"] for r in rows]),
                          ("one_minus_common_share", [1 - r["reuse"]["common_share"] for r in rows]),
                          ("one_minus_overlap", [1 - r["reuse"]["overlap"] for r in rows])):
        rho, interval = spearman(key, outcome), bootstrap(spearman, (key, outcome), rng, cfg["bootstrap_reps"])
        figure[name] = {"spearman": rho, "interval": interval, "reading": ordering(rho, interval)}
    for name, positive in (("named_set_changed", [r["reuse"]["set_changed"] for r in rows]),
                           ("decode_changed", [not r["reuse"]["identical"] for r in rows])):
        changed, least = sum(positive), cfg["auc_min_class"]
        if min(changed, len(rows) - changed) < least:
            # Too few handoffs on one side for the number to mean anything; say how many changed instead.
            figure[f"auc_{name}"] = {"auc": None, "changed": changed, "of": len(rows),
                                     "reason": f"fewer than {least} handoffs in one class"}
        else:
            figure[f"auc_{name}"] = {"auc": auc(key, positive), "changed": changed, "of": len(rows),
                                     "interval": bootstrap(auc, (key, positive), rng, cfg["bootstrap_reps"])}

    if not names_reported:
        for arm in ARMS:
            for field in ("overlap", "grounding", "names", "named_set_changed"):
                table[arm][field] = None
        for step in ladder:
            step["overlap_among_passing"] = step["passing_with_named_set_changed"] = None
        figure["one_minus_overlap"] = figure["auc_named_set_changed"] = None
        # FRESH's own names are withheld with the arms': the fallback says the names readings are not reported.
        fresh["names"] = fresh["grounding"] = None
        for r in rows:
            r["fresh"]["names"] = r["fresh"]["grounding"] = None
        for r in rows:
            for arm in ARMS:
                for field in ("overlap", "grounding", "names", "set_changed"):
                    r[arm][field] = None
    body = {"cell": report["cell"], "complete": report["complete"], "failed": report.get("failed"),
            "scored": len(rows), "registered": len(order),
            "names_reported": names_reported, "fresh_degenerate": degenerate,
            "observed": manifest["observed"], "excluded": [r for r in manifest["records"] if r["excluded"]],
            "controls": report["scores"][scored[0]]["controls"], "unrelated_key_anchor": anchor,
            "fresh": fresh, "table": table, "ladder": ladder, "figure": figure, "handoffs": rows}
    write(args.output / "summary.json", body)
    return body


def main(argv=None):
    parser = argparse.ArgumentParser(prog="python -m passing_check.run_a")
    action = parser.add_mutually_exclusive_group(required=True)
    for name in ("prepare", "run", "summarize"):
        action.add_argument(f"--{name}", action="store_true")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--cell", required=True)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--traces", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--limit", type=int, default=0, help="score only the first N handoffs of the run order")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--scaled", type=Path, help="with --run: read the reduced-size RANDOM arms this file lists")
    parser.add_argument("--main", type=Path, help="with --summarize of a scaled pass: the two-arm run's output")
    args = parser.parse_args(argv)
    cfg, cell = load_config(args.config, args.cell)
    if args.prepare:
        if args.traces is None:
            parser.error("--prepare requires --traces")
        prepare(args, cfg, cell)
        return 0
    if args.output is None:
        parser.error("--run and --summarize require --output")
    manifest = checked_manifest(args)
    if args.run:
        run(args, cfg, cell, manifest)
    else:
        body = summarize(args, cfg, manifest)
        print(json.dumps({k: body[k] for k in ("cell", "scored", "registered", "failed", "controls", "table", "ladder",
                                               "figure", "brackets") if k in body},
                         indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
