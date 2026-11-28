"""Run A end to end on a synthetic trajectory, a character tokenizer and a tiny random model, on CPU.
Proves the plumbing (prepare -> run -> summarize, the refusals, the resume), not a claim."""
import json

import numpy as np
import pytest
import torch
from transformers import Qwen3Config, Qwen3ForCausalLM

from passing_check import run_a

CONFIG = """
model = "tiny"
revision = "none"
submission_match = "demo"
seed = 7
chunk_tokens = 8
continuation_tokens = 12
continuation_chunk = 4
decode_tokens = 10
prefix_control_tokens = 6
control_max_logit_error = 0.0005
key_match_relative = 0.001
ladder_fractions = [1.0, 0.1]
bootstrap_reps = 50
auc_min_class = 5
layer0_key_limit = 1e-6
fallback_min_distinct_words = 2
stop_tokens = ["<end>"]
[cells.native]
context_cap = 400
"""


class CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) % 60 + 2 for c in text]

    def decode(self, ids, skip_special_tokens=False):
        return "".join(chr(int(i) + 62) for i in ids)

    def convert_tokens_to_ids(self, tokens):
        return [1 for _ in tokens]

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<user>" + messages[0]["content"] + "<end><assistant>"


def _msg(kind, text):
    return {"id": ["langchain", "schema", "messages", kind], "kwargs": {"content": text, "type": kind[:-7].lower()}}


def _result(text, model):
    return {"generations": [[{"text": text}]], "llm_output": {"model_name": model}, "run": None}


@pytest.fixture
def setup(tmp_path, monkeypatch):
    torch.manual_seed(3)
    cfg = Qwen3Config(vocab_size=64, hidden_size=32, intermediate_size=48, num_hidden_layers=2,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=8, max_position_embeddings=512,
                     attention_dropout=0)
    cfg._attn_implementation = "sdpa"
    model = Qwen3ForCausalLM(cfg).eval()
    monkeypatch.setattr(run_a.AutoTokenizer, "from_pretrained", lambda *a, **k: CharTokenizer())
    monkeypatch.setattr(run_a.AutoConfig, "from_pretrained", lambda *a, **k: model.config)
    monkeypatch.setattr(run_a.AutoModelForCausalLM, "from_pretrained", lambda *a, **k: model)
    traces = tmp_path / "traces"
    sub = traces / "20990101_demo"
    sub.mkdir(parents=True)
    for i, body in enumerate(("the agent edited utils/misc.py and ran git_tree", "it changed a/b.py then foo_bar",
                              "nothing much happened here at all")):
        doc = [[_msg("HumanMessage", "task " + body), _result("working on " + body, "model-a")],
               [[_msg("HumanMessage", "history: working on " + body + ". summarise in points")],
                _result("1. " + body + " and more words follow", "model-b")]]
        (sub / f"task-{i}_traj.json").write_text(json.dumps(doc), encoding="utf-8")
    config = tmp_path / "run-a.toml"
    config.write_text(CONFIG, encoding="utf-8")
    base = ["--config", str(config), "--cell", "native", "--inputs", str(tmp_path / "in")]
    return tmp_path, traces, base


def test_prepare_run_and_summarize_end_to_end(setup):
    tmp_path, traces, base = setup
    out = ["--output", str(tmp_path / "out"), "--device", "cpu"]
    assert run_a.main(["--prepare", "--traces", str(traces), *base]) == 0
    manifest = json.loads((tmp_path / "in" / "manifest.json").read_text())
    assert (manifest["observed"], manifest["included"]) == (3, 3)
    with pytest.raises(ValueError, match="never overwritten"):
        run_a.main(["--prepare", "--traces", str(traces), *base])
    assert run_a.main(["--run", "--limit", "2", *base, *out]) == 0
    with pytest.raises(ValueError, match="--resume"):
        run_a.main(["--run", *base, *out])
    assert run_a.main(["--run", "--resume", *base, *out]) == 0
    assert run_a.main(["--summarize", *base, *out]) == 0
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert report["complete"] and summary["scored"] == summary["registered"] == 3
    assert summary["controls"]["fresh_repeat"] == {"max_logit_error": 0.0, "decode_identical": True}
    assert 0 < summary["controls"]["prefix_copy"]["max_logit_error"] <= 0.0005   # moved, by rounding only
    for row in summary["handoffs"]:
        assert row["random"]["key_mean"] == pytest.approx(row["reuse"]["key_mean"], rel=1e-3)
        assert row["reuse"]["key_mean"] > 0 and row["reuse"]["mean_kl"] >= 0
        assert 0 <= row["reuse"]["agreement_32"] <= 1
        assert row["reuse"]["overlap"] is None and row["random"]["set_changed"] is None    # withheld with the table's
        assert 0 <= row["reuse"]["common_share"] <= 1
    assert [step["fraction_of_anchor"] for step in summary["ladder"]] == [1.0, 0.1]
    assert set(summary["figure"]) == {"mean_kl", "one_minus_common_share", "one_minus_overlap",
                                      "auc_named_set_changed", "auc_decode_changed"}
    manifest = json.loads((tmp_path / "in" / "manifest.json").read_text())
    assert manifest["stop_ids"] == [1, 1][:len(manifest["stop_ids"])] and manifest["template_head"] == 6
    assert summary["controls"]["prefix_copy"]["tokens"] == 6
    assert 0 <= summary["table"]["reuse"]["ended_within_decode"] <= 3 and "ended_within_decode" in summary["fresh"]
    assert summary["controls"]["layer0_reuse_key"] <= 1e-6
    assert summary["controls"]["random_key_ratio"] == pytest.approx(1.0, abs=1e-3)
    assert summary["names_reported"] is False and len(summary["fresh_degenerate"]) == 3   # a random model names nothing
    assert summary["table"]["reuse"]["overlap"] is None and summary["figure"]["one_minus_overlap"] is None
    assert summary["table"]["reuse"]["common_share"] is not None and summary["failed"] is None
    assert summary["table"]["reuse"]["key_p90"]["median"] >= summary["table"]["reuse"]["key_mean"]["p10"]
    assert summary["table"]["reuse"]["key_max"]["median"] >= summary["table"]["reuse"]["key_p90"]["median"]
    assert summary["figure"]["auc_decode_changed"]["auc"] is None            # three handoffs: below the class floor
    assert summary["figure"]["auc_decode_changed"]["of"] == 3
    assert "common_share_among_passing" in summary["ladder"][0]
    assert summary["unrelated_key_anchor"] > 0


def test_a_changed_input_or_output_file_is_refused(setup):
    tmp_path, traces, base = setup
    out = ["--output", str(tmp_path / "out"), "--device", "cpu"]
    run_a.main(["--prepare", "--traces", str(traces), *base])
    run_a.main(["--run", "--limit", "1", *base, *out])
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    scored = tmp_path / "out" / next(iter(report["scores"].values()))["file"]
    scored.write_bytes(scored.read_bytes() + b"x")
    with pytest.raises(ValueError, match="differs from the report"):
        run_a.main(["--summarize", *base, *out])
    first = next(r for r in json.loads((tmp_path / "in" / "manifest.json").read_text())["records"])
    (tmp_path / "in" / first["file"]).write_bytes(b"x")
    with pytest.raises(ValueError, match="differs from the manifest"):
        run_a.main(["--run", "--resume", *base, *out])


def test_rank_correlation_and_auc_on_known_cases():
    assert run_a.ranks([10, 20, 20, 5]).tolist() == [2.0, 3.5, 3.5, 1.0]
    assert run_a.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == pytest.approx(1.0)
    assert run_a.spearman([1, 2, 3, 4], [4, 3, 2, 1]) == pytest.approx(-1.0)
    assert run_a.spearman([1, 2, 3], [5, 5, 5]) is None
    assert run_a.auc([1, 2, 3, 4], [False, False, True, True]) == 1.0
    assert run_a.auc([1, 2, 3, 4], [True, True, False, False]) == 0.0
    assert run_a.auc([1, 1, 1, 1], [True, False, True, False]) == 0.5
    assert run_a.auc([1, 2], [True, True]) is None
    interval = run_a.bootstrap(run_a.spearman, ([1, 2, 3, 4, 5], [1, 2, 3, 5, 4]), np.random.default_rng(0), 100)
    assert interval["low"] <= interval["high"] and 0 < interval["defined"] <= 100


def test_the_registered_wording_follows_the_interval_and_the_correlation():
    span = lambda low, high: {"low": low, "high": high}      # noqa: E731
    assert run_a.ordering(0.7, span(0.4, 0.9)) == "orders"
    assert run_a.ordering(0.5, span(0.1, 0.8)) == "orders"
    assert run_a.ordering(0.4, span(0.1, 0.7)) == "orders weakly"
    assert run_a.ordering(0.7, span(-0.1, 0.9)) == "does not order"
    assert run_a.ordering(0.3, span(0.0, 0.6)) == "does not order"
    assert run_a.ordering(-0.7, span(-0.9, -0.4)) == "orders in reverse"
    assert run_a.ordering(None, span(None, None)) == "undefined"
    assert run_a.ordering(0.7, span(None, None)) == "undefined"


def test_a_recorded_failed_control_blocks_the_resume_and_shows_in_the_summary(setup):
    tmp_path, traces, base = setup
    out = ["--output", str(tmp_path / "out"), "--device", "cpu"]
    run_a.main(["--prepare", "--traces", str(traces), *base])
    run_a.main(["--run", "--limit", "1", *base, *out])
    path = tmp_path / "out" / "report.json"
    report = json.loads(path.read_text())
    report["failed"] = {"handoff": report["run_order"][1], "message": "planted", "controls": {}}
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="a control failed"):
        run_a.main(["--run", "--resume", *base, *out])
    assert run_a.main(["--summarize", *base, *out]) == 0            # the scored prefix is still read
    summary = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert summary["failed"]["message"] == "planted" and summary["complete"] is False and summary["scored"] == 1


def test_a_control_that_fails_is_written_to_the_report_before_the_run_stops(setup):
    tmp_path, traces, base = setup
    config = tmp_path / "run-a.toml"
    config.write_text(config.read_text().replace("control_max_logit_error = 0.0005", "control_max_logit_error = 0.0"))
    out = ["--output", str(tmp_path / "out"), "--device", "cpu"]
    run_a.main(["--prepare", "--traces", str(traces), *base])
    with pytest.raises(ValueError, match="prefix copy"):
        run_a.main(["--run", *base, *out])
    report = json.loads((tmp_path / "out" / "report.json").read_text())
    assert "prefix copy" in report["failed"]["message"] and not report["scores"]
    with pytest.raises(ValueError, match="a control failed"):
        run_a.main(["--run", "--resume", *base, *out])


def test_the_reduced_size_pass_reads_scale_squared_on_the_keys_and_the_same_fresh_decode(setup):
    tmp_path, traces, base = setup
    main_out, scaled_out = tmp_path / "out", tmp_path / "out-scaled"
    scaled = tmp_path / "run-a-scaled.toml"
    scaled.write_text("scales = [0.3, 0.1]\n", encoding="utf-8")
    run_a.main(["--prepare", "--traces", str(traces), *base])
    run_a.main(["--run", *base, "--output", str(main_out), "--device", "cpu"])
    main_summary_args = ["--summarize", *base, "--output", str(main_out)]
    assert run_a.main(main_summary_args) == 0
    run_a.main(["--run", *base, "--output", str(scaled_out), "--device", "cpu", "--scaled", str(scaled)])
    assert run_a.main(["--summarize", *base, "--output", str(scaled_out), "--main", str(main_out)]) == 0
    main_summary = json.loads((main_out / "summary.json").read_text())
    summary = json.loads((scaled_out / "summary.json").read_text())
    assert summary["pass"] == "scaled" and summary["arms"] == ["random_0.3", "random_0.1"]
    with pytest.raises(ValueError, match="pass --main"):
        run_a.main(["--summarize", *base, "--output", str(scaled_out)])
    with pytest.raises(ValueError, match="missing under --main"):
        run_a.main(["--summarize", *base, "--output", str(scaled_out), "--main", str(tmp_path / "nowhere")])
    assert summary["checked_against_main"] and "ladder" not in summary and "figure" not in summary
    by_handoff = {r["handoff"]: r for r in main_summary["handoffs"]}
    for row in summary["handoffs"]:
        reuse = by_handoff[row["handoff"]]["reuse"]["key_mean"]
        assert row["reuse_key_mean"] == pytest.approx(reuse, rel=1e-9)
        assert row["random_0.3"]["key_mean"] == pytest.approx(0.09 * reuse, rel=1e-3)
        assert row["random_0.1"]["key_mean"] == pytest.approx(0.01 * reuse, rel=1e-3)
        # Smaller noise along the same direction moves the predictions less.
        assert row["random_0.1"]["mean_kl"] < row["random_0.3"]["mean_kl"] < by_handoff[row["handoff"]]["random"]["mean_kl"]
        assert row["reuse"]["mean_kl"] == pytest.approx(by_handoff[row["handoff"]]["reuse"]["mean_kl"])
        assert row["reuse"]["common_share"] == pytest.approx(by_handoff[row["handoff"]]["reuse"]["common_share"])
    for reading in ("mean_kl", "one_minus_common_share"):
        assert sum(summary["brackets"][reading].values()) == 3
    assert set(summary["controls"]["scaled_key_ratio"]) == {"random_0.3", "random_0.1"}
    assert summary["names_reported"] is False and summary["fresh"]["names"] is None
    assert all(r["fresh"]["names"] is None and r["random_0.1"]["overlap"] is None for r in summary["handoffs"])
    assert main_summary["fresh"]["names"] is None and main_summary["handoffs"][0]["fresh"]["grounding"] is None
    # A control that fails on the first handoff leaves nothing scored; the summary says so by name.
    report_path = scaled_out / "report.json"
    report = json.loads(report_path.read_text())
    report_path.write_text(json.dumps(report | {"scores": {}, "failed": {"handoff": "h", "message": "planted", "controls": {}}}))
    with pytest.raises(ValueError, match="before any handoff was scored: planted"):
        run_a.main(["--summarize", *base, "--output", str(scaled_out), "--main", str(main_out)])
    report_path.write_text(json.dumps(report))
    scales = [0.1, 0.3, 1.0]
    assert run_a.bracket(0.05, [0.1, 0.5, 2.0], scales) == "below 0.1"
    assert run_a.bracket(0.1, [0.1, 0.5, 2.0], scales) == "0.1 to 0.3"
    assert run_a.bracket(0.7, [0.1, 0.5, 2.0], scales) == "0.3 to 1"
    assert run_a.bracket(2.0, [0.1, 0.5, 2.0], scales) == "at or above 1"
    assert run_a.bracket(0.7, [0.5, 0.1, 2.0], scales) == "not ordered"
    assert run_a.bracket(0.7, [0.5, 0.5, 2.0], scales) == "0.3 to 1"          # tied arms count as ordered
    # A first-layer failure still leaves the key ratio in the report, in either pass.
    config = tmp_path / "run-a.toml"
    strict = tmp_path / "strict.toml"
    strict.write_text(config.read_text().replace("layer0_key_limit = 1e-6", "layer0_key_limit = -1.0"))
    for extra, field in (([], "random_key_ratio"), (["--scaled", str(scaled)], "scaled_key_ratio")):
        strict_in, strict_out = tmp_path / f"in-{field}", tmp_path / f"out-{field}"
        strict_base = ["--config", str(strict), "--cell", "native", "--inputs", str(strict_in)]
        run_a.main(["--prepare", "--traces", str(traces), *strict_base])
        with pytest.raises(ValueError, match="first-layer keys"):
            run_a.main(["--run", *strict_base, "--output", str(strict_out), "--device", "cpu", *extra])
        failed = json.loads((strict_out / "report.json").read_text())["failed"]
        assert field in failed["controls"] and "layer0_reuse_key" in failed["controls"]
        assert str(failed["controls"]) in failed["message"]
    # A scaled pass cannot be resumed as a two-arm run, and the reverse.
    with pytest.raises(ValueError, match="resume refused"):
        run_a.main(["--run", "--resume", *base, "--output", str(scaled_out), "--device", "cpu"])
    # A fresh decode that differs from the two-arm run's is refused.
    other = tmp_path / "other"
    other.mkdir()
    for path in main_out.glob("*.npz"):
        with np.load(path) as arrays:
            changed = {k: arrays[k] for k in arrays.files}
        changed["fresh_decode"] = changed["fresh_decode"] + 1
        np.savez_compressed(other / path.name, **changed)
    with pytest.raises(ValueError, match="fresh decode differs"):
        run_a.main(["--summarize", *base, "--output", str(scaled_out), "--main", str(other)])
