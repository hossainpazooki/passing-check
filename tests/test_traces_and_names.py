"""The trace loader on a synthetic trajectory, and the readings taken on decoded text."""
import json

import numpy as np
import pytest

from passing_check.names import (agreement, common_share, cut_at, distinct_words, first_difference, grounding,
                                 jaccard, named)
from passing_check.stats import make_rng, quantile, summary
from passing_check.traces import handoffs_from, load_handoffs, load_trajectory, matching_pairs


def _msg(kind, text):
    return {"id": ["langchain", "schema", "messages", kind], "kwargs": {"content": text, "type": kind.lower()}}


def _result(text, model):
    return {"generations": [[{"text": text}]], "llm_output": {"model_name": model}, "run": None}


def _write(tmp_path, doc):
    sub = tmp_path / "20990101_demo"
    sub.mkdir()
    (sub / "task-1_traj.json").write_text(json.dumps(doc), encoding="utf-8")
    return sub


def test_a_change_of_serving_model_is_one_handoff_with_the_receivers_own_prompt(tmp_path):
    doc = [
        [_msg("SystemMessage", "sys"), _msg("HumanMessage", "do it"), _result("did A", "model-a")],
        [[_msg("HumanMessage", "sys"), _msg("HumanMessage", "did A"), _msg("HumanMessage", "summarise")],
         _result("summary", "model-b")],
        [_msg("HumanMessage", "more"), _result("did B", "model-b")],
    ]
    sub = _write(tmp_path, doc)
    traj, texts, is_response = load_trajectory(sub / "task-1_traj.json", sub.name)
    assert len(traj.messages) == len(texts) == len(is_response) == 9 and sum(is_response) == 3
    (h,) = load_handoffs([sub])
    assert h.handoff_id == "20990101_demo/task-1_traj#2"    # turn 0 is the system message, 1 and 2 the responses
    assert (h.sender_model, h.receiver_model) == ("model-a", "model-b")
    assert h.sender_text == "sys\ndo it\ndid A"
    assert h.receiver_text == "sys\ndid A\nsummarise"       # the nested prompt list is flattened
    assert h.response_text == "summary"
    assert handoffs_from(traj, texts, is_response) == [h]


def test_a_change_of_model_that_lands_on_prompt_history_is_not_a_handoff(tmp_path):
    ai = {"id": ["langchain", "schema", "messages", "AIMessage"],
          "kwargs": {"content": "old turn", "type": "ai", "response_metadata": {"model_id": "model-a"}}}
    doc = [[_msg("HumanMessage", "q"), _result("answer", "model-b")], [ai, _msg("HumanMessage", "next")]]
    sub = _write(tmp_path, doc)
    assert load_handoffs([sub]) == []


def test_a_node_of_unknown_role_is_refused(tmp_path):
    odd = {"id": ["langchain", "schema", "messages", "FunctionMessage"], "kwargs": {"content": "x"}}
    sub = _write(tmp_path, [[odd, _result("a", "m")]])
    with pytest.raises(ValueError, match="unknown role"):
        load_trajectory(sub / "task-1_traj.json", sub.name)


def test_the_loader_refuses_shapes_it_does_not_know(tmp_path):
    sub = _write(tmp_path, {"not": "a list"})
    with pytest.raises(ValueError):
        load_trajectory(sub / "task-1_traj.json", sub.name)
    (sub / "task-1_traj.json").write_text("[[1, 2]]", encoding="utf-8")
    with pytest.raises(ValueError):
        load_trajectory(sub / "task-1_traj.json", sub.name)


def test_matching_pairs_are_the_matching_blocks_token_by_token():
    pairs = matching_pairs([1, 2, 3, 4, 5], [1, 2, 9, 3, 4, 5])
    assert pairs.tolist() == [[0, 0], [1, 1], [2, 3], [3, 4], [4, 5]]
    assert matching_pairs([1], [2]).shape == (0, 2) and matching_pairs([1], [2]).dtype == np.int64


def test_names_are_backticked_paths_and_identifiers_and_prose_is_never_searched():
    text = ("The agent edited `astropy/utils/misc.py`, changed `InheritDocstrings.__init__` and called "
            "`inspect.isfunction(x)`; e.g. it ran `git_tree`, saw file(s) and tests.Then it set `using=db` "
            "and `db`. <|im_end|> im_start plain_word")
    assert named(text) == {"astropy/utils/misc.py", "InheritDocstrings.__init__", "inspect.isfunction", "git_tree"}
    assert named("The agent reviewed the issue and proposed a fix.") == frozenset()
    assert named("`save()` and `Model`") == {"save", "Model"}


def test_overlap_grounding_distinct_words_and_the_cut():
    a, b = named("see `a/b.py` and `foo_bar`"), named("see `a/b.py` and `baz_qux`")
    assert jaccard(a, b) == pytest.approx(1 / 3) and jaccard(a, a) == 1.0
    assert jaccard(frozenset(), frozenset()) == 1.0 and jaccard(a, frozenset()) == 0.0
    assert grounding(a, "prompt mentions a/b.py only") == 0.5
    assert grounding(frozenset(), "anything") is None
    assert distinct_words("The the THE cat, cat_2!") == 3
    assert cut_at([5, 6, 1, 7], {1}) == [5, 6] and cut_at([5, 6], {1}) == [5, 6] and cut_at([1], {1}) == []


def test_first_difference_and_agreement():
    assert first_difference([1, 2, 3], [1, 2, 3]) == -1
    assert first_difference([1, 2, 3], [1, 9, 3]) == 1
    assert first_difference([1, 2], [1, 2, 3]) == 2
    assert agreement([1, 2, 3, 4], [1, 2, 9, 4], 4) == 0.75
    assert agreement([1, 2], [1, 2], 4) == 0.5          # a missing position is a disagreement


def test_the_common_share_survives_an_inserted_token():
    a = [1, 2, 3, 4, 5, 6]
    assert common_share(a, a) == 1.0
    assert agreement(a, [9] + a[:5], 6) == 0.0          # one inserted token zeroes positional agreement
    assert common_share(a, [9] + a[:5]) == pytest.approx(5 / 6)
    assert common_share([1, 2, 3], [4, 5, 6]) == 0.0
    assert common_share([], []) == 1.0 and common_share([1], []) == 0.0


def test_the_quantile_convention_and_the_seeded_generator():
    assert quantile([3, 1, 2, 4], 0.9) == 4 and quantile([3, 1, 2, 4], 0.1) == 1
    assert summary([1, 2, 3, 4]) == {"n": 4, "median": 2.5, "p10": 1, "p90": 4}
    with pytest.raises(ValueError):
        summary([])
    with pytest.raises(TypeError):
        make_rng(True)
    assert make_rng(5).integers(100) == make_rng(5).integers(100)
